"""R6b: eastern birth rate as a function of LOCAL colonisation age, with the clock carried inside the scan.

Setup as R6 / grad_inverse.py (step 3, habitat frozen at the 1966-2025 mean, release only, loss = East BBS log-lik
under the fit's own observation model, L-BFGS with exact JAX gradients). Fmax x fscale everywhere, x g(E) in East
cells, where E is the soft years-occupied clock of clock_forward.py (threshold: 1% of the reference arm's modern
occupied median, unit-free). Compare with R6 'cal' (Fmax x m(calendar year), same parameter count).
Usage (inside sbx_env.sh): python r6b_clock.py --out DIR [--maxiter 200] [--scale .5] [--check]
"""
import argparse, json, os, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from scipy.optimize import minimize
from harness import Sandbox, capture, edit_habitat, edit_args, region_masks, c_of, ap
import clock_forward as cf

AGE_KNOTS = [0., 2., 5., 10., 15., 25., 40., 85.]
CAL_KNOTS = [1940, 1955, 1965, 1975, 1985, 1995, 2010, 2025]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--maxiter', type=int, default=200)
    p.add_argument('--scale', type=float, default=.5); p.add_argument('--fscale0', type=float, default=.4)
    p.add_argument('--check', action='store_true')
    p.add_argument('--cal', action='store_true', help='also an East calendar multiplier m(t) (CAL_KNOTS): AGE + CAL')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    ref = sb.run(dict(name='fitted'))
    ref_nll = -sum(v['loglik'] for k, v in ref['abundance'].items() if k.startswith('east'))
    thr = .01 * sb.base_med
    d = edit_habitat(sb.data, sb.years, 0., [1966, 2025])
    x0, fwd0 = capture(d, sb.lat)
    if x0.get('occupancy_b') is not None:
        raise ValueError('clock fork needs occupancy off (the clock uses the unused psi slot)')
    x0 = edit_args(x0, dict(region='east_only'), sb.lon, sb.years, sb.rows, sb.cols)
    fwd = cf.build()
    mask = np.zeros(np.asarray(x0['land_mask']).shape, bool); mask[sb.rows, sb.cols] = region_masks(sb.lon)['east']
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    east_rows = jnp.asarray(np.flatnonzero(sb.obs_ok)[sb.region['east'][sb.o_cell]])
    y = jnp.asarray(sb.data['observed_results'])
    log_thr = float(np.log(thr))
    na = len(AGE_KNOTS); nc = len(CAL_KNOTS) if a.cal else 0
    Wc = jnp.asarray(np.stack([np.interp(sb.years.astype(float), CAL_KNOTS, np.eye(len(CAL_KNOTS))[i])
                               for i in range(len(CAL_KNOTS))], 1))
    east_land = jnp.asarray(region_masks(sb.lon)['east'])
    shape = np.asarray(x0['Fmax_flat']).shape

    def set_clock(vals):
        cf.CLOCK.update(active=True, knots=jnp.asarray(AGE_KNOTS), values=vals, mask=jnp.asarray(mask),
                        log_thr=log_thr, scale=a.scale)

    def args(theta):
        x = dict(x0); F = jnp.asarray(x0['Fmax_flat']) * jnp.exp(theta[-1])
        if nc:
            m = Wc @ jnp.exp(theta[na:na + nc])
            F = F * jnp.where(east_land[None, :], m[:, None], 1.).reshape(shape)
        Sa, Sj = jnp.asarray(x['Sa_flat']), jnp.asarray(x['Sj_flat'])
        x.update(Fmax_flat=F, c_flat=c_of(F, Sa, Sj, x))
        return x

    def neg_ll(theta):
        set_clock(jnp.exp(theta[:na]))
        out = fwd(**args(theta))
        original = ap.forward_sim_age_structured
        ap.forward_sim_age_structured = lambda *aa, **kw: out
        try:
            tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=lat),
                                              rng_seed=104)).get_trace(data=d, prior_scale=1.)
        finally:
            ap.forward_sim_age_structured = original
        return -jnp.sum(tr['obs']['fn'].log_prob(y)[east_rows])

    vg = jax.jit(jax.value_and_grad(neg_ll))
    log = (a.out / ('r6b_clock_cal.jsonl' if nc else 'r6b_clock.jsonl')).open('w')
    # identity check: clock multiplier 1 everywhere vs the original forward (c recomputed: differences ~1e-8 rel)
    th1 = np.r_[np.zeros(na + nc), 0.]
    set_clock(jnp.ones(len(AGE_KNOTS))); o1 = fwd(**args(th1)); cf.CLOCK['active'] = False
    o0 = fwd0(**args(th1))
    ident = float(jnp.max(jnp.abs(o1[0] - o0[0])) / jnp.max(jnp.abs(o0[0])))
    rec = dict(arm='fitted', nll=ref_nll, threshold_model_units=thr, identity_max_rel_diff=ident)
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n')
    theta0 = np.r_[np.zeros(na + nc), np.log(a.fscale0)]
    t0 = time.time(); v0, g0 = vg(jnp.asarray(theta0))
    print(json.dumps(dict(first_eval_sec=round(time.time() - t0, 1), nll0=float(v0))), flush=True)
    if a.check:
        for i in (1, 4, len(theta0) - 1):
            h = 1e-4; e = np.zeros_like(theta0); e[i] = h
            fd = (float(vg(jnp.asarray(theta0 + e))[0]) - float(vg(jnp.asarray(theta0 - e))[0])) / (2 * h)
            r = dict(check=i, autodiff=float(g0[i]), finite_diff=fd); print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + '\n')
    n = [0]
    def f(theta):
        v, g = vg(jnp.asarray(theta)); v, g = float(v), np.asarray(g, float); n[0] += 1
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            v, g = 1e12, np.zeros_like(theta)
        log.write(json.dumps(dict(eval=n[0], theta=list(map(float, theta)), nll=v,
                                  grad_norm=float(np.linalg.norm(g)), sec=round(time.time() - t0, 1))) + '\n')
        log.flush()
        return v, g
    res = minimize(f, theta0, jac=True, method='L-BFGS-B', bounds=[(-4., 3.)] * len(theta0),
                   options=dict(maxiter=a.maxiter))
    th = res.x
    # full readout through the harness with the fork installed as the model's forward
    set_clock(jnp.exp(jnp.asarray(th[:na])))
    orig = ap.forward_sim_age_structured; ap.forward_sim_age_structured = fwd
    try:
        arm = dict(name='r6b_clock', region='east_only', habitat_alpha=0, fmax_scale=float(np.exp(th[-1])))
        if nc:
            m = np.asarray(Wc) @ np.exp(th[na:na + nc])
            arm['rate_array'] = dict(Fmax=np.where(np.asarray(east_land)[None, :], m[:, None], 1.))
        best = sb.run(arm)
    finally:
        ap.forward_sim_age_structured = orig; cf.CLOCK['active'] = False
    readout_nll = -sum(v['loglik'] for k, v in best['abundance'].items() if k.startswith('east'))
    out = dict(knots=AGE_KNOTS, scale=a.scale, fitted_reference_nll=ref_nll, best_nll=float(res.fun),
               readout_nll=readout_nll, converged=bool(res.success), message=str(res.message), n_eval=n[0],
               knot_multiplier=list(np.exp(th[:na])), fmax_scale=float(np.exp(th[-1])),
               effective_by_age=list(np.exp(th[:na]) * np.exp(th[-1])),
               cal_knots=CAL_KNOTS if nc else None, cal_multiplier=list(np.exp(th[na:na + nc])) if nc else None,
               minutes=round((time.time() - t0) / 60, 1), sha=best['final_state_sha256'][:12],
               offsets=best['curves']['rel_10pct']['arrival_offset']['by_cohort'],
               east_obs_over_expected={k: v['obs_over_expected'] for k, v in best['abundance'].items()
                                       if k.startswith('east')},
               east_zero_share={k: [v['zero_share_obs'], v['zero_share_pred']] for k, v in best['abundance'].items()
                                if k.startswith('east')})
    (a.out / ('r6b_clock_cal_summary.json' if nc else 'r6b_clock_summary.json')).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()

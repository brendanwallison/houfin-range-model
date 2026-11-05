"""R5c (gradient version): which TIME SHAPE does a fixed habitat need, scored in the fit's own BBS terms.

Same question as inverse.py, rebuilt for speed: habitat frozen at its 1966-2025 mean, release only (West removed), a
piecewise-linear multiplier m(t) (KNOTS, log-parameterised) on ONE eastern vital-rate array (K or Fmax) plus a constant
Fmax scale. Loss = minus the East BBS log-likelihood under the fit's own observation model (route / observer / year
effects, NB2), i.e. no unit conversion and no hand weights; arrival timing is carried by the zero / positive rows.
JAX autodiff through the forward scan and the observation model gives the exact gradient, so L-BFGS needs tens of
evaluations instead of Nelder-Mead's hundreds. The best point is replayed through Sandbox.run for the full readout.
Usage (inside sbx_env.sh): python grad_inverse.py --field K|Fmax --out DIR [--maxiter 80] [--check]
"""
import argparse, json, os, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from scipy.optimize import minimize
from harness import Sandbox, capture, edit_habitat, edit_args, region_masks, c_of, ap
from inverse import KNOTS, arm_for


def build_objective(sb, field):
    d = edit_habitat(sb.data, sb.years, 0., [1966, 2025])
    x0, fwd = capture(d, sb.lat)
    x0 = edit_args(x0, dict(region='east_only'), sb.lon, sb.years, sb.rows, sb.cols)
    T, N = len(sb.years), len(sb.lon)
    W = jnp.asarray(np.stack([np.interp(sb.years.astype(float), KNOTS, np.eye(len(KNOTS))[i])
                              for i in range(len(KNOTS))], 1))                  # (T, knots) linear interpolation
    east = jnp.asarray(region_masks(sb.lon)['east'])
    shape = np.asarray(x0[f'{field}_flat']).shape
    east_rows = jnp.asarray(np.flatnonzero(sb.obs_ok)[sb.region['east'][sb.o_cell]])
    y = jnp.asarray(sb.data["observed_results"])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}   # numpy latents break indexing by traced arrays under jit

    def neg_ll(theta):
        m = W @ jnp.exp(theta[:len(KNOTS)])                                        # (T,)
        mult = jnp.where(east[None, :], m[:, None], 1.).reshape(shape)
        x = dict(x0)
        x[f'{field}_flat'] = jnp.asarray(x0[f'{field}_flat']) * mult
        F = jnp.asarray(x['Fmax_flat']) * jnp.exp(theta[-1])
        Sa = jnp.minimum(jnp.asarray(x['Sa_flat']), .995); Sj = jnp.minimum(jnp.asarray(x['Sj_flat']), Sa)
        x.update(Fmax_flat=F, Sa_flat=Sa, Sj_flat=Sj, c_flat=c_of(F, Sa, Sj, x))
        out = fwd(**x)
        original = ap.forward_sim_age_structured
        ap.forward_sim_age_structured = lambda *aa, **kw: out
        try:
            tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=lat),
                                              rng_seed=104)).get_trace(data=d, prior_scale=1.)
        finally:
            ap.forward_sim_age_structured = original
        return -jnp.sum(tr['obs']['fn'].log_prob(y)[east_rows])

    return jax.jit(jax.value_and_grad(neg_ll))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--field', default='K'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--maxiter', type=int, default=80); p.add_argument('--fscale0', type=float, default=.4)
    p.add_argument('--check', action='store_true', help='finite-difference check of two gradient coordinates')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    ref = sb.run(dict(name='fitted'))
    ref_nll = -sum(v['loglik'] for k, v in ref['abundance'].items() if k.startswith('east'))
    vg = build_objective(sb, a.field)
    theta0 = np.r_[np.zeros(len(KNOTS)), np.log(a.fscale0)]
    t0 = time.time(); v0, g0 = vg(jnp.asarray(theta0)); jax.block_until_ready(g0)
    print(json.dumps(dict(first_eval_sec=round(time.time() - t0, 1), nll0=float(v0), ref_nll=ref_nll)), flush=True)
    log = (a.out / f'grad_{a.field}.jsonl').open('w')
    log.write(json.dumps(dict(arm='fitted', nll=ref_nll)) + '\n')
    if a.check:
        for i in (2, len(theta0) - 1):
            h = 1e-4; e = np.zeros_like(theta0); e[i] = h
            fd = (float(vg(jnp.asarray(theta0 + e))[0]) - float(vg(jnp.asarray(theta0 - e))[0])) / (2 * h)
            rec = dict(check=i, autodiff=float(g0[i]), finite_diff=fd)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n')
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
    best = sb.run(arm_for(th, a.field))   # full readout (offsets, obs / expected by era) at the optimum
    out = dict(field=a.field, knots=KNOTS, fitted_reference_nll=ref_nll, best_nll=float(res.fun),
               best_multiplier=list(np.exp(th[:len(KNOTS)])), best_fmax_scale=float(np.exp(th[-1])),
               converged=bool(res.success), message=str(res.message), n_eval=n[0],
               minutes=round((time.time() - t0) / 60, 1), sha=best['final_state_sha256'][:12],
               offsets=best['curves']['rel_10pct']['arrival_offset']['by_cohort'],
               east_obs_over_expected={k: v['obs_over_expected'] for k, v in best['abundance'].items()
                                       if k.startswith('east')},
               east_zero_share={k: [v['zero_share_obs'], v['zero_share_pred']] for k, v in best['abundance'].items()
                                if k.startswith('east')})
    (a.out / f'grad_{a.field}_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()

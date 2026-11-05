"""R23: one innovation per cell, active only while the cell is on the wavefront (user: "a version that can elegantly
target the wavefront?").

Standard readouts on frozen habitat; optionally (--k_real) capacity K from the real-Z capture (the R21 / A9 decoupled
structure: Z's change handles abundance). Fecundity in a cell is multiplied by exp(sigma u_c exp(-E/tau) exp(-(N/K)/rho))
(frontinnov_forward.py; E = soft years-occupied clock, threshold 1% of the reference modern occupied median). Fitted: u
(one per land cell, prior N(0, 1)) and the constant Fmax scale; objective = BBS -log-lik (25-49N rows) + 0.5 |u|^2.
L-BFGS with exact gradients, GPU float64. Readouts: score, offsets, never reached, u summary by region (East / Plains /
West mean, sd, share |u| > 1), and the u map (npz) to see WHERE corrections are wanted.
Usage (inside sbx_env.sh): python frontinnov.py --out DIR [--sigma 1 --tau 10 --rho .3 --k_real --maxiter 200]
"""
import argparse, json, math, os, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from scipy.optimize import minimize
from harness import Sandbox, capture, edit_habitat, c_of, ap
import frontinnov_forward as fif

OPT_SCALE = 1e4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--sigma', type=float, default=1.)
    p.add_argument('--tau', type=float, default=10.); p.add_argument('--rho', type=float, default=.3)
    p.add_argument('--k_real', action='store_true'); p.add_argument('--maxiter', type=int, default=200)
    p.add_argument('--tag', default=''); p.add_argument('--tight', action='store_true')
    p.add_argument('--gate', choices=['ratio', 'allee'], default='ratio',
                   help='ratio: exp(-(N/K)/rho) (R23); allee: absolute Allee mate-finding deficit (R23b)')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    ref = sb.run(dict(name='fitted'))
    ref_nll = -sum(v['loglik'] for v in ref['abundance'].values())
    d = edit_habitat(sb.data, sb.years, 0., [1966, 2025])
    x0, fwd = capture(d, sb.lat)
    if x0.get('occupancy_b') is not None:
        raise ValueError('front-innovation fork needs occupancy off (the clock uses the psi slot)')
    if a.k_real:
        x_real, _ = capture(sb.data, sb.lat); x0 = dict(x0); x0['K_flat'] = jnp.asarray(x_real['K_flat'])
    fwd_fi = fif.build()
    gshape = tuple(np.asarray(x0['land_mask']).shape); N = len(sb.rows)
    thr = .01 * sb.base_med
    band = sb.region['east'] | sb.region['plains'] | sb.region['west']
    rows = jnp.asarray(np.flatnonzero(sb.obs_ok)[band[sb.o_cell]])
    y = jnp.asarray(sb.data['observed_results'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}

    def set_fi(u):
        fif.FI.update(active=True, grid=jnp.zeros(gshape).at[sb.rows, sb.cols].set(u), sigma=a.sigma, tau=a.tau,
                      rho=a.rho, log_thr=float(np.log(thr)), scale=.5, gate=a.gate)

    def args(th):
        x = dict(x0); F = jnp.asarray(x0['Fmax_flat']) * jnp.exp(th[-1])
        x.update(Fmax_flat=F, c_flat=c_of(F, jnp.asarray(x['Sa_flat']), jnp.asarray(x['Sj_flat']), x))
        return x

    def neg_obj(th):
        set_fi(th[:-1]); out = fwd_fi(**args(th)); fif.FI['active'] = False
        original = ap.forward_sim_age_structured
        ap.forward_sim_age_structured = lambda *aa, **kw: out
        try:
            tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=lat),
                                              rng_seed=104)).get_trace(data=d, prior_scale=1.)
        finally:
            ap.forward_sim_age_structured = original
        return -jnp.sum(tr['obs']['fn'].log_prob(y)[rows]) + 0.5 * jnp.sum(th[:-1] ** 2)

    # identity: u = 0 reproduces the original forward
    th0 = np.zeros(N + 1)
    set_fi(jnp.zeros(N)); o1 = fwd_fi(**args(jnp.asarray(th0))); fif.FI['active'] = False
    o0 = fwd(**args(jnp.asarray(th0)))
    ident = float(jnp.max(jnp.abs(o1[0] - o0[0])) / jnp.max(jnp.abs(o0[0])))
    vg = jax.jit(jax.value_and_grad(neg_obj))
    t0 = time.time(); v0, _ = vg(jnp.asarray(th0))
    log = (a.out / f'frontinnov{a.tag}.jsonl').open('w')
    rec = dict(identity_max_rel=ident, start=float(v0), fitted_reference_nll=ref_nll, n_u=N, sigma=a.sigma, tau=a.tau,
               rho=a.rho, gate=a.gate, k_real=a.k_real, first_eval_sec=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n')
    n = [0]
    def f(th):
        v, g = vg(jnp.asarray(th)); v, g = float(v), np.asarray(g, float); n[0] += 1
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            v, g = 1e12, np.zeros_like(th)
        log.write(json.dumps(dict(eval=n[0], obj=v, u_rms=float(np.sqrt(np.mean(th[:-1] ** 2))), sec=round(time.time() - t0, 1))) + '\n')
        log.flush()
        return v / OPT_SCALE, g / OPT_SCALE
    res = minimize(f, th0, jac=True, method='L-BFGS-B', bounds=[(-6., 6.)] * N + [(-4., 3.)],
                   options=dict(maxiter=a.maxiter, **(dict(ftol=1e-15, gtol=1e-12) if a.tight else {})))
    th = res.x; u = th[:-1]
    set_fi(jnp.asarray(u)); orig = ap.forward_sim_age_structured; ap.forward_sim_age_structured = fwd_fi
    try:
        best = sb.run(dict(name=f'frontinnov{a.tag}', habitat_alpha=0, fmax_scale=float(np.exp(th[-1])),
                           **({'forward_args': {'K_flat': x0['K_flat']}} if a.k_real else {})))
    finally:
        ap.forward_sim_age_structured = orig; fif.FI['active'] = False
    ab = best['abundance']; readout = -sum(v['loglik'] for v in ab.values())
    by = {reg: dict(mean=float(u[sb.region[reg]].mean()), sd=float(u[sb.region[reg]].std()),
                    share_abs_gt1=float((np.abs(u[sb.region[reg]]) > 1).mean()),
                    mean_lon_weighted=None) for reg in ('east', 'plains', 'west')}
    out = dict(sigma=a.sigma, tau=a.tau, rho=a.rho, gate=a.gate, k_real=a.k_real, identity_max_rel=ident, start=float(v0),
               end_obj=float(res.fun) * OPT_SCALE, readout_nll=readout, prior_term=float(.5 * np.sum(u ** 2)),
               fitted_reference_nll=ref_nll,
               gap_closed=float((float(v0) - readout) / (float(v0) - ref_nll)) if float(v0) > ref_nll else None,
               converged=bool(res.success), n_eval=n[0], minutes=round((time.time() - t0) / 60, 1),
               fmax_scale=float(np.exp(th[-1])), u_by_region=by,
               offsets=best['curves']['rel_10pct']['arrival_offset']['by_cohort'],
               never_reached=best['curves']['rel_10pct']['arrival_offset']['never'],
               loglik_by={k: v['loglik'] for k, v in ab.items()})
    np.savez_compressed(a.out / f'frontinnov{a.tag}_u.npz', u=u, lon=sb.lon, lat=sb.latd)
    (a.out / f'frontinnov{a.tag}_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('identity_max_rel', 'start', 'readout_nll', 'prior_term', 'gap_closed', 'converged',
                                          'n_eval', 'fmax_scale', 'u_by_region', 'offsets', 'never_reached')}), flush=True)


if __name__ == '__main__':
    main()

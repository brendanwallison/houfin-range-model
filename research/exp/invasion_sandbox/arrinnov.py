"""R24: innovations on ARRIVALS' settlement success at the Allee knife's edge (user's option 2, 2026-11-03).

With the model's simple dispersal a fraction of the population reaches everywhere; under the base's ABSOLUTE
mate-finding Allee that trickle mostly fails to breed, so whether it establishes sits on a knife's edge. Here the
juvenile arrivals in a cell are multiplied by exp(sigma u D), D = the Allee mate-finding deficit at the cell's absolute
density (arrinnov_forward.py): the correction bites only where arrivals land below the Allee knee and fades as the
cell fills. --per cell: one u per land cell; --per cellyear: one u per cell-year in --w0..--w1 (u stays ~0 wherever D
is ~0, so settled cell-years cost nothing). Prior u ~ N(0, 1).
Standard readouts on frozen habitat; --k_real takes K from the real-Z capture (the R21 / A9 decoupled structure).
Fitted: u and the constant Fmax scale; objective = BBS -log-lik (25-49N rows) + 0.5 |u|^2. L-BFGS, exact gradients.
Readouts: score, gap closed, offsets, never reached, u by region (x era for cellyear), u map (npz).
Usage (inside sbx_env.sh): python arrinnov.py --out DIR [--per cell|cellyear --sigma 1 --k_real --maxiter 200]
"""
import argparse, json, os, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from scipy.optimize import minimize
from harness import Sandbox, capture, edit_habitat, c_of, ap
import arrinnov_forward as aif

OPT_SCALE = 1e4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--sigma', type=float, default=1.)
    p.add_argument('--per', choices=['cell', 'cellyear'], default='cell')
    p.add_argument('--w0', type=int, default=1951); p.add_argument('--w1', type=int, default=2020)
    p.add_argument('--k_real', action='store_true'); p.add_argument('--maxiter', type=int, default=200)
    p.add_argument('--tag', default=''); p.add_argument('--tight', action='store_true')
    p.add_argument('--agg_k', default='base', help='Allee aggregation k: base (overlay), none (Poisson) or a number')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    ref = sb.run(dict(name='fitted'))
    ref_nll = -sum(v['loglik'] for v in ref['abundance'].values())
    if a.agg_k != 'base':   # R25: Allee SHAPE (same slope at 0; k small = gentler knee). Read at trace time by allee_mated_fraction
        from src.model import age_forward as _af
        _af.ALLEE_AGG_K = None if a.agg_k == 'none' else float(a.agg_k)
    d = edit_habitat(sb.data, sb.years, 0., [1966, 2025])
    x0, fwd = capture(d, sb.lat)
    if a.k_real:
        x_real, _ = capture(sb.data, sb.lat); x0 = dict(x0); x0['K_flat'] = jnp.asarray(x_real['K_flat'])
    fwd_fa = aif.build()
    gshape = tuple(np.asarray(x0['land_mask']).shape); N = len(sb.rows); T = len(sb.years)
    ia = np.flatnonzero((sb.years >= a.w0) & (sb.years <= a.w1)); nA = ia.size
    n_u = N if a.per == 'cell' else nA * N
    band = sb.region['east'] | sb.region['plains'] | sb.region['west']
    rows = jnp.asarray(np.flatnonzero(sb.obs_ok)[band[sb.o_cell]])
    y = jnp.asarray(sb.data['observed_results'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    dt = jnp.asarray(x0['Sa_flat']).dtype

    def set_fa(u):
        if a.per == 'cell':
            aif.FA.update(active=True, grid=jnp.zeros(gshape, dt).at[sb.rows, sb.cols].set(u), grid_t=None, sigma=a.sigma)
        else:
            g = jnp.zeros((T,) + gshape, dt).at[jnp.asarray(ia)[:, None], sb.rows[None, :], sb.cols[None, :]].set(
                u.reshape(nA, N))
            aif.FA.update(active=True, grid=None, grid_t=g, sigma=a.sigma)

    def args(th):
        x = dict(x0); F = jnp.asarray(x0['Fmax_flat']) * jnp.exp(th[-1])
        x.update(Fmax_flat=F, c_flat=c_of(F, jnp.asarray(x['Sa_flat']), jnp.asarray(x['Sj_flat']), x))
        return x

    def neg_obj(th):
        set_fa(th[:-1]); out = fwd_fa(**args(th)); aif.FA['active'] = False
        original = ap.forward_sim_age_structured
        ap.forward_sim_age_structured = lambda *aa, **kw: out
        try:
            tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=lat),
                                              rng_seed=104)).get_trace(data=d, prior_scale=1.)
        finally:
            ap.forward_sim_age_structured = original
        return -jnp.sum(tr['obs']['fn'].log_prob(y)[rows]) + 0.5 * jnp.sum(th[:-1] ** 2)

    th0 = np.zeros(n_u + 1)
    set_fa(jnp.zeros(n_u, dt)); o1 = fwd_fa(**args(jnp.asarray(th0))); aif.FA['active'] = False
    o0 = fwd(**args(jnp.asarray(th0)))
    ident = float(jnp.max(jnp.abs(o1[0] - o0[0])) / jnp.max(jnp.abs(o0[0])))
    vg = jax.jit(jax.value_and_grad(neg_obj))
    t0 = time.time(); v0, _ = vg(jnp.asarray(th0))
    log = (a.out / f'arrinnov_{a.per}{a.tag}.jsonl').open('w')
    rec = dict(identity_max_rel=ident, start=float(v0), fitted_reference_nll=ref_nll, per=a.per, n_u=n_u, agg_k=a.agg_k, agg_k_effective=str(__import__('src.model.age_forward', fromlist=['x']).ALLEE_AGG_K),
               sigma=a.sigma, k_real=a.k_real, window=[a.w0, a.w1], first_eval_sec=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n')
    n = [0]
    def f(th):
        v, g = vg(jnp.asarray(th)); v, g = float(v), np.asarray(g, float); n[0] += 1
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            v, g = 1e12, np.zeros_like(th)
        log.write(json.dumps(dict(eval=n[0], obj=v, u_rms=float(np.sqrt(np.mean(th[:-1] ** 2))), sec=round(time.time() - t0, 1))) + '\n')
        log.flush()
        return v / OPT_SCALE, g / OPT_SCALE
    res = minimize(f, th0, jac=True, method='L-BFGS-B', bounds=[(-8., 8.)] * n_u + [(-4., 3.)],
                   options=dict(maxiter=a.maxiter, **(dict(ftol=1e-15, gtol=1e-12) if a.tight else {})))
    th = res.x; u = th[:-1]
    set_fa(jnp.asarray(u, dt)); orig = ap.forward_sim_age_structured; ap.forward_sim_age_structured = fwd_fa
    try:
        best = sb.run(dict(name=f'arrinnov_{a.per}{a.tag}', habitat_alpha=0, fmax_scale=float(np.exp(th[-1])),
                           **({'forward_args': {'K_flat': x0['K_flat']}} if a.k_real else {})))
    finally:
        ap.forward_sim_age_structured = orig; aif.FA['active'] = False
    ab = best['abundance']; readout = -sum(v['loglik'] for v in ab.values())
    regs = ('east', 'plains', 'west')
    if a.per == 'cell':
        by = {r: dict(mean=float(u[sb.region[r]].mean()), sd=float(u[sb.region[r]].std()),
                      share_abs_gt1=float((np.abs(u[sb.region[r]]) > 1).mean())) for r in regs}
    else:
        U = u.reshape(nA, N); yrs = sb.years[ia]; by = {}
        for r in regs:
            for e0, e1 in ((1951, 1965), (1966, 1980), (1981, 1995), (1996, 2020)):
                m = (yrs >= e0) & (yrs <= e1)
                if m.any():
                    blk = U[m][:, sb.region[r]]
                    by[f'{r}_{e0}-{e1}'] = dict(mean=float(blk.mean()), share_abs_gt_half=float((np.abs(blk) > .5).mean()),
                                                min=float(blk.min()), max=float(blk.max()))
    out = dict(per=a.per, agg_k=a.agg_k, sigma=a.sigma, k_real=a.k_real, window=[a.w0, a.w1], identity_max_rel=ident, start=float(v0),
               end_obj=float(res.fun) * OPT_SCALE, readout_nll=readout, prior_term=float(.5 * np.sum(u ** 2)),
               fitted_reference_nll=ref_nll,
               gap_closed=float((float(v0) - readout) / (float(v0) - ref_nll)) if float(v0) > ref_nll else None,
               converged=bool(res.success), n_eval=n[0], minutes=round((time.time() - t0) / 60, 1),
               fmax_scale=float(np.exp(th[-1])), u_by_region=by,
               offsets=best['curves']['rel_10pct']['arrival_offset']['by_cohort'],
               never_reached=best['curves']['rel_10pct']['arrival_offset']['never'],
               loglik_by={k: v['loglik'] for k, v in ab.items()})
    np.savez_compressed(a.out / f'arrinnov_{a.per}{a.tag}_u.npz', u=u.astype(np.float32), lon=sb.lon, lat=sb.latd,
                        years=sb.years[ia])
    (a.out / f'arrinnov_{a.per}{a.tag}_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('identity_max_rel', 'start', 'readout_nll', 'prior_term', 'gap_closed', 'converged',
                                          'n_eval', 'fmax_scale', 'offsets', 'never_reached')}), flush=True)


if __name__ == '__main__':
    main()

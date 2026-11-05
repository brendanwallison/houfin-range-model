"""R22: independent cell-year ABUNDANCE innovations with a looser prior -- how much do they close, and WHERE do they go?

User question (2026-11-03), after Codex's TACC fits found independent cell-year innovations (1966-2020, prior SD .5% /
1%) barely used: would a looser prior do better? Sandbox test on the current base, standard readouts on FROZEN habitat:
the forward's own process_noise argument multiplies both age classes after the annual transition,
    N <- N exp(eps),  eps(t, cell) = sigma u  (median-one; see eps_of),  u ~ N(0, 1) iid per active cell-year,
active window --w0 .. --w1 (default 1951-2020, i.e. from this base's initialisation; Codex used 1966-2020), zero
elsewhere. Exact zeros stay zero (innovations cannot create birds ahead of a front). Fitted: u (all active cell-years)
and the constant Fmax scale; objective = BBS -log-lik (25-49N rows) + 0.5 |u|^2. GPU float64.
Readouts: gap closed (frozen -> fit), offsets, never reached; and where the innovations go -- mean u and mean |eps| by
region x era x cell state (front N/K < .1, filling, near K), and the accumulated log adjustment East / West.
Usage (inside sbx_env.sh): python innov.py --sigma S --out DIR [--w0 1951 --w1 2020 --maxiter 200]
"""
import argparse, json, os, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from scipy.optimize import minimize
from harness import Sandbox, capture, edit_habitat, c_of, to_land, ap

OPT_SCALE = 1e4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sigma', type=float, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--w0', type=int, default=1951); p.add_argument('--w1', type=int, default=2020)
    p.add_argument('--maxiter', type=int, default=200); p.add_argument('--alpha', type=float, default=0.)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    ref = sb.run(dict(name='fitted'))
    ref_nll = -sum(v['loglik'] for v in ref['abundance'].values())
    d = edit_habitat(sb.data, sb.years, a.alpha, [1966, 2025]) if a.alpha != 1 else sb.data
    x0, fwd = capture(d, sb.lat)
    if 'process_noise' not in x0:
        raise ValueError('forward has no process_noise argument')
    T, N = len(sb.years), len(sb.rows)
    act = (sb.years >= a.w0) & (sb.years <= a.w1); ia = np.flatnonzero(act); nA = ia.size
    band = sb.region['east'] | sb.region['plains'] | sb.region['west']
    rows = jnp.asarray(np.flatnonzero(sb.obs_ok)[band[sb.o_cell]])
    y = jnp.asarray(sb.data['observed_results'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    dt = jnp.asarray(x0['Sa_flat']).dtype

    def eps_of(u):
        e = jnp.zeros((T, N), dt)
        # MEDIAN-one (no -sigma^2/2): mean-one centring adds a systematic sigma^2/2 per-year decline that compounds over
        # decades (70 yr at sigma .1: -.35 log; at .3: -3.15) -- negligible at Codex's sigma .005-.01, a brake here
        return e.at[jnp.asarray(ia)].set(a.sigma * u.reshape(nA, N))

    def args(th):
        x = dict(x0); F = jnp.asarray(x0['Fmax_flat']) * jnp.exp(th[-1])
        x.update(Fmax_flat=F, c_flat=c_of(F, jnp.asarray(x['Sa_flat']), jnp.asarray(x['Sj_flat']), x),
                 process_noise=eps_of(th[:-1]))
        return x

    def neg_obj(th):
        out = fwd(**args(th))
        original = ap.forward_sim_age_structured
        ap.forward_sim_age_structured = lambda *aa, **kw: out
        try:
            tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=lat),
                                              rng_seed=104)).get_trace(data=d, prior_scale=1.)
        finally:
            ap.forward_sim_age_structured = original
        return -jnp.sum(tr['obs']['fn'].log_prob(y)[rows]) + 0.5 * jnp.sum(th[:-1] ** 2)

    vg = jax.jit(jax.value_and_grad(neg_obj))
    th0 = np.zeros(nA * N + 1)
    t0 = time.time(); v0, _ = vg(jnp.asarray(th0))
    log = (a.out / f'innov_s{a.sigma}.jsonl').open('w')
    rec = dict(sigma=a.sigma, window=[a.w0, a.w1], n_innov=int(nA * N), start=float(v0), fitted_reference_nll=ref_nll,
               first_eval_sec=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n')
    n = [0]
    def f(th):
        v, g = vg(jnp.asarray(th)); v, g = float(v), np.asarray(g, float); n[0] += 1
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            v, g = 1e12, np.zeros_like(th)
        log.write(json.dumps(dict(eval=n[0], obj=v, u_rms=float(np.sqrt(np.mean(th[:-1] ** 2))), sec=round(time.time() - t0, 1))) + '\n')
        log.flush()
        return v / OPT_SCALE, g / OPT_SCALE
    res = minimize(f, th0, jac=True, method='L-BFGS-B', options=dict(maxiter=a.maxiter))
    th = res.x; u = th[:-1].reshape(nA, N); eps = np.asarray(eps_of(jnp.asarray(th[:-1])))
    best = sb.run(dict(name=f'innov_s{a.sigma}', **({} if a.alpha == 1 else dict(habitat_alpha=a.alpha)),
                       fmax_scale=float(np.exp(th[-1])), forward_args=dict(process_noise=jnp.asarray(eps, dt))),
                  keep_density=True)
    dens = best.pop('_density'); ab = best['abundance']
    K = to_land(x0['K_flat'], sb.rows, sb.cols, T, int(np.asarray(sb.data['land_mask']).shape[1]))
    rel = dens / np.maximum(K, 1e-12); occ = dens >= .01 * sb.base_med
    where = {}
    for reg in ('east', 'plains', 'west'):
        for e0, e1 in ((1951, 1965), (1966, 1980), (1981, 1995), (1996, 2020)):
            tm = (sb.years >= e0) & (sb.years <= e1) & act
            for st, lo, hi in (('front', 0, .1), ('filling', .1, .7), ('nearK', .7, 1e9)):
                m = tm[:, None] & sb.region[reg][None, :] & occ & (rel >= lo) & (rel < hi)
                if m.sum() >= 5:
                    where[f'{reg}_{e0}-{e1}_{st}'] = dict(n=int(m.sum()), mean_eps=float(eps[m].mean()),
                                                         mean_abs_eps=float(np.abs(eps[m]).mean()))
    acc = {reg: float(np.median(eps[act][:, sb.region[reg]].sum(0))) for reg in ('east', 'plains', 'west')}
    readout_nll = -sum(v['loglik'] for v in ab.values())
    out = dict(sigma=a.sigma, window=[a.w0, a.w1], alpha=a.alpha, fitted_reference_nll=ref_nll, start_obj=float(v0),
               end_obj=float(res.fun) * OPT_SCALE, readout_nll=readout_nll, prior_term=float(0.5 * np.sum(th[:-1] ** 2)),
               gap_closed=float((float(v0) - readout_nll) / (float(v0) - ref_nll)) if float(v0) > ref_nll else None,
               converged=bool(res.success), n_eval=n[0], minutes=round((time.time() - t0) / 60, 1),
               fmax_scale=float(np.exp(th[-1])), u_rms=float(np.sqrt(np.mean(u ** 2))), max_abs_eps=float(np.abs(eps).max()),
               accumulated_log_adjustment_median=acc, where=where,
               offsets=best['curves']['rel_10pct']['arrival_offset']['by_cohort'],
               never_reached=best['curves']['rel_10pct']['arrival_offset']['never'],
               loglik_by={k: v['loglik'] for k, v in ab.items()})
    np.savez_compressed(a.out / f'innov_s{a.sigma}_eps.npz', eps=eps.astype(np.float32), years=sb.years)
    (a.out / f'innov_s{a.sigma}_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('sigma', 'start_obj', 'readout_nll', 'prior_term', 'gap_closed', 'converged',
                                          'n_eval', 'u_rms', 'max_abs_eps', 'accumulated_log_adjustment_median',
                                          'offsets', 'never_reached')}), flush=True)


if __name__ == '__main__':
    main()

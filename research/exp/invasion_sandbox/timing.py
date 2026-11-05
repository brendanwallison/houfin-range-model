"""R26: a per-cell first-detection TIMING term with a log-density link, added to the frozen-habitat objective.

Why: the NB zero costs ~mu, so early fronts made of fractional birds are nearly free (R22). A term linear in mu inherits
that; here presence uses a LOG link, so 1e-6 and 1e-3 birds differ. One event per cell (BBS real routes, 25-49N):
  p_ct = sigmoid(a + b log mu_ct), mu_ct = model expected count summed over the cell's surveyed rows in year t,
  L_c = prod_{surveyed t < t1} (1 - p_ct) * p_{c,t1} (first detection in t1), or prod over all surveyed t (never detected).
Cells first detected within their first 3 surveyed years are dropped (left-censored). a, b (b = softplus) fitted.
Objective = BBS -log-lik (25-49N rows) + w * timing -log-lik + priors. Frozen readouts; --k_real as R21/A9.
Free: Fmax scale, Allee gamma scale (n50), and with --corr one establishment correction per cell (frontinnov_forward,
absolute Allee gate, clock tau). GPU float64.
Usage (inside sbx_env.sh): python timing.py --out DIR [--w 1 --corr --k_real --maxiter 1500 --tag T]
"""
import argparse, json, os, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from scipy.optimize import minimize
from harness import Sandbox, capture, edit_habitat, c_of, ap
import frontinnov_forward as fif

OPT_SCALE = 1e4


def timing_groups(sb):
    """Observation rows (full index space) and group ids for the cell-years that enter the timing term."""
    data = sb.data; years = sb.years
    lookup = sb.lookup
    ocell = lookup[np.asarray(data['obs_rows']), np.asarray(data['obs_cols'])]
    oyr = years[np.asarray(data['obs_time_indices'])]; y = np.asarray(data['observed_results'])
    real = np.asarray(data['obs_obsroute_idx']) >= 0 if 'obs_obsroute_idx' in data else np.ones(len(y), bool)
    band = sb.region['east'] | sb.region['plains'] | sb.region['west']
    keep = real & (ocell >= 0)
    keep[keep] = band[ocell[keep]]
    surveyed = {}
    for i in np.flatnonzero(keep):
        surveyed.setdefault(ocell[i], set()).add(int(oyr[i]))
    rows, gid, det, gcell, gyear = [], [], [], [], []
    key = {}
    n_det = n_never = n_left = 0
    for c, ys in surveyed.items():
        ys = sorted(ys); t1 = sb.first_obs[c]
        if np.isfinite(t1) and t1 < ys[0] + 3:
            n_left += 1; continue
        use = [t for t in ys if not np.isfinite(t1) or t <= t1]
        n_det += int(np.isfinite(t1)); n_never += int(not np.isfinite(t1))
        for t in use:
            key[(c, t)] = len(key); det.append(bool(np.isfinite(t1) and t == t1)); gcell.append(c); gyear.append(t)
    for i in np.flatnonzero(keep):
        g = key.get((ocell[i], int(oyr[i])))
        if g is not None:
            rows.append(i); gid.append(g)
    info = dict(cells_detected=n_det, cells_never=n_never, cells_left_censored=n_left, cell_years=len(key),
                rows=len(rows))
    return np.asarray(rows), np.asarray(gid), np.asarray(det), np.asarray(gcell), np.asarray(gyear), info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--w', type=float, default=1.)
    p.add_argument('--corr', action='store_true'); p.add_argument('--k_real', action='store_true')
    p.add_argument('--sigma', type=float, default=1.); p.add_argument('--tau', type=float, default=10.)
    p.add_argument('--maxiter', type=int, default=1500); p.add_argument('--tag', default='')
    p.add_argument('--alpha', type=float, default=0., help='share of the fitted habitat change kept in the readouts (R27)')
    p.add_argument('--fix_n50', action='store_true', help='hold the Allee n50 at the base value (R27 control)')
    p.add_argument('--agg_k', default='base', help='Allee aggregation k: base, none (Poisson) or a number (R29)')
    p.add_argument('--full', action='store_true',
                   help='R31: objective = the FULL model log density (all observation rows incl. other latitudes and '
                        'pseudo-zeros, the NestWatch fecundity likelihood, priors) instead of BBS 25-49N rows')
    p.add_argument('--fit_regions', default='east,plains,west',
                   help='R32: BBS rows of these regions only enter the objective (region-transfer checks)')
    p.add_argument('--init_n50', type=float, default=1., help='R36: starting n50 multiplier (multi-start check)')
    p.add_argument('--zmap', action='store_true', help='R38: corrections = static readout of time-mean Z (needs --corr)')
    p.add_argument('--zscale', type=float, default=.1)
    p.add_argument('--gate', choices=['allee', 'none', 'ratio'], default='allee',
                   help='R34: correction gate (allee = absolute Allee deficit; none = every density, a dimension-matched control)')
    p.add_argument('--keys', default='Z_gathered,Z_disp_gathered',
                   help='R33: habitat inputs that --alpha scales (default both; Z_gathered alone keeps Z_disp real)')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    ref = sb.run(dict(name='fitted'))
    ref_nll = -sum(v['loglik'] for v in ref['abundance'].values())
    if a.agg_k != 'base':
        from src.model import age_forward as _af
        _af.ALLEE_AGG_K = None if a.agg_k == 'none' else float(a.agg_k)
    keys = tuple(a.keys.split(','))
    d = edit_habitat(sb.data, sb.years, a.alpha, [1966, 2025], keys=keys) if a.alpha != 1 else sb.data
    x0, fwd = capture(d, sb.lat)
    if a.k_real:
        x_real, _ = capture(sb.data, sb.lat); x0 = dict(x0); x0['K_flat'] = jnp.asarray(x_real['K_flat'])
    fwd_fi = fif.build()
    gshape = tuple(np.asarray(x0['land_mask']).shape); N = len(sb.rows)
    band = np.zeros(len(sb.rows), bool)
    for r in a.fit_regions.split(','):
        band |= sb.region[r]
    rows = jnp.asarray(np.flatnonzero(sb.obs_ok)[band[sb.o_cell]])
    y = jnp.asarray(sb.data['observed_results'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    t_rows, t_gid, t_det, g_cell, g_year, tinfo = timing_groups(sb)
    G = len(t_det); t_rows_j, t_gid_j, t_det_j = jnp.asarray(t_rows), jnp.asarray(t_gid), jnp.asarray(t_det)
    if a.zmap:   # R38: corrections are a static READOUT of the time-mean Z (1966-2025), u_cell = Zstd . (zscale b)
        Zm = np.asarray(sb.data['Z_gathered'], float)[(sb.years >= 1966) & (sb.years <= 2025)].mean(0)
        Zstd = jnp.asarray((Zm - Zm.mean(0)) / (Zm.std(0) + 1e-9)); nu = Zstd.shape[1]
        cell_u = lambda b: Zstd @ (a.zscale * b)
    else:
        nu = N if a.corr else 0
        cell_u = lambda b: b
    thr = .01 * sb.base_med

    def unpack(th):
        return th[:nu], th[nu], th[nu + 1], th[nu + 2], jax.nn.softplus(th[nu + 3])

    def args(th):
        _, lf, lg, _, _ = unpack(th)
        x = dict(x0); F = jnp.asarray(x0['Fmax_flat']) * jnp.exp(lf)
        x.update(Fmax_flat=F, c_flat=c_of(F, jnp.asarray(x['Sa_flat']), jnp.asarray(x['Sj_flat']), x),
                 allee_gamma=jnp.asarray(x0['allee_gamma']) * jnp.exp(lg))
        return x

    def parts(th):
        u, _, _, ta, tb = unpack(th)
        if a.corr:
            fif.FI.update(active=True, grid=jnp.zeros(gshape).at[sb.rows, sb.cols].set(cell_u(u)), sigma=a.sigma, tau=a.tau,
                          rho=.3, log_thr=float(np.log(thr)), scale=.5, gate=a.gate)
            out = fwd_fi(**args(th)); fif.FI['active'] = False
        else:
            out = fwd(**args(th))
        original = ap.forward_sim_age_structured
        ap.forward_sim_age_structured = lambda *aa, **kw: out
        try:
            tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=lat),
                                              rng_seed=104)).get_trace(data=d, prior_scale=1.)
        finally:
            ap.forward_sim_age_structured = original
        bbs = -jnp.sum(tr['obs']['fn'].log_prob(y)[rows])
        mu = jnp.asarray(tr['expected_obs']['value'])[t_rows_j]
        mu_g = jax.ops.segment_sum(mu, t_gid_j, num_segments=G)
        z = ta + tb * jnp.log(mu_g + 1e-30)
        tim = -jnp.sum(jnp.where(t_det_j, jax.nn.log_sigmoid(z), jax.nn.log_sigmoid(-z)))
        if a.full:   # every sample site (latents are conditioned, so their prior terms are constants here)
            full = -sum(jnp.sum(s['fn'].log_prob(s['value']) * (s['scale'] if s.get('scale') is not None else 1.))
                        for s in tr.values() if s['type'] == 'sample')
            return full, tim, 0.5 * jnp.sum(u ** 2)
        return bbs, tim, 0.5 * jnp.sum(u ** 2)

    def site_terms(th):   # readout only: observed-data sites (excluding conditioned latents)
        u = unpack(th)[0]
        if a.corr:
            fif.FI.update(active=True, grid=jnp.zeros(gshape).at[sb.rows, sb.cols].set(cell_u(u)), sigma=a.sigma, tau=a.tau,
                          rho=.3, log_thr=float(np.log(thr)), scale=.5, gate=a.gate)
            out = fwd_fi(**args(th)); fif.FI['active'] = False
        else:
            out = fwd(**args(th))
        original = ap.forward_sim_age_structured
        ap.forward_sim_age_structured = lambda *aa, **kw: out
        try:
            tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=lat),
                                              rng_seed=104)).get_trace(data=d, prior_scale=1.)
        finally:
            ap.forward_sim_age_structured = original
        return {nm: float(jnp.sum(s['fn'].log_prob(s['value']) * (s['scale'] if s.get('scale') is not None else 1.)))
                for nm, s in tr.items() if s['type'] == 'sample' and nm not in lat}

    def ref_terms():   # the fit itself (real Z, unedited forward): observed-data site terms
        tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=lat),
                                          rng_seed=104)).get_trace(data=sb.data, prior_scale=1.)
        return {nm: float(jnp.sum(s['fn'].log_prob(s['value']) * (s['scale'] if s.get('scale') is not None else 1.)))
                for nm, s in tr.items() if s['type'] == 'sample' and nm not in lat}

    def neg_obj(th):
        bbs, tim, pr = parts(th)
        return bbs + a.w * tim + pr

    nth = nu + 4
    th0 = np.zeros(nth); th0[nu + 1] = -np.log(a.init_n50)   # gamma scale = 1 / n50 multiplier
    vg = jax.jit(jax.value_and_grad(neg_obj)); pj = jax.jit(parts)
    t0 = time.time(); v0, _ = vg(jnp.asarray(th0)); b0, tm0, _ = pj(jnp.asarray(th0))
    log = (a.out / f'timing{a.tag}.jsonl').open('w')
    rec = dict(full=a.full, fit_regions=a.fit_regions, keys=a.keys, site_terms_start=site_terms(jnp.asarray(th0)), site_terms_fit=ref_terms(), start=float(v0), start_bbs=float(b0), start_timing=float(tm0), fitted_reference_nll=ref_nll, w=a.w,
               corr=a.corr, k_real=a.k_real, timing_info=tinfo, first_eval_sec=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n')
    n = [0]
    def f(th):
        v, g = vg(jnp.asarray(th)); v, g = float(v), np.asarray(g, float); n[0] += 1
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            v, g = 1e12, np.zeros_like(th)
        log.write(json.dumps(dict(eval=n[0], obj=v, sec=round(time.time() - t0, 1))) + '\n'); log.flush()
        return v / OPT_SCALE, g / OPT_SCALE
    bounds = [(-6., 6.)] * nu + [(-4., 3.), ((0., 0.) if a.fix_n50 else (-5., 5.)), (-60., 60.), (-10., 10.)]
    res = minimize(f, th0, jac=True, method='L-BFGS-B', bounds=bounds,
                   options=dict(maxiter=a.maxiter, ftol=1e-15, gtol=1e-12))
    th = res.x; u, lf, lg, ta, tb = unpack(jnp.asarray(th)); u = np.asarray(u)
    bbs1, tim1, pr1 = (float(v) for v in pj(jnp.asarray(th)))
    if a.corr:
        fif.FI.update(active=True, grid=jnp.zeros(gshape).at[sb.rows, sb.cols].set(cell_u(jnp.asarray(u))), sigma=a.sigma,
                      tau=a.tau, rho=.3, log_thr=float(np.log(thr)), scale=.5, gate=a.gate)
    orig = ap.forward_sim_age_structured
    if a.corr:
        ap.forward_sim_age_structured = fwd_fi
    try:
        fa = {'allee_gamma': jnp.asarray(x0['allee_gamma']) * float(np.exp(float(lg)))}
        if a.k_real:
            fa['K_flat'] = x0['K_flat']
        best = sb.run(dict(name=f'timing{a.tag}', habitat_alpha=a.alpha, habitat_keys=keys, fmax_scale=float(np.exp(float(lf))), forward_args=fa))
    finally:
        ap.forward_sim_age_structured = orig; fif.FI['active'] = False
    ab = best['abundance']; readout = -sum(v['loglik'] for v in ab.values())
    uc = np.asarray(cell_u(jnp.asarray(u))) if a.corr else u
    by = ({r: dict(mean=float(uc[sb.region[r]].mean()), sd=float(uc[sb.region[r]].std())) for r in ('east', 'plains', 'west')}
          if a.corr else None)
    out = dict(full=a.full, fit_regions=a.fit_regions, keys=a.keys, site_terms_end=site_terms(jnp.asarray(th)), site_terms_fit=ref_terms(), alpha=a.alpha, agg_k=a.agg_k, fix_n50=a.fix_n50, w=a.w, corr=a.corr, k_real=a.k_real, timing_info=tinfo, start=float(v0), start_bbs=float(b0),
               start_timing=float(tm0), end_obj=float(res.fun) * OPT_SCALE, bbs_nll=bbs1, timing_nll=tim1, prior_term=pr1,
               readout_nll=readout, fitted_reference_nll=ref_nll, fmax_scale=float(np.exp(float(lf))),
               allee_gamma_scale=float(np.exp(float(lg))), n50_scale=float(np.exp(-float(lg))), a=float(ta), b=float(tb),
               converged=bool(res.success), n_eval=n[0], minutes=round((time.time() - t0) / 60, 1), u_by_region=by,
               offsets=best['curves']['rel_10pct']['arrival_offset']['by_cohort'],
               never_reached=best['curves']['rel_10pct']['arrival_offset']['never'],
               loglik_by={k: v['loglik'] for k, v in ab.items()}, abundance=ab)
    if a.corr:
        np.savez_compressed(a.out / f'timing{a.tag}_u.npz', u=uc, coef=u, lon=sb.lon, lat=sb.latd)
    (a.out / f'timing{a.tag}_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('bbs_nll', 'timing_nll', 'readout_nll', 'fmax_scale', 'n50_scale', 'a', 'b',
                                          'converged', 'n_eval', 'offsets', 'never_reached')}), flush=True)


if __name__ == '__main__':
    main()

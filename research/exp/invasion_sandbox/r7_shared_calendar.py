"""R7: one shared calendar rise vs East-only, with the native West present.

Step 3, habitat frozen at the 1966-2025 mean, native West + 1940 release, loss = minus the BBS log-likelihood over ALL
real rows under the fit's own observation model; L-BFGS with exact JAX gradients. Fmax x fscale x m(calendar year):
  east:     m on East cells only (Plains and West keep fitted rates on the fixed habitat)
  shared:   one m on every cell
  separate: independent East / not-East multipliers
R8 modes (measured HYDE population density, DESK covariate states, 27 km -> 54 km 2x2 block mean; z = standardised
log1p density; zbar = the cell's 1966-2025 mean): Fmax x exp(bs zbar [+ bt (z_t - zbar)]) on every cell
  pop_static: bs only; pop_split: bs + bt; pop_split_cal: bs + bt + the shared calendar knots.
Usage (inside sbx_env.sh): python r7_shared_calendar.py --mode east|shared|separate|pop_static|pop_split|pop_split_cal
       --out DIR [--maxiter 200]
"""
import argparse, json, os, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from scipy.optimize import minimize
from harness import Sandbox, capture, edit_habitat, region_masks, c_of, to_land, ap

STATES = os.path.expanduser('~/houfin/work/houfin/processed/encoder/states/yearly_states')


def bui_z(sb, channel=5):
    """(T, N_land) standardised log1p BUI channel (default 5 = bui_q90) on 54 km land cells: mean over the 27 km
    children WITH coverage (bui_avail > .5). Cells without coverage get the covered mean in every year (no effect)."""
    T = len(sb.years); out = np.full((T, len(sb.rows)), np.nan)
    for ti, y in enumerate(sb.years):
        b = np.load(os.path.join(STATES, f'state_{int(y)}.npz'))['bui']
        v = np.where(b[..., 0] > .5, np.log1p(np.clip(b[..., channel], 0, None)), np.nan)
        ny, nx = v.shape; vp = np.full((2 * ((ny + 1) // 2), 2 * ((nx + 1) // 2)), np.nan); vp[:ny, :nx] = v
        with np.errstate(all='ignore'):
            c = np.nanmean(vp.reshape(vp.shape[0] // 2, 2, vp.shape[1] // 2, 2), axis=(1, 3))
        out[ti] = c[sb.rows, sb.cols]
    cov = np.isfinite(out).all(0)
    m, sd = out[:, cov].mean(), out[:, cov].std()
    z = np.where(cov[None], (out - m) / sd, 0.)
    w = (sb.years >= 1966) & (sb.years <= 2025)
    zbar = z[w].mean(0)
    return zbar, z - zbar[None], float(1 - cov.mean())


def population_z(sb):
    """(T, N_land) standardised log1p HYDE density on the 54 km land cells; 54 km cell (r, c) = 27 km children
    (2r..2r+1, 2c..2c+1), mean over non-missing children (coarsen_model_grid layout: origin-aligned, padded S / E)."""
    T = len(sb.years); out = np.full((T, len(sb.rows)), np.nan)
    for ti, y in enumerate(sb.years):
        h = np.load(os.path.join(STATES, f'state_{int(y)}.npz'))['hyde'][..., 0]
        ny, nx = h.shape; hp = np.full((2 * ((ny + 1) // 2), 2 * ((nx + 1) // 2)), np.nan); hp[:ny, :nx] = h
        b = hp.reshape(hp.shape[0] // 2, 2, hp.shape[1] // 2, 2)
        with np.errstate(all='ignore'):
            c = np.nanmean(np.log1p(b), axis=(1, 3))
        out[ti] = c[sb.rows, sb.cols]
    miss = ~np.isfinite(out)
    out = np.where(miss, np.nanmean(out), out)
    z = (out - out.mean()) / out.std()
    w = (sb.years >= 1966) & (sb.years <= 2025)
    zbar = z[w].mean(0)
    return zbar, z - zbar[None], float(miss.mean())

KNOTS = [1940, 1955, 1965, 1975, 1985, 1995, 2010, 2025]
OPT_SCALE = 1e4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['east', 'shared', 'separate', 'pop_static', 'pop_split', 'pop_split_cal',
                                      'bui_split', 'pop_bui_split', 'bui_split_cal', 'bui_regional', 'fs_only',
                                      'bui_spline'],
                   required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--maxiter', type=int, default=200)
    p.add_argument('--fscale0', type=float, default=1.)
    p.add_argument('--bui_channel', type=int, default=5, help='bui stream channel: 3 = q50, 5 = q90 (default)')
    p.add_argument('--tag', default='', help='suffix for output names (sensitivity arms)')
    p.add_argument('--alpha', type=float, default=0., help='habitat change kept: 0 = frozen (default), 1 = fitted')
    p.add_argument('--bui_field', default='Fmax', choices=['Fmax', 'K', 'Sa', 'allee'],
                   help='rate the BUI term acts on: Fmax / K multiplicatively, Sa on the logit scale (R11, R12)')
    p.add_argument('--bui_freeze_before', type=int, default=0,
                   help='hold BUI change before this year at its value then (R11c survivorship sensitivity)')
    p.add_argument('--target', default='Fmax', choices=['Fmax', 'lowdens'],
                   help='lowdens (R14): the Fmax multiplier becomes a low-density boost G fading with N/K')
    p.add_argument('--loss_region', default='all', choices=['all', 'east', 'plains', 'west'],
                   help='score only the rows of this region (R10: identify a response where there is no front)')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    ref = sb.run(dict(name='fitted'))
    ref_nll = -sum(v['loglik'] for v in ref['abundance'].values())
    d = edit_habitat(sb.data, sb.years, a.alpha, [1966, 2025]) if a.alpha != 1 else sb.data
    x0, fwd = capture(d, sb.lat)
    shape = np.asarray(x0['Fmax_flat']).shape
    nld = 1 if a.target == 'lowdens' else 0   # extra parameter log rho, placed just before log fscale
    if nld:
        import lowdens_forward as ldf
        fwd_ld = ldf.build()
        ld_grid_shape = (len(sb.years),) + tuple(np.asarray(x0['land_mask']).shape)
        def ld_grid(m_):
            return jnp.ones(ld_grid_shape).at[:, sb.rows, sb.cols].set(m_)
    if a.bui_field == 'allee':   # R13: settlement eases mate finding (allee_forward fork)
        import allee_forward as afw
        fwd_allee = afw.build()
        grid_shape = (len(sb.years),) + tuple(np.asarray(x0['land_mask']).shape)
        def allee_grid(mK_):
            return jnp.ones(grid_shape).at[:, sb.rows, sb.cols].set(mK_)
        print(json.dumps(dict(allee=dict(gamma=np.asarray(x0['allee_gamma']).tolist(),
                                         relative=bool(x0.get('allee_relative'))))), flush=True)
    nk = len(KNOTS)
    W = jnp.asarray(np.stack([np.interp(sb.years.astype(float), KNOTS, np.eye(nk)[i]) for i in range(nk)], 1))
    east = jnp.asarray(region_masks(sb.lon)['east'])
    groups = dict(east=[east], shared=[jnp.ones_like(east)], separate=[east, ~east],
                  pop_static=[], pop_split=[], pop_split_cal=[jnp.ones_like(east)], bui_split=[], pop_bui_split=[],
                  bui_split_cal=[jnp.ones_like(east)],
                  bui_regional=[], fs_only=[], bui_spline=[])[a.mode]
    npop = dict(pop_static=1, pop_split=2, pop_split_cal=2, pop_bui_split=2).get(a.mode, 0)
    nbui = dict(bui_split=2, pop_bui_split=2, bui_split_cal=2, bui_regional=3, bui_spline=6).get(a.mode, 0)
    pop_info = {}
    if npop:
        zbar, dz, miss = population_z(sb)
        pop_info = dict(missing_share=miss, dz_east_1940=float(dz[sb.years == 1940][0][sb.region['east']].mean()),
                        dz_east_2020=float(dz[sb.years == 2020][0][sb.region['east']].mean()),
                        dz_west_1940=float(dz[sb.years == 1940][0][sb.region['west']].mean()),
                        dz_west_2020=float(dz[sb.years == 2020][0][sb.region['west']].mean()))
        print(json.dumps(dict(population=pop_info)), flush=True)
        zbar_j, dz_j = jnp.asarray(zbar), jnp.asarray(dz)
    if nbui:
        bzbar, bdz, bmiss = bui_z(sb, a.bui_channel)
        if a.bui_freeze_before:
            i_f = int(np.flatnonzero(sb.years == a.bui_freeze_before)[0]); bdz = bdz.copy(); bdz[:i_f] = bdz[i_f]
        e = sb.region['east']; w_ = sb.region['west']; yi = lambda yy: int(np.flatnonzero(sb.years == yy)[0])
        pop_info.update(bui_uncovered_share=bmiss, bui_dz_east_1940=float(bdz[yi(1940)][e].mean()),
                        bui_dz_east_2020=float(bdz[yi(2020)][e].mean()), bui_dz_west_1940=float(bdz[yi(1940)][w_].mean()),
                        bui_dz_west_2020=float(bdz[yi(2020)][w_].mean()))
        print(json.dumps(dict(bui=pop_info)), flush=True)
        bzbar_j, bdz_j = jnp.asarray(bzbar), jnp.asarray(bdz)
        bztot = bzbar[None, :] + bdz; bztot_j = jnp.asarray(bztot)
        cov_ = np.abs(bdz).sum(0) > 0
        bknots = np.quantile(bztot[:, cov_], [.02, .2, .4, .6, .8, .98]); bknots_j = jnp.asarray(bknots)
        pop_info.update(bui_spline_knots=bknots.tolist())
    ng = len(groups) * nk
    to_land_np = lambda arr: to_land(arr, sb.rows, sb.cols, len(sb.years), int(np.asarray(sb.data['land_mask']).shape[1]))
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    band = ((sb.region['east'] | sb.region['plains'] | sb.region['west']) if a.loss_region == 'all'
            else sb.region[a.loss_region])                              # 25-49N, as the readout's regions
    rows = jnp.asarray(np.flatnonzero(sb.obs_ok)[band[sb.o_cell]])
    y = jnp.asarray(sb.data['observed_results'])

    def mult_of(theta, xp=jnp):
        """-> (Fmax multiplier, K multiplier), each (T, N_land)."""
        m = xp.ones((len(sb.years), len(sb.lon))); mK = xp.ones((len(sb.years), len(sb.lon)))
        for gi, gm in enumerate(groups):
            mt = (np.asarray(W) if xp is np else W) @ xp.exp(theta[gi * nk:(gi + 1) * nk])
            m = xp.where((np.asarray(gm) if xp is np else gm)[None, :], mt[:, None], m)
        if npop:
            zb, dd = (zbar, dz) if xp is np else (zbar_j, dz_j)
            lin = theta[ng] * zb[None, :] + (theta[ng + 1] * dd if npop == 2 else 0.)
            m = m * xp.exp(lin)
        if nbui:
            zb, dd = (bzbar, bdz) if xp is np else (bzbar_j, bdz_j)
            if nbui == 6:   # bui_spline: G = exp(piecewise-linear in total BUI z); uncovered cells sit at z = 0
                zt, kn = (bztot, bknots) if xp is np else (bztot_j, bknots_j)
                mb = xp.exp(xp.interp(zt, kn, theta[ng + npop:ng + npop + 6]))
            elif nbui == 3:   # bui_regional: shared bs, bt for East and for not-East
                em = np.asarray(sb.region['east']) if xp is np else east
                bt = xp.where(em[None, :], theta[ng + npop + 1], theta[ng + npop + 2])
                mb = xp.exp(theta[ng + npop] * zb[None, :] + bt * dd)
            else:
                mb = xp.exp(theta[ng + npop] * zb[None, :] + theta[ng + npop + 1] * dd)
            if nbui == 6:
                m = m * mb
            elif a.bui_field in ('K', 'Sa', 'allee'):   # Sa: exp(logit shift) (sa_of); allee: gamma multiplier
                mK = mK * mb
            else:
                m = m * mb
        return m, mK

    def neg_ll(theta):
        x = dict(x0)
        mF, mK = mult_of(theta)
        if nld:   # the multiplier is the low-density boost G; fecundity at density is fscale only
            F = jnp.asarray(x0['Fmax_flat']) * jnp.exp(theta[-1])
        else:
            F = jnp.asarray(x0['Fmax_flat']) * (mF * jnp.exp(theta[-1])).reshape(shape)
        Sa = jnp.asarray(x['Sa_flat'])
        if a.bui_field == 'K':
            x['K_flat'] = jnp.asarray(x0['K_flat']) * mK.reshape(np.asarray(x0['K_flat']).shape)
        elif a.bui_field == 'Sa':
            Sa = sa_of(Sa, mK.reshape(Sa.shape), jnp)
        Sa = jnp.minimum(Sa, .995); Sj = jnp.minimum(jnp.asarray(x['Sj_flat']), Sa)
        x.update(Fmax_flat=F, Sa_flat=Sa, Sj_flat=Sj, c_flat=c_of(F, Sa, Sj, x))   # as harness rate_array
        if a.bui_field == 'allee':
            afw.ALLEE.update(active=True, grid=allee_grid(mK))
            out = fwd_allee(**x)
            afw.ALLEE['active'] = False
        elif nld:
            ldf.LD.update(active=True, grid=ld_grid(mF), rho=jnp.exp(theta[-2]))
            out = fwd_ld(**x)
            ldf.LD['active'] = False
        else:
            out = fwd(**x)
        original = ap.forward_sim_age_structured
        ap.forward_sim_age_structured = lambda *aa, **kw: out
        try:
            tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=lat),
                                              rng_seed=104)).get_trace(data=d, prior_scale=1.)
        finally:
            ap.forward_sim_age_structured = original
        return -jnp.sum(tr['obs']['fn'].log_prob(y)[rows])

    def sa_of(Sa0, mexp, xp):
        """Adult survival shifted on the logit scale by log(mexp)."""
        lg = xp.log(xp.clip(Sa0, 1e-6, 1 - 1e-6)) - xp.log1p(-xp.clip(Sa0, 1e-6, 1 - 1e-6)) + xp.log(mexp)
        return 1. / (1. + xp.exp(-lg))

    vg = jax.jit(jax.value_and_grad(neg_ll))
    if nld:   # identity: G = 1 reproduces the original forward
        ldf.LD.update(active=True, grid=jnp.ones(ld_grid_shape), rho=.1); o1 = fwd_ld(**x0); ldf.LD['active'] = False
        o0 = fwd(**x0)
        print(json.dumps(dict(lowdens_identity_max_rel=float(jnp.max(jnp.abs(o1[0] - o0[0])) / jnp.max(jnp.abs(o0[0]))))),
              flush=True)
    if a.bui_field == 'allee':   # identity: grid of ones reproduces the original forward
        afw.ALLEE.update(active=True, grid=jnp.ones(grid_shape)); o1 = fwd_allee(**x0); afw.ALLEE['active'] = False
        o0 = fwd(**x0)
        print(json.dumps(dict(allee_identity_max_rel=float(jnp.max(jnp.abs(o1[0] - o0[0])) / jnp.max(jnp.abs(o0[0]))))),
              flush=True)
    theta0 = np.r_[np.zeros(ng + npop + nbui), [np.log(.1)] * nld, np.log(a.fscale0)]
    log = (a.out / f'r7_{a.mode}{a.tag}.jsonl').open('w')
    t0 = time.time(); v0, _ = vg(jnp.asarray(theta0))
    rec = dict(arm='fitted', nll=ref_nll, nll_frozen_start=float(v0), first_eval_sec=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n')
    n = [0]
    def f(theta):
        v, g = vg(jnp.asarray(theta)); v, g = float(v), np.asarray(g, float); n[0] += 1
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            v, g = 1e12, np.zeros_like(theta)
        log.write(json.dumps(dict(eval=n[0], theta=list(map(float, theta)), nll=v,
                                  grad_norm=float(np.linalg.norm(g)), sec=round(time.time() - t0, 1))) + '\n')
        log.flush()
        return v / OPT_SCALE, g / OPT_SCALE   # O(1) objective for L-BFGS-B's first step (raw |grad| ~ 2e4)
    res = minimize(f, theta0, jac=True, method='L-BFGS-B', bounds=[(-4., 3.)] * ng + [(-20., 20.)] * (npop + nbui) + [(np.log(.005), np.log(2.))] * nld + [(-4., 3.)],
                   options=dict(maxiter=a.maxiter))
    th = res.x
    m, mK = mult_of(th, np)
    Sa0n = np.asarray(x0['Sa_flat'], float)
    nll_at_x = float(vg(jnp.asarray(th))[0])
    if a.bui_field == 'allee':
        afw.ALLEE.update(active=True, grid=allee_grid(jnp.asarray(mK)))
        _orig_fwd = ap.forward_sim_age_structured; ap.forward_sim_age_structured = fwd_allee
    if nld:
        ldf.LD.update(active=True, grid=ld_grid(jnp.asarray(m)), rho=float(np.exp(th[-2])))
        _orig_fwd = ap.forward_sim_age_structured; ap.forward_sim_age_structured = fwd_ld
    best = sb.run(dict(name=f'r7_{a.mode}', fmax_scale=float(np.exp(th[-1])),
                       **({} if a.alpha == 1 else dict(habitat_alpha=a.alpha)),
                       rate_array=(dict() if nld else dict(Fmax=m, K=mK) if a.bui_field in ('Fmax', 'K') else
                                   dict(Fmax=m) if a.bui_field == 'allee' else
                                   dict(Fmax=m, Sa=sa_of(Sa0n, mK.reshape(Sa0n.shape), np) / Sa0n))))
    if a.bui_field == 'allee':
        ap.forward_sim_age_structured = _orig_fwd; afw.ALLEE['active'] = False
    if nld:
        ap.forward_sim_age_structured = _orig_fwd; ldf.LD['active'] = False
    ab = best['abundance']
    out = dict(mode=a.mode, knots=KNOTS, fitted_reference_nll=ref_nll, frozen_start_nll=float(v0),
               best_nll=float(res.fun) * OPT_SCALE, nll_at_x=nll_at_x, readout_nll=-sum(v['loglik'] for v in ab.values()),
               converged=bool(res.success), message=str(res.message), n_eval=n[0],
               fmax_scale=float(np.exp(th[-1])),
               multipliers={name: list(np.exp(th[gi * nk:(gi + 1) * nk]) * np.exp(th[-1])) for gi, name in
                            enumerate(dict(east=['east'], shared=['all'], separate=['east', 'not_east'],
                                           pop_static=[], pop_split=[], pop_split_cal=['all'], bui_split=[],
                                           pop_bui_split=[], bui_split_cal=['all'], bui_regional=[],
                                           fs_only=[], bui_spline=[])[a.mode])},
               population=dict(pop_info, bs=float(th[ng]) if npop else None,
                               bt=float(th[ng + 1]) if npop == 2 else None,
                               bui_bs=float(th[ng + npop]) if nbui else None,
                               bui_bt=float(th[ng + npop + 1]) if nbui else None,
                               bui_bt_not_east=float(th[ng + npop + 2]) if nbui == 3 else None),
               loss_region=a.loss_region, bui_field=a.bui_field, bui_freeze_before=a.bui_freeze_before,
               target=a.target, rho=float(np.exp(th[-2])) if nld else None,
               bui_spline_values=list(map(float, th[ng + npop:ng + npop + 6])) if nbui == 6 else None,
               G_east=dict((str(y), dict(p10=float(np.percentile(m[sb.years == y][0][sb.region['east']], 10)),
                                         median=float(np.median(m[sb.years == y][0][sb.region['east']])),
                                         p90=float(np.percentile(m[sb.years == y][0][sb.region['east']], 90))))
                           for y in (1940, 1960, 1980, 2000, 2020)) if nld else None,
               east_change_1940_2020=dict(
                   Fmax=float(np.median((m * np.exp(th[-1]))[sb.years == 2020][0][sb.region['east']] /
                                        (m * np.exp(th[-1]))[sb.years == 1940][0][sb.region['east']])),
                   K=float(np.median(mK[sb.years == 2020][0][sb.region['east']] / mK[sb.years == 1940][0][sb.region['east']]))
                   if a.bui_field in ('K', 'allee') else 1.0,
                   Sa_delta=float(np.median((lambda S: S[sb.years == 2020][0] - S[sb.years == 1940][0])(
                       to_land_np(sa_of(Sa0n, mK.reshape(Sa0n.shape), np)) - to_land_np(Sa0n))[sb.region['east']]))
                   if a.bui_field == 'Sa' else 0.0),
               minutes=round((time.time() - t0) / 60, 1), sha=best['final_state_sha256'][:12],
               offsets=best['curves']['rel_10pct']['arrival_offset']['by_cohort'],
               never_reached=best['curves']['rel_10pct']['arrival_offset']['never'],
               loglik_by=dict((k, v['loglik']) for k, v in ab.items()),
               ref_loglik_by=dict((k, v['loglik']) for k, v in ref['abundance'].items()),
               obs_over_expected=dict((k, v['obs_over_expected']) for k, v in ab.items()),
               zero_share=dict((k, [v['zero_share_obs'], v['zero_share_pred']]) for k, v in ab.items()))
    (a.out / f'r7_{a.mode}{a.tag}_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict((k, out[k]) for k in ('mode', 'fitted_reference_nll', 'frozen_start_nll', 'best_nll', 'nll_at_x',
                                                 'readout_nll', 'converged', 'n_eval', 'minutes'))), flush=True)


if __name__ == '__main__':
    main()

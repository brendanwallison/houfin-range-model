"""R16: middle-ground readouts built on Z -- can Z's change, routed to the RIGHT part of the density response, replace
the fitted habitat change? (no covariates, no calendar time)

The four existing habitat readouts (Sa, Sj, Fmax, K from Z . beta) are held at FROZEN habitat (1966-2025 mean Z), and
one NEW readout of the original, time-varying Z is fitted (L-BFGS, exact JAX gradients, GPU float64):
  theta    crowding curvature theta(x, t) = exp(theta0 + z(x, t) . b)       (zcrowd_forward fork; K stays the equilibrium)
  lowdens  low-density boost G(x, t) = exp(g0 + z(x, t) . b), fading with N/K at rate rho  (lowdens_forward fork)
z = the model's own Z_gathered, standardised per dimension over land cell-years. --zsource static uses each cell's
1966-2025 mean z instead (the readout sees only space): the matched control for what Z's TIME change adds.
Objective: minus the BBS log-lik over all 25-49N rows + ridge |b|^2 / (2 sd^2) (sd = --ridge_sd); fscale (constant Fmax
scale) is also fitted. Identity: b = 0 and theta0 = log(fitted power) (or g0 = 0) reproduces the frozen replay.
Usage (inside sbx_env.sh): python zreadout.py --mode theta|lowdens --zsource time|static --out DIR [--maxiter 100]
"""
import argparse, json, math, os, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from scipy.optimize import minimize
from harness import Sandbox, capture, edit_habitat, c_of, ap

OPT_SCALE = 1e4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['theta', 'lowdens'], required=True)
    p.add_argument('--zsource', choices=['time', 'static'], default='time')
    p.add_argument('--out', type=Path, required=True); p.add_argument('--maxiter', type=int, default=100)
    p.add_argument('--ridge_sd', type=float, default=.5); p.add_argument('--tag', default='')
    p.add_argument('--b_scale', type=float, default=1., help='b = b_scale * u (conditioning; u is optimised)')
    p.add_argument('--rho_max', type=float, default=2., help='upper bound on the lowdens fade rho (x K)')
    p.add_argument('--choice', choices=['none', 'band', 'dir'], default='none',
                   help='B1 habitat-informed juvenile kernel choice from the fitted path-habitat signal logit(Q) of the '
                        'path AHEAD: band = across all band x direction kernels; dir = within band (step 27 form)')
    p.add_argument('--kappa', type=float, default=0., help='fixed choice strength (not fitted)')
    p.add_argument('--dep_ramp', type=float, default=0.,
                   help='B7: East-only departure boost s(t) = 1 + dep_ramp * logistic((year - dep_mid) / dep_tau), '
                        'p -> 1 - (1 - p)^s (dep_forward); fixed, lowdens mode only')
    p.add_argument('--dep_mid', type=float, default=1955.); p.add_argument('--dep_tau', type=float, default=5.)
    p.add_argument('--dep_s0', type=float, default=None,
                   help='R20: East s(t) = dep_s0 + (dep_s1 - dep_s0) logistic(...) (overrides dep_ramp when set)')
    p.add_argument('--dep_s1', type=float, default=1.)
    p.add_argument('--dep_floor', type=float, default=0., help='B9: departure floor f (all cells): p -> f + (1 - f) p')
    p.add_argument('--k_real', action='store_true',
                   help='capacity K from the REAL-Z capture (abundance keeps Z time change); other readouts frozen')
    p.add_argument('--settle', default=None,
                   help='B3 destination-habitat settlement: BLOCK,WEIGHT[,TAU] passed to the forward (settle_block, '
                        'settle_weight in capacity|r0|capacity_r0|capacity_fitness|uniform, settle_tau); dispersers only')
    p.add_argument('--fit_floor', action='store_true',
                   help='B9b: the floor is a FITTED parameter (logit, start at --dep_floor), lowdens mode only')
    p.add_argument('--depart_kappa', type=float, default=0.,
                   help='B2 origin-habitat philopatry: departure x exp(-k max(0, log lambda0 - median)), <= 1, static, '
                        'from the frozen-habitat rates (good cells keep their birds)')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    ref = sb.run(dict(name='fitted'))
    ref_nll = -sum(v['loglik'] for v in ref['abundance'].values())
    d = edit_habitat(sb.data, sb.years, 0., [1966, 2025])            # existing readouts on frozen habitat
    x0, fwd = capture(d, sb.lat)
    if a.k_real:   # K keeps Z's real time change (mix_replay: K carries most of the likelihood)
        x_real_, _ = capture(sb.data, sb.lat)
        x0 = dict(x0); x0['K_flat'] = jnp.asarray(x_real_['K_flat'])
        print(json.dumps(dict(k_real=True)), flush=True)
    choice_W = None
    if a.choice != 'none':   # B1: per-origin multipliers on dispersers per kernel, mass-preserving at each origin
        from src.model import age_forward as af_
        Q = np.asarray(x0['Q_flat'], float)                                   # (T, N_land, K): frozen-habitat path survival
        sig = np.log(np.clip(Q, 1e-9, 1 - 1e-9)) - np.log1p(-np.clip(Q, 1e-9, 1 - 1e-9))
        mass = np.real(np.asarray(x0['juvenile_fft_kernel_stack'])[:, 0, 0]).astype(float)
        if a.choice == 'dir':
            choice_W = np.asarray(af_.directional_choice_weights(jnp.asarray(sig), jnp.asarray(mass), a.kappa))
        else:
            nb = sig.shape[-1] // af_.N_DIRECTIONS
            s4 = sig.reshape(sig.shape[:-1] + (af_.N_DIRECTIONS, nb))
            A = s4[..., np.asarray(af_.OPPOSITE_DIRECTION), :].reshape(sig.shape)   # habitat AHEAD of each kernel
            pk = mass / mass.sum()
            e = np.exp(a.kappa * (A - (A * pk).sum(-1, keepdims=True)))
            choice_W = e / (e * pk).sum(-1, keepdims=True)                     # sum_k p_k v_k = 1 at every origin
        x0 = dict(x0); x0['directional_weights'] = jnp.asarray(choice_W, jnp.asarray(x0['Sa_flat']).dtype)
        print(json.dumps(dict(choice=a.choice, kappa=a.kappa, W_p05=float(np.percentile(choice_W, 5)),
                              W_p95=float(np.percentile(choice_W, 95)))), flush=True)
    settle_args = {}
    if a.settle:
        parts = a.settle.split(',')
        settle_args = dict(settle_block=int(parts[0]), settle_weight=parts[1], settle_dispersers_only=True)
        if len(parts) > 2:
            settle_args['settle_tau'] = float(parts[2])
        missing = [k for k in settle_args if k not in x0]
        if missing:
            raise ValueError(f'forward has no {missing}')
        x0 = dict(x0); x0.update(settle_args)
        print(json.dumps(dict(settle=settle_args)), flush=True)
    if a.k_real:   # the readout must carry the real-Z K too (harness forward_args override)
        settle_args['K_flat'] = x0['K_flat']
    depart_scale = None
    if a.depart_kappa > 0:
        from harness import potential_lambda, to_land
        T_, Nx_ = len(sb.years), int(np.asarray(sb.data['land_mask']).shape[1])
        Sa_, Sj_, F_ = (to_land(x0[k], sb.rows, sb.cols, T_, Nx_).mean(0) for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat'))
        ll0 = np.log(np.maximum(potential_lambda(Sa_, Sj_, F_, bool(x0.get('yearling_breeding'))), 1e-6))
        depart_scale = np.exp(-a.depart_kappa * np.maximum(0., ll0 - np.median(ll0)))
        x0 = dict(x0); x0['departure_scale'] = jnp.asarray(depart_scale, jnp.asarray(x0['Sa_flat']).dtype)
        print(json.dumps(dict(depart_kappa=a.depart_kappa, scale_p05=float(np.percentile(depart_scale, 5)),
                              scale_median=float(np.median(depart_scale)))), flush=True)
    # the new readout's input: the model's own (unfrozen) Z, standardised
    Z = np.asarray(sb.data['Z_gathered'], float)                      # (T, N_land, M)
    mu, sd = Z.reshape(-1, Z.shape[-1]).mean(0), Z.reshape(-1, Z.shape[-1]).std(0) + 1e-9
    z = (Z - mu) / sd
    if a.zsource == 'static':
        w = (sb.years >= 1966) & (sb.years <= 2025); z = np.broadcast_to(z[w].mean(0, keepdims=True), z.shape).copy()
    M = z.shape[-1]; z_j = jnp.asarray(z)
    grid_shape = (len(sb.years),) + tuple(np.asarray(x0['land_mask']).shape)
    to_grid = lambda v, fill: jnp.full(grid_shape, fill).at[:, sb.rows, sb.cols].set(v)
    from src.model import density_regulation as dr
    base_power = float(dr.CROWDING_POWER)
    if a.mode == 'theta':
        import zcrowd_forward as fk
        fwd_k = fk.build(); fk.install()
        theta0 = np.r_[np.log(base_power), np.zeros(M), 0.]           # [log theta0, b (M), log fscale]
        bounds = [(np.log(.25), np.log(8.))] + [(-3. / a.b_scale, 3. / a.b_scale)] * M + [(-4., 3.)]
    else:
        import lowdens_forward as fk
        if a.dep_ramp > 0 or a.dep_s0 is not None or a.dep_floor > 0 or a.fit_floor:
            import dep_forward as dfk
            fwd_k = dfk.build(fk._EDITS, dict(LD=fk.LD))
        else:
            fwd_k = fk.build()
        theta0 = np.r_[0., np.zeros(M), np.log(min(.3, a.rho_max) * .9), 0.]                # [g0, b (M), log rho, log fscale]
        if a.fit_floor:   # insert logit(floor) before log fscale
            f0 = min(max(a.dep_floor, 1e-3), .9)
            theta0 = np.r_[theta0[:-1], math.log(f0 / (1 - f0)), theta0[-1]]
        bounds = [(-4., 4.)] + [(-3. / a.b_scale, 3. / a.b_scale)] * M + [(np.log(.005), np.log(a.rho_max))] + \
            ([(-7., 2.)] if a.fit_floor else []) + [(-4., 3.)]
    band = sb.region['east'] | sb.region['plains'] | sb.region['west']
    rows = jnp.asarray(np.flatnonzero(sb.obs_ok)[band[sb.o_cell]])
    y = jnp.asarray(sb.data['observed_results'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}

    def readout_grid(th):
        b = a.b_scale * th[1:1 + M]
        lin = th[0] + jnp.tensordot(z_j, b, axes=([2], [0]))           # (T, N_land)
        return to_grid(jnp.exp(lin), base_power if a.mode == 'theta' else 1.0)

    def args(th):
        x = dict(x0); F = jnp.asarray(x0['Fmax_flat']) * jnp.exp(th[-1])
        Sa, Sj = jnp.asarray(x['Sa_flat']), jnp.asarray(x['Sj_flat'])
        x.update(Fmax_flat=F, c_flat=c_of(F, Sa, Sj, x)); return x

    def run_fork(th):
        g = readout_grid(th)
        if a.mode == 'theta':
            fk.THETA.update(active=True, grid=g); out = fwd_k(**args(th)); fk.THETA['active'] = False
            fk.THETA_CUR[0] = None   # the scan left a per-year tracer here; later crowding calls must not see it
        else:
            if a.fit_floor:
                dfk.DEP['floor'] = jax.nn.sigmoid(th[-2])
            fk.LD.update(active=True, grid=g, rho=jnp.exp(th[-3] if a.fit_floor else th[-2])); out = fwd_k(**args(th))
            fk.LD['active'] = False
        return out

    def neg_ll(th):
        out = run_fork(th)
        original = ap.forward_sim_age_structured
        ap.forward_sim_age_structured = lambda *aa, **kw: out
        try:
            tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=lat),
                                              rng_seed=104)).get_trace(data=d, prior_scale=1.)
        finally:
            ap.forward_sim_age_structured = original
        ll = -jnp.sum(tr['obs']['fn'].log_prob(y)[rows])
        return ll + 0.5 * jnp.sum((a.b_scale * th[1:1 + M]) ** 2) / a.ridge_sd ** 2

    vg = jax.jit(jax.value_and_grad(neg_ll))
    # identity: zero readout == the frozen replay through the original forward
    o1 = run_fork(jnp.asarray(theta0)); o0 = fwd(**args(jnp.asarray(theta0)))
    ident = float(jnp.max(jnp.abs(o1[0] - o0[0])) / jnp.max(jnp.abs(o0[0])))
    if a.mode == 'lowdens' and (a.dep_ramp > 0 or a.dep_s0 is not None or a.dep_floor > 0 or a.fit_floor):   # AFTER identity
        ramp = 1. / (1. + np.exp(-(sb.years.astype(float) - a.dep_mid) / a.dep_tau))
        s_e = (1. + a.dep_ramp * ramp) if a.dep_s0 is None else (a.dep_s0 + (a.dep_s1 - a.dep_s0) * ramp)
        east_f = sb.region['east'][None, :].astype(float)
        s_land = 1. + (s_e[:, None] - 1.) * east_f
        dfk.DEP.update(active=True, grid=to_grid(jnp.asarray(s_land), 1.0), floor=float(a.dep_floor))
        print(json.dumps(dict(dep_ramp=a.dep_ramp, dep_mid=a.dep_mid, s_east_1940=float(s_land[sb.years == 1940][0][sb.region['east']].mean()),
                              s_east_1970=float(s_land[sb.years == 1970][0][sb.region['east']].mean()))), flush=True)
        vg = jax.jit(jax.value_and_grad(neg_ll))   # retrace with the ramp active
    tag = f'{a.mode}_{a.zsource}{a.tag}'
    log = (a.out / f'r16_{tag}.jsonl').open('w')
    t0 = time.time(); v0, _ = vg(jnp.asarray(theta0))
    rec = dict(fitted_reference_nll=ref_nll, frozen_start_nll=float(v0), identity_max_rel=ident, M=M, base_power=base_power)
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n')
    n = [0]
    def f(th):
        v, g = vg(jnp.asarray(th)); v, g = float(v), np.asarray(g, float); n[0] += 1
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            v, g = 1e12, np.zeros_like(th)
        log.write(json.dumps(dict(eval=n[0], obj=v, b_norm=float(np.linalg.norm(th[1:1 + M])), sec=round(time.time() - t0, 1))) + '\n'); log.flush()
        return v / OPT_SCALE, g / OPT_SCALE
    res = minimize(f, theta0, jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=a.maxiter))
    th = res.x
    # full readout through the harness with the fork installed and the readout grid fixed
    g = readout_grid(jnp.asarray(th))
    if a.mode == 'theta':
        fk.THETA.update(active=True, grid=g)
    else:
        if a.fit_floor:
            dfk.DEP['floor'] = float(1. / (1. + np.exp(-th[-2])))
        fk.LD.update(active=True, grid=g, rho=float(np.exp(th[-3] if a.fit_floor else th[-2])))
    orig = ap.forward_sim_age_structured; ap.forward_sim_age_structured = fwd_k
    try:
        best = sb.run(keep_density=True, arm=dict(name=f'r16_{tag}', habitat_alpha=0, fmax_scale=float(np.exp(th[-1])),
                           **({} if (choice_W is None and depart_scale is None and not settle_args) else dict(forward_args=dict(
                               **settle_args,
                               **({} if choice_W is None else dict(directional_weights=jnp.asarray(
                                   choice_W, jnp.asarray(x0['Sa_flat']).dtype))),
                               **({} if depart_scale is None else dict(departure_scale=jnp.asarray(
                                   depart_scale, jnp.asarray(x0['Sa_flat']).dtype))))))))
    finally:
        ap.forward_sim_age_structured = orig
        if a.mode == 'theta':
            fk.THETA['active'] = False; fk.restore()
        else:
            fk.LD['active'] = False
    ab = best['abundance']
    dens = best.pop('_density')
    near = sb.dist_km <= 150
    ser = dens[:, near].sum(1)
    def r_win(y0, y1):
        n0, n1 = ser[sb.years == y0][0], ser[sb.years == y1][0]
        return float(np.log(n1 / n0) / (y1 - y0)) if n0 > 0 and n1 > 0 else None
    release_r = {f'{y0}-{y1}': r_win(y0, y1) for y0, y1 in ((1951, 1962), (1962, 1971), (1971, 1985))}
    gl = np.asarray(g)[:, sb.rows, sb.cols]
    era = {str(yy): dict(p10=float(np.percentile(gl[sb.years == yy][0][sb.region['east']], 10)),
                         median=float(np.median(gl[sb.years == yy][0][sb.region['east']])),
                         p90=float(np.percentile(gl[sb.years == yy][0][sb.region['east']], 90)),
                         west_median=float(np.median(gl[sb.years == yy][0][sb.region['west']])))
           for yy in (1940, 1960, 1980, 2000, 2020)}
    out = dict(mode=a.mode, zsource=a.zsource, ridge_sd=a.ridge_sd, choice=a.choice, kappa=a.kappa, depart_kappa=a.depart_kappa, dep_ramp=a.dep_ramp, dep_mid=a.dep_mid,
               k_real=a.k_real, settle=a.settle, dep_floor=(float(1. / (1. + np.exp(-th[-2]))) if a.fit_floor else a.dep_floor), fit_floor=a.fit_floor,
               dep_s0=a.dep_s0, dep_s1=a.dep_s1, fitted_reference_nll=ref_nll, frozen_start_nll=float(v0),
               best_obj=float(res.fun) * OPT_SCALE, readout_nll=-sum(v['loglik'] for v in ab.values()),
               ridge_term=float(0.5 * np.sum((a.b_scale * th[1:1 + M]) ** 2) / a.ridge_sd ** 2), identity_max_rel=ident,
               converged=bool(res.success), n_eval=n[0], minutes=round((time.time() - t0) / 60, 1),
               fmax_scale=float(np.exp(th[-1])), intercept=float(th[0]), rho=float(np.exp(th[-3] if a.fit_floor else th[-2])) if a.mode == 'lowdens' else None,
               b=list(map(float, a.b_scale * th[1:1 + M])), b_scale=a.b_scale, readout_east_by_year=era,
               offsets=best['curves']['rel_10pct']['arrival_offset']['by_cohort'],
               never_reached=best['curves']['rel_10pct']['arrival_offset']['never'],
               loglik_by={k: v['loglik'] for k, v in ab.items()}, ref_loglik_by={k: v['loglik'] for k, v in ref['abundance'].items()},
               obs_over_expected={k: v['obs_over_expected'] for k, v in ab.items()}, release_area_r=release_r)
    (a.out / f'r16_{tag}_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('mode', 'zsource', 'frozen_start_nll', 'best_obj', 'readout_nll', 'ridge_term',
                                          'identity_max_rel', 'converged', 'n_eval', 'fmax_scale', 'intercept', 'rho')}), flush=True)


if __name__ == '__main__':
    main()

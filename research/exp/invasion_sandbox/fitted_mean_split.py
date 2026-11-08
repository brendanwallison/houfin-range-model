"""Why does fig 04's fitted annual mean sit below the observed one, and fall in 2021-25? (user, 2026-11-05: "is the disease
effect a little too strong?")

Fig 04 plots, per year, the mean over ALL BBS rows of expected_obs: the model mean with each route's effect at its MODE
(training routes) or at 0 (routes never seen in training). Two observation-layer reasons that mean is below the mean
count, independent of the dynamics:
  (1) Jensen: E[exp(u) | data] = exp(mode + var/2), not exp(mode). Laplace route posterior var = sigma^2 / (1 + H_r).
  (2) unseen routes at u = 0 are the MEDIAN route; the mean route is exp(sigma^2 / 2) higher (2.2x at sigma 1.25).
This prints, per year: observed mean; fig-04 mean (A); A with the Jensen correction for seen routes (B); B plus the
unseen-route mean correction (C); and the same three for TRAINING rows only and for the unseen-route share.
Laplace fits only (needs laplace_route_sigma). Run like laplace_check.py: fitted_mean_split.py --params NPZ --out JSON
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))
    import jax, jax.numpy as jnp
    from jax.scipy.special import gammaln
    from numpyro import handlers
    from harness import Sandbox, ap
    from src.config_utils import load_age_model_config
    sig_cfg = float(load_age_model_config()['population_model']['observation']['observer_effect']['sigma_u'])
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as z:
        lat.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
        sg = float(z['laplace_route_sigma']); m = float(z['laplace_route_mean']) if 'laplace_route_mean' in z.files else 0.
    eps = np.asarray(lat['observer_route_eps'])
    e_star = (eps * sig_cfg - m) / sg                                 # laplace_fit stores eps = (m + sg e*) / sig_cfg
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data={**lat, 'observer_route_eps': jnp.zeros_like(eps)}), 1)).get_trace(data=sb.data, prior_scale=1.)
    y_np = np.asarray(sb.data['observed_results'], float); w_np = np.asarray(sb.data['obs_weight'], float)
    ok = np.asarray(sb.obs_ok)
    train = np.isfinite(w_np) & (w_np > 0) & np.isfinite(y_np)
    idx = np.asarray(sb.data['obs_obsroute_idx']); on = idx >= 0; idc = np.maximum(idx, 0); n = eps.size
    mu0 = np.asarray(tr['expected_obs']['value'])                     # route effect 0 (first-year offset kept)
    kap = float(tr['obs_conc_kappa']['value']) if 'obs_conc_kappa' in tr else 0.
    phib = np.asarray(tr['obs_concentration']['value']) / np.power(np.maximum(mu0, 1e-3), kap)
    # per-route data curvature H_r at the mode (training rows), as in laplace_fit
    w = jnp.asarray(np.where(train, w_np, 0.)); y = jnp.asarray(np.where(train, y_np, 0.))
    mu0j = jnp.asarray(np.where(train, mu0, 1.)); phj = jnp.asarray(np.where(train, phib, 1.))

    def f(u):
        mu = mu0j * jnp.exp(u); ph = phj * jnp.power(jnp.maximum(mu, 1e-3), kap)
        return jnp.sum(w * (gammaln(y + ph) - gammaln(ph) + ph * (jnp.log(ph) - jnp.log(ph + mu)) + y * (jnp.log(jnp.maximum(mu, 1e-300)) - jnp.log(ph + mu))))
    u_row = jnp.asarray(np.where(on, m + sg * e_star[idc], 0.))
    d2 = np.asarray(jax.jvp(jax.grad(f), (u_row,), (jnp.ones_like(u_row),))[1])
    H = np.zeros(n); np.add.at(H, idc[on], -d2[on]); H = sg ** 2 * np.maximum(H, 0.)
    trained = np.zeros(n, bool); trained[idc[on & train]] = True
    seen = on & trained[idc]
    v_u = sg ** 2 / (1. + H)                                           # posterior variance of u_r (log scale)
    A = mu0 * np.exp(np.where(on, m + sg * e_star[idc], 0.)) * np.where(on & ~seen, np.exp(-m), 1.)   # fig 04: unseen at u = 0
    A = np.where(on & ~seen, mu0, A)
    B = np.where(seen, A * np.exp(.5 * v_u[idc]), A)
    C = np.where(on & ~seen, mu0 * np.exp(m + .5 * sg ** 2), B)
    yrs = np.asarray(sb.years)[np.asarray(sb.data['obs_time_indices'])]
    out = dict(params=str(a.params), sigma=sg, route_mean=m, years={})
    for t in range(1966, 2026):
        s = ok & (yrs == t)
        if not s.any():
            continue
        st = s & train
        out['years'][t] = dict(n=int(s.sum()), obs=float(y_np[s].mean()), A_fig04=float(A[s].mean()), B_jensen=float(B[s].mean()),
                               C_jensen_unseen=float(C[s].mean()), unseen_share=float((s & on & ~seen).sum() / s.sum()),
                               train_obs=float(y_np[st].mean()) if st.any() else None, train_A=float(A[st].mean()) if st.any() else None,
                               train_B=float(B[st].mean()) if st.any() else None)
    # regional split (West < 100W, Plains 100-90W, East >= 90W) of observed vs corrected model mean, per year
    try:
        import rasterio
        from rasterio.warp import transform
        from src.config_utils import load_data_config
        ref = str(load_data_config()['grid']['ref_raster']).replace('${HOUFIN_PROCESSED}', os.environ.get('HOUFIN_PROCESSED', ''))
        with rasterio.open(ref) as src:
            rr, cc = np.asarray(sb.data['obs_rows']), np.asarray(sb.data['obs_cols'])
            xx, yy = rasterio.transform.xy(src.transform, rr, cc)
            lon = np.asarray(transform(src.crs, 'EPSG:4326', xx, yy)[0])
        regs = dict(west=lon < -100, plains=(lon >= -100) & (lon < -90), east=lon >= -90)
        out['regions'] = {rn: {int(t): dict(n=int((ok & rm & (yrs == t)).sum()), obs=float(y_np[ok & rm & (yrs == t)].mean()),
                                            C=float(C[ok & rm & (yrs == t)].mean()))
                               for t in range(2005, 2026) if (ok & rm & (yrs == t)).any()} for rn, rm in regs.items()}
    except Exception as e:  # the regional split is a convenience; never block the main readout
        out['regions_error'] = repr(e)
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    for t in (1970, 1980, 1990, 1995, 2000, 2005, 2010, 2015, 2019, 2021, 2023, 2025):
        r = out['years'].get(t)
        if r:
            print(t, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == '__main__':
    main()

"""Per-row BBS log-likelihood at a fit (2026-11-06 overnight; review hypothesis: where do the BBS nats that prefer eta 1 live?).

At the fit's state (route effects at their saved conditional mode observer_route_eps), per BBS row: NB2 log pmf with the
model's own mean and concentration; plus per-route 0.5 * e^2 (route-effect penalty). Saves row metadata (year, lon, lat,
route idx, train mask, y, mu) so two fits can be differenced by region x era x count bin (row_ll_compare.py).
Usage (takeover runner, the fit's OWN overlay): row_ll_dump.py --params NPZ --out NPZ
"""
import argparse, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import jax.numpy as jnp
    from numpyro import handlers
    from scipy.special import gammaln
    from harness import Sandbox, ap
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as z:
        lat.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), 1)).get_trace(data=sb.data, prior_scale=1.)
    mu = np.asarray(tr['expected_obs']['value'], float)
    fn = tr['obs']['fn']
    conc = np.asarray(getattr(fn, 'concentration', getattr(getattr(fn, 'base_dist', fn), 'concentration', np.nan)), float)
    conc = np.broadcast_to(conc, mu.shape) if conc.size > 1 else np.full(mu.shape, float(conc))
    y = np.asarray(sb.data['observed_results'], float); idx = np.asarray(sb.data['obs_obsroute_idx'])
    w = np.asarray(sb.data['obs_weight'], float) if sb.data.get('obs_weight') is not None else np.ones_like(y)
    mask = tr['obs'].get('mask'); mask = np.ones_like(y, bool) if mask is None else np.asarray(mask, bool)
    yr = np.asarray(sb.years)[np.asarray(sb.data['obs_time_indices'])].astype(int)
    ci = sb.lookup[np.asarray(sb.data['obs_rows']), np.asarray(sb.data['obs_cols'])]
    ok = np.isfinite(y) & np.isfinite(mu) & (mu > 0)
    yy = np.where(ok, y, 0.); m = np.where(ok, mu, 1.); ph = conc
    ll = gammaln(yy + ph) - gammaln(ph) - gammaln(yy + 1) + ph * (np.log(ph) - np.log(ph + m)) + yy * (np.log(m) - np.log(ph + m))
    ll = np.where(ok, ll, np.nan)
    eps = np.asarray(lat['observer_route_eps'], float) if 'observer_route_eps' in lat else np.zeros(int(idx.max()) + 1)
    np.savez_compressed(a.out, ll=ll, mu=mu, y=y, idx=idx, year=yr, lon=sb.lon[ci], lat=sb.latd[ci],
                        train=(mask & (w > 0) & ok), route_eps=eps, conc=conc)
    print('rows', int(ok.sum()), 'train', int((mask & (w > 0) & ok).sum()), 'sum ll train', float(np.nansum(np.where(mask & (w > 0), ll, 0.))))


if __name__ == '__main__':
    main()

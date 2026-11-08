"""Rebound on a COMMON basis (reviewer, 2026-11-06): the model's paired observer x route ratio (2016-19 -> 2021-25) on exactly
the rows the data ratio uses, by region. Model = expected count per row with the fit's route effects (identical on both
windows of a pair, so route effects cancel in the ratio up to row composition).
Run like fitted_mean_split.py: paired_rebound.py --params NPZ --tag NAME"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--tag', required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))
    import jax.numpy as jnp, rasterio
    from rasterio.warp import transform
    from numpyro import handlers
    from harness import Sandbox, ap
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as z:
        lat.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), 1)).get_trace(data=sb.data, prior_scale=1.)
    mu = np.asarray(tr['expected_obs']['value'], float)
    y = np.asarray(sb.data['observed_results'], float); idx = np.asarray(sb.data['obs_obsroute_idx'])
    yrs = np.asarray(sb.years)[np.asarray(sb.data['obs_time_indices'])]
    rr, cc = np.asarray(sb.data['obs_rows']), np.asarray(sb.data['obs_cols'])
    with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as s:
        xx, yy = rasterio.transform.xy(s.transform, rr, cc); lon = np.asarray(transform(s.crs, 'EPSG:4326', xx, yy)[0])
    ok = np.isfinite(y) & (idx >= 0)
    out = {'tag': a.tag}
    for reg, (l0, l1) in dict(west=(-180, -100), plains=(-100, -90), east=(-90, 0)).items():
        m = ok & (lon >= l0) & (lon < l1)
        for (a0, a1, b0, b1) in ((2012, 2015, 2016, 2019), (2016, 2019, 2021, 2025)):
            A, B = m & (yrs >= a0) & (yrs <= a1), m & (yrs >= b0) & (yrs <= b1)
            both = np.intersect1d(np.unique(idx[A]), np.unique(idx[B]))
            sa, sb_ = np.isin(idx, both) & A, np.isin(idx, both) & B
            def ratio(v):   # ratio of per-unit means, summed (as the data check)
                pa = np.array([v[sa & (idx == k)].mean() for k in both]); pb = np.array([v[sb_ & (idx == k)].mean() for k in both])
                return float(pb.sum() / max(pa.sum(), 1e-12))
            out[f'{reg}_{a0}_{b1}'] = dict(n=int(both.size), data=round(ratio(y), 3), model=round(ratio(mu), 3))
    print(json.dumps(out))


if __name__ == '__main__':
    main()

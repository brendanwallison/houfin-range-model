"""How much does the likelihood charge for a thin, unrealistic presence? (user, 2026-11-06: 'the loss doesn't care')
For the route cells of chosen states: pseudo-zero coverage (rows with no route id, i.e. the 1902-39 absence rows) and the
NB2 cost of the model's expected counts on the observed BBS ZEROS before the state's first detection, plus the same cost if
the model had mu = 0 there. Uses the fit's own trace (expected_obs incl. fitted route effects, obs_concentration)."""
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--states', default='Iowa,Missour,Minneso')
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
    mu = np.asarray(tr['expected_obs']['value']); phi = np.asarray(tr['obs_concentration']['value'])
    y = np.asarray(sb.data['observed_results'], float); w = np.asarray(sb.data['obs_weight'], float)
    rr, cc = np.asarray(sb.data['obs_rows']), np.asarray(sb.data['obs_cols'])
    yrs = np.asarray(sb.years)[np.asarray(sb.data['obs_time_indices'])]
    ridx = np.asarray(sb.data['obs_obsroute_idx'])
    BBS = '/home/breallis/houfin/scratch/houfin/data/bbs_2026_release'
    routes = {}
    with open(f'{BBS}/Routes.csv', newline='', encoding='latin-1') as fh:
        for r in csv.DictReader(fh):
            try:
                routes[(int(r['StateNum']), int(r['Route']))] = (float(r['Latitude']), float(r['Longitude']), int(r['CountryNum']))
            except ValueError:
                pass
    land = np.asarray(sb.data['land_mask']).astype(bool)
    with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
        for st in a.states.split(','):
            snums = set(); first = None
            with open(f'{BBS}/States/{st}.csv', newline='', encoding='latin-1') as fh:
                for r in csv.DictReader(fh):
                    snums.add(int(r['StateNum']))
                    if r['AOU'].strip() == '05190' and int(r['SpeciesTotal']) > 0 and r['RPID'].strip() == '101':
                        first = min(first or 9999, int(r['Year']))
            cells = set()
            for (sn, rt), (la, lo, cn) in routes.items():
                if sn in snums and cn == 840:
                    xs, ys = transform('EPSG:4326', src.crs, [lo], [la]); r_, c_ = src.index(xs[0], ys[0])
                    if 0 <= r_ < land.shape[0] and 0 <= c_ < land.shape[1]:
                        cells.add((r_, c_))
            inst = np.array([(r_, c_) in cells for r_, c_ in zip(rr, cc)])
            pz = inst & (ridx < 0) & (yrs < 1940)
            zb = inst & (ridx >= 0) & (w > 0) & (y == 0) & (yrs < first)
            def nb0(m, ph):
                return ph * np.log1p(m / ph)            # -log P(y = 0) under NB2
            out = dict(state=st, first_bbs=first, route_cells=len(cells), pseudo_zero_rows_1902_39=int(pz.sum()),
                       pseudo_zero_cost=float(nb0(mu[pz], phi[pz]).sum()) if pz.any() else 0.,
                       bbs_zero_rows_before_first=int(zb.sum()), cost_of_model_presence_on_those_zeros=float(nb0(mu[zb], phi[zb]).sum()),
                       median_mu_on_those_rows=float(np.median(mu[zb])) if zb.any() else None,
                       share_rows_mu_gt_0_05=float((mu[zb] > .05).mean()) if zb.any() else None)
            print(json.dumps(out))


if __name__ == '__main__':
    main()

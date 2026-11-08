"""ESTABLISHMENT LAG readout (user, 2026-11-06): gauge-free comparison of when mating becomes common vs when BBS first
detects House Finch, per eastern state.

Veit & Lewis 1996 (p. 266): first occurrence -> first breeding lag 3-10 yr. A direct "first occurrence" comparison needs an
absolute one-bird threshold (the fictional count -> bird conversion), so the common anchor is the FIRST BBS DETECTION, which
data and model share through the same observation model:
  data : first year a standard BBS run (RPID 101) in the state recorded House Finch (raw 2026 release state files).
  model: for the routes actually run in that state each year, P(>= 1 detection) = 1 - prod_r (1 - p_r), with p_r the NB2
         detection at the route's cell, marginal over the route effect N(0, sigma^2) (Gauss-Hermite); the model's first-
         detection year = the MEDIAN of the implied first-detection-year distribution.
  mating: first year the mated fraction 1 - (1 + gamma N / k)^(-k) (gamma = ln2 / n50, N in route counts; the model's own
         absolute aggregated Allee) reaches theta (.25 / .5 / .75) in any of the state's route cells.
Model gap = model first detection - mating year; compare with (data first BBS detection - first breeding record) from the
literature. Usage (takeover runner, the fit's overlay): state_lag.py --params NPZ --out JSON
"""
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np

EAST = ['Iowa', 'Illinoi', 'Indiana', 'Arkansa', 'Vermont', 'Ohio', 'Kentuck', 'Tenness', 'Michiga', 'Wiscons', 'Minneso',
        'Missour', 'Alabama', 'Mississ', 'Georgia', 'Louisia', 'Florida', 'NCaroli', 'SCaroli', 'Virgini', 'W_Virgi',
        'Pennsyl', 'NYork', 'Maine', 'NHampsh', 'Massach', 'Connect', 'RhodeIs', 'NJersey', 'Delawar', 'Marylan',
        'Ontario', 'Quebec']
BBS = '/home/breallis/houfin/scratch/houfin/data/bbs_2026_release'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))
    import jax.numpy as jnp
    import rasterio
    from numpyro import handlers
    from harness import Sandbox, ap
    from src.config_utils import load_age_model_config
    cfg = load_age_model_config()['population_model']
    n50 = float(np.log1p(np.exp(float(cfg['pinned_sites']['n50_raw']))))           # softplus, route counts
    kagg = float(cfg['allee_prior'].get('aggregation_k') or 1e9)
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as z:
        lat.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
        sigma = float(z['laplace_route_sigma']) if 'laplace_route_sigma' in z.files else 0.97
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), 1)).get_trace(data=sb.data, prior_scale=1.)
    scale = float(sb.data['pop_scalar'])
    N = (np.asarray(tr['Na_grid']['value']) + np.asarray(tr['Nj_grid']['value'])) * scale        # (T, Ny, Nx) route counts
    conc = float(np.ravel(np.asarray(tr['concentration']['value']))[0])
    kap = float(np.ravel(np.asarray(tr['obs_conc_kappa']['value']))[0]) if 'obs_conc_kappa' in tr else 0.
    yrs = np.asarray(sb.years).astype(int)
    land = np.asarray(sb.data['land_mask']).astype(bool); lr, lc = np.nonzero(land)
    xg, wg = np.polynomial.hermite_e.hermegauss(24); wg = wg / wg.sum()

    def p_detect(mu):   # marginal over the route effect; NB2 zero prob with phi = conc * max(mu,1e-3)^kappa
        m = mu[..., None] * np.exp(sigma * xg)
        ph = conc * np.power(np.maximum(m, 1e-3), kap)
        return 1. - np.sum(np.power(1. + m / ph, -ph) * wg, axis=-1)

    def mated(Nc):
        x = np.log(2.) * Nc / n50
        return -np.expm1(-kagg * np.log1p(x / kagg))
    routes = {}
    with open(f'{BBS}/Routes.csv', newline='', encoding='latin-1') as fh:
        for r in csv.DictReader(fh):
            try:
                routes[(int(r['CountryNum']), int(r['StateNum']), int(r['Route']))] = (float(r['Latitude']), float(r['Longitude']))
            except ValueError:
                pass
    with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
        from rasterio.warp import transform

        def cell(la, lo):
            xs, ys = transform('EPSG:4326', src.crs, [lo], [la]); r, c = src.index(xs[0], ys[0])
            if 0 <= r < land.shape[0] and 0 <= c < land.shape[1] and land[r, c]:
                return r, c
            d = (lr - r) ** 2 + (lc - c) ** 2; i = int(np.argmin(d)); return int(lr[i]), int(lc[i])
        out = {'n50_route_counts': n50, 'aggregation_k': kagg, 'sigma': sigma, 'concentration': conc, 'kappa': kap, 'states': {}}
        for st in EAST:
            runs = {}; hofi = set()
            with open(f'{BBS}/States/{st}.csv', newline='', encoding='latin-1') as fh:
                for r in csv.DictReader(fh):
                    if r['RPID'].strip() != '101':
                        continue
                    key = (int(r['CountryNum']), int(r['StateNum']), int(r['Route'])); y = int(r['Year'])
                    if key not in routes:
                        continue
                    runs.setdefault(y, set()).add(key)
                    if r['AOU'].strip() == '05190' and int(r['SpeciesTotal']) > 0:
                        hofi.add(y)
            if not runs:
                continue
            cells = {k: cell(*routes[k]) for ks in runs.values() for k in ks}
            ucells = sorted(set(cells.values())); cr = np.array([c[0] for c in ucells]); cc = np.array([c[1] for c in ucells])
            surv = 1.; dist_first = {}; mate = {th: None for th in (.25, .5, .75)}
            mf_series = np.array([mated(N[t][cr, cc]).max() if cr.size else 0. for t in range(len(yrs))])
            sustained, crossings = {}, {}
            for th in (.25, .5):
                above = mf_series >= th
                crossings[str(th)] = int(np.sum(above[1:] & ~above[:-1]))       # upward crossings (re-establishments count)
                ok = [i for i in range(len(yrs) - 4) if above[i:i + 5].all()]
                sustained[str(th)] = int(yrs[ok[0]]) if ok else None
            for t, y in enumerate(yrs):
                Nm = N[t][cr, cc]; mf = mated(Nm).max() if Nm.size else 0.
                for th in mate:
                    if mate[th] is None and mf >= th:
                        mate[th] = int(y)
                if y in runs:
                    rc = np.array([cells[k] for k in runs[y]])
                    pany = 1. - np.prod(1. - p_detect(N[t][rc[:, 0], rc[:, 1]]))
                    dist_first[int(y)] = surv * pany; surv *= (1. - pany)
            cum, med = 0., None
            for y in sorted(dist_first):
                cum += dist_first[y]
                if med is None and cum >= .5:
                    med = y
            out['states'][st] = dict(n_routes=len(cells), n_cells=len(ucells), data_first_bbs=min(hofi) if hofi else None,
                                     first_run=min(runs), model_first_bbs_median=med,
                                     mate_year={str(k): v for k, v in mate.items()}, sustained_year=sustained, upward_crossings=crossings,
                                     model_gap_mate50_to_detect=(med - mate[.5]) if (med and mate[.5]) else None)
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'states'}))
    for st, d in out['states'].items():
        print(f"{st:9s} data_firstBBS {d['data_first_bbs']} (first run {d['first_run']}) | model_firstBBS {d['model_first_bbs_median']} | "
              f"mate .25/.5/.75 {d['mate_year']['0.25']}/{d['mate_year']['0.5']}/{d['mate_year']['0.75']} | gap {d['model_gap_mate50_to_detect']}")


if __name__ == '__main__':
    main()

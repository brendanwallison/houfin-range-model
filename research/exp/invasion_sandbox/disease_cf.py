"""Disease depression implied by a fit (2026-11-06, FINAL ROUND; literature target: eastern abundance settled at ~.4-.5 of
its pre-MG trajectory -- Hochachka & Dhondt 2000 PNAS; Henschen et al. 2023 "about 40% of pre-MG numbers").

Replays the fitted point with and without the epizootic (harness disease_scale 0; no refit) and reports the ratio of
eastern abundance (with / without disease) by year, over all eastern land cells and over the cells that were densest
before 1994 (top third of 1990-93 density; the literature's "high-density populations").
Usage (takeover runner, the fit's overlay): disease_cf.py --params NPZ --out JSON
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    from harness import Sandbox
    sb = Sandbox(os.environ['SBX_REFINED'])
    with np.load(a.params) as z:
        sb.lat = dict(sb.lat); sb.lat.update({k: np.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
    yrs = np.asarray(sb.years).astype(int)
    on = sb.run(dict(name='fitted'), keep_density=True).pop('_density')
    off = sb.run(dict(name='disease_off', disease_scale=0.0), keep_density=True).pop('_density')
    east = sb.region['east']
    pre = (yrs >= 1990) & (yrs <= 1993)
    d0 = on[pre][:, east].mean(0); top = np.zeros_like(east); idx = np.flatnonzero(east)
    top[idx[d0 >= np.quantile(d0, 2 / 3)]] = True
    out = dict(params=str(a.params), years={})
    for y in range(1994, 2026):
        i = int(np.nonzero(yrs == y)[0][0])
        out['years'][y] = dict(east=float(on[i, east].sum() / max(off[i, east].sum(), 1e-30)),
                               east_dense=float(on[i, top].sum() / max(off[i, top].sum(), 1e-30)))
    w = lambda y0, y1, k: float(np.mean([out['years'][y][k] for y in range(y0, y1 + 1)]))
    out['summary'] = {f'{k}_{y0}_{y1}': round(w(y0, y1, k), 3) for k in ('east', 'east_dense') for (y0, y1) in ((1997, 2004), (2005, 2015), (2016, 2025))}
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['summary']))


if __name__ == '__main__':
    main()

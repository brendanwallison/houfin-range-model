"""Realised growth of the eastern population in a replayed fit vs the literature anchors (unit-free).

Anchors (literature agent, 2026-11-03): ~40-100 founders (1940) -> ~280 birds in 1951 (Elliott & Arbib 1953) gives
r = .12-.18 / yr including the 1947-48 crash; north-eastern CBC birds per party-hour .06 (1962) -> .42 (1971) gives
r about .22 (Bock & Lepthien 1976, abstract-only). Readout: r = log(N1 / N0) / (t1 - t0) of the model's EAST total
(any unit cancels), plus the north-east box (lon > -80, lat > 38) for the CBC-style window, for the reference arm
and any arms given. Also r of the release-cell neighbourhood (within 150 km) for the 1940-51 window.
Usage (inside sbx_env.sh): python realized_r.py --out DIR [--spec arms.json]
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from harness import Sandbox, BIRDS_PER_KM2_PER_UNIT

WINDOWS = [(1941, 1951), (1951, 1962), (1962, 1971), (1971, 1985), (1985, 1994)]


def rates(sb, row, tot):
    yrs = sb.years
    ne = sb.region['east'] & (sb.lon > -80) & (sb.latd > 38)
    near = sb.dist_km <= 150
    out = {}
    for name, m in (('east', sb.region['east']), ('north_east', ne), ('release_150km', near)):
        series = tot[:, m].sum(1) * sb.cell_km2 * BIRDS_PER_KM2_PER_UNIT
        r = {}
        for a, b in WINDOWS:
            n0, n1 = series[yrs == a][0], series[yrs == b][0]
            r[f'{a}-{b}'] = float(np.log(n1 / n0) / (b - a)) if n0 > 0 and n1 > 0 else None
        out[name] = r
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--spec', type=Path)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    arms = [dict(name='fitted')] + (json.loads(a.spec.read_text())['arms'] if a.spec else [])
    res = {}
    for arm in arms:
        row = sb.run(arm, keep_density=True)
        res[arm['name']] = rates(sb, row, row.pop('_density'))
        print(arm['name'], json.dumps(res[arm['name']]), flush=True)
    (a.out / 'realized_r.json').write_text(json.dumps(dict(anchors={'1941-1951': [.12, .18], '1962-1971': .22},
                                                          step=os.environ.get('SBX_STEP'), arms=res), indent=1) + '\n')


if __name__ == '__main__':
    main()

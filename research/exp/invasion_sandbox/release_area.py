"""F1: what happens near the release in the fitted replay, year by year (current base anomaly: 1951-62 stall)?

Within --radius km of the release cell: total density, occupied cells, and density-weighted means of Fmax, Sa, Sj,
potential lambda, mated fraction (absolute or relative Allee as fitted), crowding factor (power-aware), N/K, and
replacement R0 = F Sj / (1 - Sa) (yearling) -- plus the share of the area's density in cells with R0 < 1 (sinks).
Unit-free except density (model units). Prints selected years and writes JSON.
Usage (inside sbx_env.sh): python release_area.py --out DIR [--radius 160]
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from harness import Sandbox, capture, to_land, potential_lambda
from src.model import age_forward as af
try:
    from src.model.density_regulation import CROWDING_POWER as POWER
except ImportError:
    POWER = 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--radius', type=float, default=160.)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    row = sb.run(dict(name='fitted'), keep_density=True); N = row.pop('_density')
    x, _ = capture(sb.data, sb.lat)
    T, Nx = len(sb.years), int(np.asarray(sb.data['land_mask']).shape[1])
    Sa, Sj, F, K, c = (to_land(x[k], sb.rows, sb.cols, T, Nx) for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat', 'K_flat', 'c_flat'))
    yb = bool(x.get('yearling_breeding')); gamma = float(np.asarray(x['allee_gamma'])); rel = bool(x.get('allee_relative'))
    m = sb.dist_km <= a.radius
    lam = potential_lambda(Sa, Sj, F, yb)
    R0 = (F * Sj / np.maximum(1 - Sa, 1e-9)) if yb else (F * Sa * Sj / np.maximum(1 - Sa, 1e-9))
    out = dict(radius_km=a.radius, n_cells=int(m.sum()), release_cell=int(sb.rel), years={})
    for i, y in enumerate(sb.years):
        if y < 1945 or y > 1975:
            continue
        n = N[i][m]; w = n / max(n.sum(), 1e-30); nk = n / np.maximum(K[i][m], 1e-12)
        mated = np.asarray(af.allee_mated_fraction(gamma * (nk if rel else n)))
        crowd = 1. / (1. + c[i][m] * nk ** POWER)
        out['years'][int(y)] = dict(total=float(n.sum()), occupied=int((n >= .01 * sb.base_med).sum()),
                                    F=float((w * F[i][m]).sum()), Sa=float((w * Sa[i][m]).sum()), Sj=float((w * Sj[i][m]).sum()),
                                    lambda0=float((w * lam[i][m]).sum()), mated=float((w * mated).sum()), crowding=float((w * crowd).sum()),
                                    N_over_K=float((w * nk).sum()), sink_share=float(w[R0[i][m] < 1].sum()))
    (a.out / 'release_area.json').write_text(json.dumps(out, indent=1) + '\n')
    for y, r in out['years'].items():
        if y % 2 == 1 or y in (1950, 1952):
            print(y, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == '__main__':
    main()

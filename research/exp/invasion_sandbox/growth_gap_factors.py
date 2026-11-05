"""Decompose the potential-vs-realised growth gap in eastern occupied cells into the model's own factors, by era.

Factors at each cell-year (fitted replay, census before reproduction): crowding 1 / (1 + c N/K) and the Allee mated
fraction (as in age_forward.reproduction_age_structured), plus the crowding half-density 1/c (N/K at which
crowding halves fecundity) and fecundity F. Medians over occupied eastern cells (N >= 1% of the reference modern
occupied median), for all cells and for FILLING cells (.1 <= N/K < .7). Unit-free.
Usage (inside sbx_env.sh): python growth_gap_factors.py --out DIR
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from harness import Sandbox, capture, to_land
from src.model import age_forward as af
try:
    from src.model.density_regulation import CROWDING_POWER as POWER   # current lineage (power 1 or 2)
except ImportError:
    POWER = 1.0                                                        # older code: Beverton-Holt power 1

ERAS = [(1941, 1955), (1956, 1965), (1966, 1975), (1976, 1990), (1991, 2010)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    row = sb.run(dict(name='fitted'), keep_density=True)
    N = row.pop('_density')
    x, _ = capture(sb.data, sb.lat)
    T, Nx = len(sb.years), int(np.asarray(sb.data['land_mask']).shape[1])
    F, K, c, Sa, Sj = (to_land(x[k], sb.rows, sb.cols, T, Nx) for k in ('Fmax_flat', 'K_flat', 'c_flat', 'Sa_flat', 'Sj_flat'))
    gamma = float(np.asarray(x['allee_gamma'])); rel_allee = bool(x.get('allee_relative'))
    occ_thr = .01 * sb.base_med; east = sb.region['east']
    out = dict(gamma=gamma, allee_relative=rel_allee, crowding_power=POWER, eras={})
    for a0, b0 in ERAS:
        idx = [i for i, y in enumerate(sb.years[:-1]) if a0 <= y <= b0]
        rec = {}
        for label, lo, hi in (('all_occupied', 0, 1e9), ('filling', .1, .7)):
            vals = {k: [] for k in ('F', 'Sa', 'Sj', 'half_density_1_over_c', 'crowding', 'allee', 'rel')}
            for i in idx:
                rel = N[i] / np.maximum(K[i], 1e-12)
                m = east & (N[i] >= occ_thr) & (rel >= lo) & (rel < hi)
                if not m.any():
                    continue
                crowd = 1. / (1. + c[i][m] * rel[m] ** POWER)
                al = np.asarray(af.allee_mated_fraction(gamma * (rel[m] if rel_allee else N[i][m])))
                for k, v in (('F', F[i][m]), ('Sa', Sa[i][m]), ('Sj', Sj[i][m]),
                             ('half_density_1_over_c', 1. / np.maximum(c[i][m], 1e-9)),
                             ('crowding', crowd), ('allee', al), ('rel', rel[m])):
                    vals[k] += list(v)
            if vals['F']:
                rec[label] = {k: float(np.median(v)) for k, v in vals.items()} | dict(n=len(vals['F']))
        out['eras'][f'{a0}-{b0}'] = rec
        print(f'{a0}-{b0}', {lab: {k: round(v, 3) for k, v in r.items()} for lab, r in rec.items()}, flush=True)
    (a.out / 'growth_gap_factors.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()

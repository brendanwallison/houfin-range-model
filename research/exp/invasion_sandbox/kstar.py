"""Re-measure the Allee pin on [cur] (review 2, item 6): n50 / K* in the model's own density units.

E055 pinned n50 = V&L's n50/K* (.0197; V&L's r sensitivity .0066-.033) x 51a's median capacity (3.0 route counts) -- a
calibration transferred from another run. Here: n50_model = ln 2 / allee_gamma (the forward's own gamma, so no unit
bridge) and K* = median capacity K (real Z) over 2010-2025 in cells with a BBS detection, by region. Prints n50 / K*
for the pin and for fitted multipliers (n50 x m -> ratio x m).
Usage (inside sbx_env.sh): python kstar.py --out FILE.json
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from harness import Sandbox, capture, to_land


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sb = Sandbox(os.environ['SBX_REFINED'])
    x, _ = capture(sb.data, sb.lat)
    n50 = float(np.log(2.) / np.asarray(x['allee_gamma']).ravel()[0])
    K = np.asarray(x['K_flat']); K = to_land(K, sb.rows, sb.cols, sb.T, sb.Nx) if K.ndim > 1 else np.broadcast_to(K, (sb.T, len(sb.rows)))
    modern = (sb.years >= 2010) & (sb.years <= 2025)
    occ = np.isfinite(sb.first_obs)
    out = dict(n50_model_units=n50, vl_ratio=.0197, vl_range=[.0066, .033], multipliers={})
    for reg in ('east', 'plains', 'west'):
        m = sb.region[reg] & occ
        ks = float(np.median(K[modern][:, m]))
        out[reg] = dict(K_star=ks, n_cells=int(m.sum()), pin_ratio=n50 / ks)
    for mult in (1., 1.67, 2.9, 5.2, 6.3, 8.0, 13.3):
        out['multipliers'][str(mult)] = {reg: n50 * mult / out[reg]['K_star'] for reg in ('east', 'plains', 'west')}
    a.out.write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

"""Allee-dead / missed-cell check for a replayed arm (R26 T0's strong-Allee solution vs the fit and the uncorrected base).

For each region (25-49N East / Plains / West): share of cells with a BBS detection (observed occupied) that the arm never
occupies (density never >= 1% of the reference occupied median, the sandbox's occupancy threshold), and the share
never occupied by 2025 among all land cells; plus the modern (2010-25) median model/observed abundance ratio over
observed-occupied cells. Arms: fitted; frozen readouts + K real (base); the same + Fmax / n50 scales from a summary.
Usage (inside sbx_env.sh): python allee_dead.py --summary T0_summary.json --out FILE.json
"""
import argparse, json, os
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from harness import Sandbox, capture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); s = json.loads(a.summary.read_text())
    sb = Sandbox(os.environ['SBX_REFINED'])
    x_real, _ = capture(sb.data, sb.lat)
    arms = {'fitted': dict(name='fitted'),
            'frozen_Kreal': dict(name='frozen_Kreal', habitat_alpha=0, forward_args={'K_flat': x_real['K_flat']}),
            'T0': dict(name='T0', habitat_alpha=s.get('alpha', 0), fmax_scale=s['fmax_scale'],
                       forward_args={'K_flat': x_real['K_flat'],
                                     'allee_gamma': jnp.asarray(x_real['allee_gamma']) / s['n50_scale']})}
    obs_occ = np.isfinite(sb.first_obs)
    res = {}
    for nm, arm in arms.items():
        r = sb.run(arm, keep_density=True); dens = r.pop('_density')
        occ_ever = (dens >= .01 * sb.base_med).any(0)
        row = {}
        for reg in ('east', 'plains', 'west'):
            m = sb.region[reg]
            row[reg] = dict(observed_occupied=int((m & obs_occ).sum()),
                            missed_share=float((m & obs_occ & ~occ_ever).sum() / max((m & obs_occ).sum(), 1)),
                            never_occupied_share=float((m & ~occ_ever).sum() / m.sum()))
        row['offsets'] = r['curves']['rel_10pct']['arrival_offset']['by_cohort']
        row['never_reached'] = r['curves']['rel_10pct']['arrival_offset']['never']
        res[nm] = row
    a.out.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res))


if __name__ == '__main__':
    main()

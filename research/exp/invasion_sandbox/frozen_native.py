"""Frozen-1902 native test (user, 2026-11-06): with habitat frozen at its 1902 state and only the native (western) seed,
the invasion of the East should NOT happen. Replays the fitted point (no refit) under:
  native_fitted   : native seed only (no Long Island release), fitted habitat
  native_frozen   : native seed only, habitat (Z and Z_disp) frozen at the 1902-1906 mean; trend_routing deviation zeroed
  both_frozen     : release + native, habitat frozen
Reports East (lon >= -95) and Plains (-100..-95) total density by decade, and the first year the native population
occupies cells east of 95W / 90W / 85W (density >= 1% of the fitted modern occupied median).
Usage (takeover runner, fit overlay): frozen_native.py --params NPZ --out JSON
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import jax.numpy as jnp
    from harness import Sandbox
    sb = Sandbox(os.environ['SBX_REFINED'])
    with np.load(a.params) as z:
        sb.lat = dict(sb.lat); sb.lat.update({k: np.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
    yrs = np.asarray(sb.years).astype(int)
    base = sb.run(dict(name='fitted'), keep_density=True); Nfit = base.pop('_density')
    thr = .01 * sb.base_med
    data0 = sb.data
    out = dict(params=str(a.params), threshold=thr, arms={})
    arms = [('fitted', dict(name='fitted'), False), ('native_fitted', dict(name='native_fitted', region='west_only'), False),
            ('native_frozen', dict(name='native_frozen', region='west_only', habitat_alpha=0.0, alpha_ref=[1902, 1906]), True),
            ('both_frozen', dict(name='both_frozen', habitat_alpha=0.0, alpha_ref=[1902, 1906]), True)]
    for name, arm, frozen in arms:
        sb.data = dict(data0)
        if frozen and 'trend_zdev' in sb.data:
            sb.data['trend_zdev'] = jnp.zeros_like(jnp.asarray(sb.data['trend_zdev']))
        try:
            N = Nfit if name == 'fitted' else sb.run(arm, keep_density=True).pop('_density')
        finally:
            sb.data = data0
        east, plains = sb.lon >= -95, (sb.lon >= -100) & (sb.lon < -95)
        r = dict(east_total={int(y): float(N[i][east].sum()) for i, y in enumerate(yrs) if y % 10 == 0 or y == 2025},
                 plains_total={int(y): float(N[i][plains].sum()) for i, y in enumerate(yrs) if y % 10 == 0 or y == 2025})
        for lon in (-95, -90, -85):
            occ = (N[:, sb.lon >= lon] >= thr).any(1)
            r[f'first_occ_east_of_{-lon}W'] = int(yrs[np.argmax(occ)]) if occ.any() else None
        out['arms'][name] = r
        print(name, {k: v for k, v in r.items() if k.startswith('first')}, 'East 1950/1980/2010:',
              [round(r['east_total'].get(y, 0), 2) for y in (1950, 1980, 2010)], 'Plains 1950/1980/2010:',
              [round(r['plains_total'].get(y, 0), 2) for y in (1950, 1980, 2010)])
    a.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()

"""L1: reproduce Veit & Lewis 1996 (1-D, and a 2-D check), then compare with the observed eastern timeline.

Paper setup (journal pp. 264-268): 1-D domain, reflecting at the Long Island coast, 1500 km; s .67, c 1.5,
delta 1.5, r .15 (r 0 / .25 contrasts); Weibull kernels juvenile a .468 b 41.056 km, adult a .859 b 76.893 km;
22% of adults and 27% of juveniles recovered where banded; p(N) = min(N / N2*, 1); start .5 birds / km^2 in a
10 km semicircle; range edge at .1 birds / km^2. Reported: lag to ~1960 (year 20), then ~40 km/yr over
years 30-50; under-predicted spread in the first 10 yr, slightly over-predicted later.
Observed anchors (project record): first CBC record > 54 km from Jones Beach 1955 (year 15), > 100 km 1956;
occupied radius ~200-250 km by 1965 (year 25); Mississippi (~1500 km) by ~1990 (year 50).

Usage: python run_vl_validation.py [out_dir]
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import proto

OBS = {15: 54., 16: 100., 25: 225., 50: 1500.}


def run(ndim, extent, dx, r, stay, years=61):
    g = proto.grid(ndim, dx, extent)
    kA = proto.weibull_kernel(g, .859, 76.893, stay=.22 if stay else 0.)
    kJ = proto.weibull_kernel(g, .468, 41.056, stay=.27 if stay else 0.)
    P = proto.vl_params(r=r)
    N0 = proto.release_patch(g, .5, 10.)
    t0 = time.time()
    hist, _ = proto.simulate('vl', P, g, kA, kJ, years=years, N0=N0, thr=.1)
    out = proto.lag_and_speeds(hist)
    out.update(ndim=ndim, extent_km=extent, dx_km=dx, r=r, stay=stay, K=P['K'], seconds=round(time.time() - t0, 3),
               obs_error_km={y: out['radius_at'].get(y, float(hist[y]['radius_km'])) - v for y, v in OBS.items()
                             if y < len(hist)})
    out['radius'] = [h['radius_km'] for h in hist]
    return out


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ndim, extent, dx in [(1, 1500., 5.), (1, 6000., 5.), (2, 1500., 10.), (2, 3000., 10.)]:
        for stay in (True, False):
            for r in (0., .01, .02, .05, .15, .25):
                if ndim == 2 and not stay:
                    continue
                rows.append(run(ndim, extent, dx, r, stay))
                q = rows[-1]
                print(f"{ndim}D ext{extent:.0f} stay={stay!s:5} r={r:.2f}: lag {q['lag_year']}, "
                      f"early {q['early_speed']}, late {q['late_speed']}, R(15,25,50)="
                      f"{[round(q['radius'][y]) for y in (15, 25, 50)]} [{q['seconds']}s]", flush=True)
    (out_dir / 'VL1996_VALIDATION.json').write_text(json.dumps(dict(observed_anchors_km=OBS, runs=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()

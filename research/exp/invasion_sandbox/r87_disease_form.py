"""R87 (user: "is the functional form of the disease effect -- arrival dates, recovery through resistance -- another
lever?"): read out how [cur]'s fitted disease process uses each part of its form.

m(cell, t, N) = ceiling x modifier(x) x gate(t - onset) x (1 - rec (1 - exp(-age / tau_rec))) x hill(N)  (age_fields).
Readouts: the fitted disease latents (raw); per region the onset (first year the amplitude exceeds half its cell maximum)
vs the documented arrival map's onset; the amplitude trajectory by years since local onset (resistance / recovery shape);
the spread of the spatial severity field (how much severity varies across cells within a region).
Usage (inside sbx_env.sh): python r87_disease_form.py --out FILE.json
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
    T, N = len(sb.years), len(sb.rows); yrs = np.asarray(sb.years)
    amp = np.asarray(x['disease_m_amp']).reshape(T, -1)
    amp = amp if amp.shape[1] == N else to_land(np.asarray(x['disease_m_amp']), sb.rows, sb.cols, T, sb.Nx)
    lat = {k: np.asarray(v).round(4).tolist() for k, v in sb.lat.items() if k.startswith('disease')}
    occ = np.isfinite(sb.first_obs)
    mx = amp.max(0); onset = np.array([yrs[np.argmax(amp[:, c] >= .5 * mx[c])] if mx[c] > 1e-3 else np.nan for c in range(N)])
    # documented arrival map (data) if present
    doc = None
    for k in ('disease_onset_year', 'disease_arrival_year', 'disease_onset'):
        if k in sb.data:
            doc = np.asarray(sb.data[k]).ravel()[:N]; break
    out = dict(latents=lat, by_region={})
    for reg in ('east', 'plains', 'west'):
        mo = sb.region[reg] & occ & np.isfinite(onset)
        traj = {}
        for k in (0, 2, 5, 10, 15, 20, 25):
            v = [amp[np.flatnonzero(yrs == onset[c] + k)[0], c] for c in np.flatnonzero(mo) if onset[c] + k <= yrs[-1]]
            traj[k] = float(np.mean(v)) if v else None
        out['by_region'][reg] = dict(n=int(mo.sum()), onset_median=float(np.median(onset[mo])), onset_q10_q90=[float(q) for q in np.quantile(onset[mo], [.1, .9])],
                                     peak_amplitude_median=float(np.median(mx[mo])), peak_amplitude_q10_q90=[float(q) for q in np.quantile(mx[mo], [.1, .9])],
                                     amplitude_by_years_since_onset=traj,
                                     documented_onset_median=(float(np.median(doc[mo])) if doc is not None else None))
    a.out.write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

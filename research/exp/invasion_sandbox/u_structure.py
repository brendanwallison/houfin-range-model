"""Where do per-cell establishment corrections (R23 / R23b / R24 / R25 u maps) sit? Spatial structure readout.

For each u map (npz with u over land cells) and each region (East / Plains / West, 25-49N), Spearman correlations of u
with: the fitted habitat change in potential growth (real-Z capture, mean 2006-25 minus mean 1951-65, lambda = Sa + Sj F),
the frozen-habitat potential growth, log K (real Z, 1966-2025 mean), the K change, distance from the release cell
(Jones Beach), and the first BBS detection year. Plus a standardised multiple regression of u on all of them (R^2).
Usage (inside sbx_env.sh): python u_structure.py --out FILE.json U1.npz [U2.npz ...]
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from harness import Sandbox, capture, edit_habitat, to_land, potential_lambda

REL_LON, REL_LAT = -73.5, 40.6


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('maps', nargs='+', type=Path)
    a = p.parse_args()
    sb = Sandbox(os.environ['SBX_REFINED'])
    x_real, _ = capture(sb.data, sb.lat)
    x_froz, _ = capture(edit_habitat(sb.data, sb.years, 0., [1966, 2025]), sb.lat)
    yl = bool(x_real.get('yearling_breeding'))

    def lam(x):
        Sa, Sj, F = (to_land(np.broadcast_to(np.asarray(x[k]), np.shape(x['Sa_flat'])), sb.rows, sb.cols, sb.T, sb.Nx)
                     if np.ndim(x[k]) > 1 else np.asarray(x[k])[None] for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat'))
        return potential_lambda(Sa, Sj, F, yl)
    def era(m, y0, y1):
        t = (sb.years >= y0) & (sb.years <= y1)
        return m[t].mean(0) if m.shape[0] == sb.T else m[0]
    Lr, Lf = lam(x_real), lam(x_froz)
    K = np.asarray(x_real['K_flat']); K = to_land(K, sb.rows, sb.cols, sb.T, sb.Nx) if K.ndim > 1 else K[None]
    lat1, lon1, lat2, lon2 = map(np.radians, (sb.latd, sb.lon, REL_LAT, REL_LON))
    dist = 6371 * 2 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))
    cov = dict(dlambda_fit=era(Lr, 2006, 2025) - era(Lr, 1951, 1965), lambda_frozen=era(Lf, 1966, 2025),
               logK=np.log(np.maximum(era(K, 1966, 2025), 1e-12)),
               dlogK=np.log(np.maximum(era(K, 2006, 2025), 1e-12)) - np.log(np.maximum(era(K, 1951, 1965), 1e-12)),
               dist_release_km=dist, first_detect=sb.first_obs)
    res = {}
    for f in a.maps:
        u = np.load(f)['u'].astype(float)
        if u.size != len(sb.rows):
            res[f.name] = 'skipped (not per cell)'; continue
        rr = {}
        for reg in ('east', 'plains', 'west'):
            m = sb.region[reg]
            row = dict(n=int(m.sum()), u_mean=float(u[m].mean()))
            for k, v in cov.items():
                ok = m & np.isfinite(v)
                row[f'rho_{k}'] = float(spearmanr(u[ok], v[ok])[0]) if ok.sum() > 10 else None
            names = [k for k in cov if k != 'first_detect']
            Xs = np.column_stack([cov[k][m] for k in names]); ok = np.all(np.isfinite(Xs), 1)
            Xs = (Xs[ok] - Xs[ok].mean(0)) / (Xs[ok].std(0) + 1e-12); yv = u[m][ok]
            A = np.column_stack([np.ones(len(yv)), Xs]); beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
            r2 = 1 - np.sum((yv - A @ beta) ** 2) / np.sum((yv - yv.mean()) ** 2)
            row['ols_r2'] = float(r2); row['ols_std_beta'] = {k: float(b) for k, b in zip(names, beta[1:])}
            rr[reg] = row
        res[f.name] = rr
    a.out.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

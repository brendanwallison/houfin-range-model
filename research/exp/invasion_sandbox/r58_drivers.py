"""R58 (review 4 item 5): is the fit's place-specific time change in Z-driven rates habitat-like or effort-like?

Readout only, CPU. For the fit ([cur], real Z): per land cell, the change in log K and in potential lambda (Sa + Sj Fmax)
between 1966-75 and 2010-20. Candidate drivers per cell, same windows: change in standardised log1p BUI q90 and HYDE
population density (r7_shared_calendar.bui_z / hyde_z, the encoder's own yearly states), and BBS effort / observer
composition from the observation rows (routes surveyed per year, share of first-year observers, observer-route
turnover = distinct observer-route ids per route). Spearman within region (East / Plains / West, 25-49N), and the
East-minus-West yearly trajectories of mean log K vs the same drivers (correlation after removing a linear trend).
Usage (inside sbx_env.sh): python r58_drivers.py --out FILE.json
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from harness import Sandbox, capture, to_land, potential_lambda
from r7_shared_calendar import bui_z, population_z


def window(a, years, y0, y1):
    t = (years >= y0) & (years <= y1)
    return np.nanmean(a[t], 0)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sb = Sandbox(os.environ['SBX_REFINED']); yrs = sb.years
    x, _ = capture(sb.data, sb.lat)
    K = to_land(x['K_flat'], sb.rows, sb.cols, sb.T, sb.Nx)
    Sa, Sj, F = (to_land(x[k], sb.rows, sb.cols, sb.T, sb.Nx) for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat'))
    lam = potential_lambda(Sa, Sj, F, bool(x.get('yearling_breeding')))
    logK = np.log(np.maximum(K, 1e-12))
    _b = bui_z(sb); _h = population_z(sb)
    B = _b[0][None] + _b[1]; H = _h[0][None] + _h[1]   # (time mean, anomaly, coverage) -> full standardised series
    # effort per cell-year from real BBS rows
    d = sb.data; real = np.asarray(d['obs_obsroute_idx']) >= 0 if 'obs_obsroute_idx' in d else np.ones(len(d['observed_results']), bool)
    ocell = sb.lookup[np.asarray(d['obs_rows']), np.asarray(d['obs_cols'])]; ot = np.asarray(d['obs_time_indices'])
    keep = real & (ocell >= 0)
    fy = np.asarray(d['obs_first_year'])[keep] if 'obs_first_year' in d else np.zeros(keep.sum())
    orid = np.asarray(d['obs_obsroute_idx'])[keep] if 'obs_obsroute_idx' in d else np.zeros(keep.sum())
    rid = np.asarray(d['obs_route_idx'])[keep] if 'obs_route_idx' in d else np.zeros(keep.sum())
    oc, otk = ocell[keep], ot[keep]
    N = len(sb.rows); T = sb.T
    nrows = np.zeros((T, N)); nfy = np.zeros((T, N))
    np.add.at(nrows, (otk, oc), 1.); np.add.at(nfy, (otk, oc), fy)
    fy_share = np.where(nrows > 0, nfy / np.maximum(nrows, 1), np.nan)
    eff = np.where(nrows > 0, nrows, np.nan)
    # observer-route turnover per cell over a window: distinct observer-route ids per distinct route
    def turnover(y0, y1):
        m = (yrs[otk] >= y0) & (yrs[otk] <= y1); out = np.full(N, np.nan)
        for c in np.unique(oc[m]):
            s = m & (oc == c); nr = len(np.unique(rid[s]))
            out[c] = len(np.unique(orid[s])) / max(nr, 1)
        return out
    W0, W1 = (1966, 1975), (2010, 2020)
    ch = dict(dlogK=window(logK, yrs, *W1) - window(logK, yrs, *W0),
              dlambda=window(lam, yrs, *W1) - window(lam, yrs, *W0))
    drv = dict(dBUI=window(B, yrs, *W1) - window(B, yrs, *W0), dHYDE=window(H, yrs, *W1) - window(H, yrs, *W0),
               dEffort=np.log(window(eff, yrs, *W1)) - np.log(window(eff, yrs, *W0)),
               dFirstYearShare=window(fy_share, yrs, *W1) - window(fy_share, yrs, *W0),
               dTurnover=turnover(*W1) - turnover(*W0))
    res = dict(windows=[W0, W1], cell_spearman={})
    for reg in ('east', 'plains', 'west'):
        m = sb.region[reg]; row = {}
        for ck, cv in ch.items():
            for dk, dv in drv.items():
                ok = m & np.isfinite(cv) & np.isfinite(dv)
                row[f'{ck}~{dk}'] = (float(spearmanr(cv[ok], dv[ok])[0]), int(ok.sum())) if ok.sum() > 20 else None
        res['cell_spearman'][reg] = row
    # East-minus-West yearly trajectories, linear trend removed
    t = (yrs >= 1966) & (yrs <= 2020); X = np.column_stack([np.ones(t.sum()), yrs[t] - yrs[t].mean()])
    def ew(a):
        s = np.nanmean(a[:, sb.region['east']], 1) - np.nanmean(a[:, sb.region['west']], 1)
        s = s[t]; ok = np.isfinite(s)
        r = np.full_like(s, np.nan); coef, *_ = np.linalg.lstsq(X[ok], s[ok], rcond=None); r[ok] = s[ok] - X[ok] @ coef
        return s, r
    kE, kEr = ew(logK); traj = {}
    for nm, arr in (('BUI', B), ('HYDE', H), ('logEffort', np.log(eff)), ('FirstYearShare', fy_share)):
        s, r = ew(arr); ok = np.isfinite(kE) & np.isfinite(s)
        okr = np.isfinite(kEr) & np.isfinite(r)
        traj[nm] = dict(raw_r=float(np.corrcoef(kE[ok], s[ok])[0, 1]), detrended_r=float(np.corrcoef(kEr[okr], r[okr])[0, 1]))
    res['east_minus_west_logK_vs'] = traj
    a.out.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

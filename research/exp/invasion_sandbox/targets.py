"""A1: observed invasion curves for the sandbox (East and West), from the grid54 inputs only (no House Finch model).

Run through sbx_env.sh STEP (any step: only its config paths are used). Reads metadata.pkl (BBS rows), the early CBC
file (cbc_early.npz: 1947-65 circles) and the grid's reference raster.
T1 release-cell exit (CBC: first positive circle > 54 / > 100 km from the release cell)
T2 radius from the release cell by year (50th / 90th percentile distance of positive CBC circles, then of positive
   BBS routes on a FIXED east panel; M12)
T3 positive routes per year on the fixed east panel (share and count)
T4 local fill-in: mean log1p(count) by years since a route's first positive, by arrival cohort
T5 Hovmoller: share of surveyed routes positive by year x 1-degree longitude, 33-43N, 106W-88W (both fronts)
Usage: python targets.py --out OUT_DIR [--cbc FILE]
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
from src.config_utils import load_age_model_config, load_data_config

CBC_DEFAULT = '/home/breallis/houfin/work/houfin/research/grid54/inputs/cbc_early_20261102/cbc_early.npz'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--cbc', default=CBC_DEFAULT)
    p.add_argument('--panel-min-years', type=int, default=25)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    cfg = load_age_model_config(); dcfg = load_data_config()
    m = pickle.load(open(Path(cfg['input_dir']) / 'metadata.pkl', 'rb'))
    years = np.asarray(m['years']); rows, cols = np.asarray(m['land_rows']), np.asarray(m['land_cols'])
    import rasterio
    from rasterio.warp import transform as crs_transform
    with rasterio.open(dcfg['grid']['ref_raster']) as src:
        xx, yy = map(np.asarray, rasterio.transform.xy(src.transform, rows, cols, offset='center'))
        lon, lat = map(np.asarray, crs_transform(src.crs, 'EPSG:4326', list(xx), list(yy)))
    lookup = -np.ones((int(rows.max()) + 1, int(cols.max()) + 1), int); lookup[rows, cols] = np.arange(len(rows))
    rl = np.asarray(m['inv_locations']).reshape(-1, 2)[0]; rel = int(lookup[rl[0], rl[1]])
    dist = np.hypot(xx - xx[rel], yy - yy[rel]) / 1e3
    east = (lon >= -95) & (lat >= 25) & (lat < 49)

    # ---- BBS rows (real routes only)
    route = np.asarray(m['obs_route_idx']); real = route >= 0
    cell = lookup[np.asarray(m['obs_rows']), np.asarray(m['obs_cols'])]
    yr = years[np.asarray(m['obs_time_indices'])]; cnt = np.asarray(m['observed_results'], float)
    sel = real & (cell >= 0)
    route, cell, yr, cnt = route[sel], cell[sel], yr[sel], cnt[sel]
    # fixed east panel: routes run in >= panel_min_years of 1970-2010
    win = (yr >= 1970) & (yr <= 2010)
    nyr = {r: len(set(yr[win & (route == r)])) for r in np.unique(route[win & east[cell]])}
    panel = np.array(sorted(r for r, n in nyr.items() if n >= a.panel_min_years))
    inp = np.isin(route, panel)
    T3, T2_bbs = {}, {}
    for y in range(1966, 2026):
        s = inp & (yr == y)
        if not s.any(): continue
        rt_pos = {}
        for r, c in zip(route[s], cnt[s]): rt_pos[r] = max(rt_pos.get(r, 0.), c)
        pos = np.array([v > 0 for v in rt_pos.values()])
        T3[y] = dict(surveyed=int(len(pos)), positive=int(pos.sum()), share=float(pos.mean()))
        pc = np.array([dist[cell[s][route[s] == r][0]] for r, v in rt_pos.items() if v > 0])
        if len(pc): T2_bbs[y] = dict(p50=float(np.percentile(pc, 50)), p90=float(np.percentile(pc, 90)), n=int(len(pc)))
    # T4 fill-in (east routes, all, not only panel): years since the route's first positive
    fill = {}
    first_pos = {}
    for r in np.unique(route[east[cell]]):
        s = (route == r)
        yrs_pos = yr[s & (cnt > 0)]
        if len(yrs_pos) and yr[s].min() <= yrs_pos.min() - 3: first_pos[r] = int(yrs_pos.min())
    for r, f in first_pos.items():
        s = route == r; coh = f'{(f - 1966) // 10 * 10 + 1966}s'
        for y, c in zip(yr[s], cnt[s]):
            k = int(y - f)
            if 0 <= k <= 30: fill.setdefault(coh, {}).setdefault(k, []).append(np.log1p(c))
    T4 = {coh: {k: [float(np.mean(v)), len(v)] for k, v in sorted(d.items())} for coh, d in fill.items()}
    # T5 Hovmoller across the Plains
    band = (lat >= 33) & (lat < 43)
    bins = np.arange(-106, -87, 1.)
    hov = np.full((len(range(1966, 2026)), len(bins) - 1), np.nan); hov_n = np.zeros_like(hov)
    for i, y in enumerate(range(1966, 2026)):
        s = (yr == y) & band[cell]
        b = np.digitize(lon[cell[s]], bins) - 1
        for j in range(len(bins) - 1):
            q = b == j
            if q.sum() >= 3:
                rr = {}
                for r, c in zip(route[s][q], cnt[s][q]): rr[r] = max(rr.get(r, 0.), c)
                hov[i, j] = np.mean([v > 0 for v in rr.values()]); hov_n[i, j] = len(rr)

    # ---- early CBC (1947-65)
    with np.load(a.cbc) as z:
        ci, ct, cc = z['cbc_circle'], z['cbc_t'], z['cbc_count']
        src_idx, w = z['cbc_source_idx'], z['cbc_w_local']
    circ_cell = src_idx[np.arange(len(src_idx)), np.argmax(w, 1)]
    cyr = years[ct]; cd = dist[circ_cell[ci]]
    keep = east[circ_cell[ci]]                      # eastern circles only (the file spans the native West too)
    ci, cyr, cd, cc = ci[keep], cyr[keep], cd[keep], cc[keep]
    T2_cbc, T1 = {}, {}
    for y in sorted(set(cyr)):
        s = (cyr == y); pos = s & (cc > 0)
        T2_cbc[int(y)] = dict(circles=int(len(set(ci[s]))), positive_circles=int(len(set(ci[pos]))),
                              birds=float(cc[pos].sum()),
                              p50=float(np.percentile(cd[pos], 50)) if pos.any() else 0.,
                              p90=float(np.percentile(cd[pos], 90)) if pos.any() else 0.,
                              max=float(cd[pos].max()) if pos.any() else 0.)
    for thr in (54., 100., 200.):
        ys = [y for y, d in T2_cbc.items() if d['max'] > thr]
        T1[f'first_beyond_{int(thr)}km'] = min(ys) if ys else None
    out = dict(release_cell=rel, release_lonlat=[float(lon[rel]), float(lat[rel])], panel_routes=int(len(panel)),
               T1=T1, T2_cbc=T2_cbc, T2_bbs=T2_bbs, T3=T3, T4=T4,
               T5=dict(years=list(range(1966, 2026)), lon_edges=bins.tolist(), share=np.where(np.isfinite(hov), hov, -1).tolist(),
                       n_routes=hov_n.tolist()),
               notes='CBC years are model years of cbc_t (count labelled to the summer year used by the fit); BBS first '
                     'positives are detection-biased late (SDM_BASELINE_B6: ~8 yr for late cohorts).')
    (a.out / 'INVASION_TARGETS.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(T1=T1, panel=len(panel),
                          cbc={y: (d['positive_circles'], round(d['p90'])) for y, d in T2_cbc.items() if y % 3 == 0},
                          bbs_p90={y: round(d['p90']) for y, d in T2_bbs.items() if y % 5 == 0},
                          T3_share={y: round(d['share'], 2) for y, d in T3.items() if y % 5 == 0}), indent=0))
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
        ax[0].plot(list(T2_cbc), [d['p90'] for d in T2_cbc.values()], 'o-', label='CBC p90')
        ax[0].plot(list(T2_cbc), [d['max'] for d in T2_cbc.values()], '.:', label='CBC max')
        ax[0].plot(list(T2_bbs), [d['p90'] for d in T2_bbs.values()], 's-', label='BBS panel p90')
        ax[0].plot(list(T2_bbs), [d['p50'] for d in T2_bbs.values()], 's--', label='BBS panel p50')
        ax[0].set(xlabel='Year', ylabel='Distance from release cell (km)', title='T2 radius'); ax[0].legend(fontsize=8)
        ax[1].plot(list(T3), [d['share'] for d in T3.values()], 'k.-')
        ax[1].set(xlabel='Year', ylabel='Share of panel routes positive', title=f'T3 fixed east panel ({len(panel)} routes)')
        im = ax[2].imshow(np.ma.masked_invalid(hov), aspect='auto', origin='lower', vmin=0, vmax=1,
                          extent=[bins[0], bins[-1], 1966, 2026], cmap='viridis')
        ax[2].set(xlabel='Longitude', ylabel='Year', title='T5 share of routes positive, 33-43N'); fig.colorbar(im, ax=ax[2])
        fig.savefig(a.out / 'INVASION_TARGETS.png', dpi=140)
    except Exception as e:
        print('plot failed', e)


if __name__ == '__main__':
    main()

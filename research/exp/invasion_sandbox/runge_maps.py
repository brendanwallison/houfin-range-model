"""Runge et al. (2006) source / sink maps from a contribution audit (run_contribution_audit.py -> annual_budget.npz).

C_r (breeding contribution per breeder) = next summer's breeders descended from a cell's breeders, wherever they settle,
per breeder in that cell: C = R + E, with R = retained breeders per breeder and E = successfully exported breeders per
breeder. C > 1: source (the cell more than replaces itself, counting its emigrants); C < 1: sink. Aggregated over an era
as sum(contribution) / sum(breeders) (breeder-weighted), release-pulse years excluded, on cells holding >= --min_share of
the era's peak breeder density (so empty cells are not classified).
Panels: C per era (4); R (local retention) and E (export) in 2010-25; Runge class vs potential lambda class in 2010-25.
Usage (any python with numpy/matplotlib/rasterio):
  runge_maps.py --audit DIR/contribution_audit --ref RASTER --out DIR/runge
"""
import argparse, json
from pathlib import Path
import numpy as np

ERAS = [(1952, 1965, 'Spread 1952-65'), (1966, 1993, 'BBS pre-disease 1966-93'), (1994, 2009, 'Epizootic 1994-2009'), (2010, 2025, 'Modern 2010-25')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True); p.add_argument('--ref', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--min_share', type=float, default=.01)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import rasterio
    from rasterio.warp import transform
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, ListedColormap
    z = np.load(a.audit / 'annual_budget.npz'); yrs = z['years']; lr, lc = z['land_rows'], z['land_cols']
    with rasterio.open(a.ref) as src:
        shape = (src.height, src.width)
        xx, yy = rasterio.transform.xy(src.transform, lr, lc)
        lon, lat = map(np.asarray, transform(src.crs, 'EPSG:4326', xx, yy))
    regions = {'west': lon < -100, 'plains': (lon >= -100) & (lon < -90), 'east': lon >= -90}

    def era(lo, hi, key, den_key='input_adult'):
        t = (yrs >= lo) & (yrs <= hi) & (z['pulse'].max(1) == 0)
        num = z[key][t].sum(0); den = z[den_key][t].sum(0)
        return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 1e-20), den, t.sum()

    def grid(v):
        g = np.full(shape, np.nan); g[lr, lc] = v; return g
    out = {'definition': 'C = breeder_contribution / input_adult (breeder-weighted over era; pulse years excluded)', 'eras': {}}
    fig, ax = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    land = np.zeros(shape, bool); land[lr, lc] = True
    norm = TwoSlopeNorm(1., .3, 2.)
    for i, (lo, hi, title) in enumerate(ERAS):
        C, den, n = era(lo, hi, 'breeder_contribution')
        occ = den / max(n, 1) >= a.min_share * np.nanmax(den / max(n, 1))
        Cm = np.where(occ, C, np.nan)
        rec = dict(n_occupied=int(occ.sum()), breeder_weighted_C=float(np.nansum(C[occ] * den[occ]) / den[occ].sum()),
                   source_cell_share=float((C[occ] > 1).mean()), source_breeder_share=float(den[occ & (C > 1)].sum() / den[occ].sum()))
        for rn, rm in regions.items():
            o = occ & rm
            if o.any():
                rec[rn] = dict(n=int(o.sum()), C=float(np.nansum(C[o] * den[o]) / den[o].sum()), source_cell_share=float((C[o] > 1).mean()),
                               source_breeder_share=float(den[o & (C > 1)].sum() / den[o].sum()))
        out['eras'][title] = rec
        a_ = ax[0, i]; a_.imshow(np.where(land, 0., np.nan), cmap='Greys', vmin=0, vmax=4)
        im = a_.imshow(grid(Cm), cmap='RdBu', norm=norm); a_.set_xticks([]); a_.set_yticks([])
        a_.set_title(f'{title}: Runge C (breeding)\nsource share: {rec["source_cell_share"]:.2f} of occupied cells, '
                     f'{rec["source_breeder_share"]:.2f} of breeders', fontsize=9)
    fig.colorbar(im, ax=ax[0, :], shrink=.7, label='C (breeders next summer per breeder); >1 source, <1 sink')
    # modern R vs E, and Runge class vs potential-lambda class
    lo, hi = 2010, 2025
    C, den, n = era(lo, hi, 'breeder_contribution'); occ = den / max(n, 1) >= a.min_share * np.nanmax(den / max(n, 1))
    R, _, _ = era(lo, hi, 'retained_breeders'); E, _, _ = era(lo, hi, 'successful_breeder_export')
    t = (yrs >= lo) & (yrs <= hi); lamp = np.nanmean(z['potential_lambda'][t], 0)
    for j, (v, title, kw) in enumerate([(R, 'Modern R: retained breeders / breeder', dict(cmap='RdBu', norm=norm)),
                                        (E, 'Modern E: exported breeders / breeder', dict(cmap='viridis', vmin=0, vmax=.8))]):
        a_ = ax[1, j]; a_.imshow(np.where(land, 0., np.nan), cmap='Greys', vmin=0, vmax=4)
        im2 = a_.imshow(grid(np.where(occ, v, np.nan)), **kw); a_.set_xticks([]); a_.set_yticks([]); a_.set_title(title, fontsize=9)
        fig.colorbar(im2, ax=a_, shrink=.6)
    cls = np.full(C.shape, np.nan)
    cls[occ & (C > 1) & (lamp > 1)] = 0; cls[occ & (C > 1) & (lamp <= 1)] = 1; cls[occ & (C <= 1) & (lamp > 1)] = 2; cls[occ & (C <= 1) & (lamp <= 1)] = 3
    labels = ['Runge source, lambda_pot > 1', 'Runge source, lambda_pot <= 1', 'Runge sink, lambda_pot > 1', 'Runge sink, lambda_pot <= 1']
    cmap = ListedColormap(['#1a9850', '#91cf60', '#fc8d59', '#d73027'])
    a_ = ax[1, 2]; a_.imshow(np.where(land, 0., np.nan), cmap='Greys', vmin=0, vmax=4); a_.imshow(grid(cls), cmap=cmap, vmin=-.5, vmax=3.5)
    a_.set_xticks([]); a_.set_yticks([]); a_.set_title('Modern: Runge class vs potential-lambda class', fontsize=9)
    from matplotlib.patches import Patch
    a_.legend(handles=[Patch(color=cmap(k), label=labels[k]) for k in range(4)], fontsize=7, loc='lower left')
    out['modern_agreement'] = {labels[k]: int((cls == k).sum()) for k in range(4)}
    a_ = ax[1, 3]; ok = occ & np.isfinite(C) & np.isfinite(lamp)
    a_.scatter(lamp[ok], C[ok], s=4, c=np.where(lon[ok] >= -90, '#2166ac', np.where(lon[ok] < -100, '#b2182b', '#4d9221')), alpha=.5)
    a_.axhline(1, color='k', lw=.8); a_.axvline(1, color='k', lw=.8); a_.set_xlabel('potential lambda (low density), 2010-25 mean'); a_.set_ylabel('Runge C, 2010-25')
    a_.set_title('Cells: blue East (>= 90W), green Plains, red West (< 100W)', fontsize=9); a_.set_xlim(0, 3); a_.set_ylim(0, 2.5)
    fig.suptitle(f'Runge et al. 2006 contribution metric: {a.audit.parent.name}', fontsize=12)
    fig.savefig(a.out / 'runge_source_sink.png', dpi=130)
    (a.out / 'runge_source_sink.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

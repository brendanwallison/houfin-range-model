"""E052: Annual NLCD land cover (30 m) -> class fractions of cell area on DESK's 27 km grid.

    python -I research/exp/nlcd_aggregate.py --grid <ref_grid_27km.tif> --out <nlcd_27km.npz>
        --tif 1990=<Annual_NLCD_LndCov_1990_CU_C1V1.tif> --tif 2019=<...2019...tif> [--y0 1985 --y1 2025]

Per snapshot: 30 m class codes counted in 900 m blocks (np.bincount), each block's centre mapped to its 27 km cell
(EPSG:5070 and ESRI:102003 are both CONUS Albers on NAD83; pyproj handles any difference), counts divided by the
cell's area in 30 m pixels (so water and outside-CONUS dilute -- fractions of cell area, like every other channel).
Saved per year y in [y0, y1] = the snapshot nearest to y (ties to the earlier), so epoch means over surveyed years
take each epoch's snapshot. Reads downloaded files only; run with python -I.
"""
import argparse

import numpy as np

CLASSES = [("water", (11, 12)), ("dev_open", (21,)), ("dev_lowmed", (22, 23)), ("dev_high", (24,)), ("barren", (31,)),
           ("deciduous", (41,)), ("evergreen", (42,)), ("mixed_forest", (43,)), ("shrub", (52,)),
           ("grassland", (71,)), ("pasture_hay", (81,)), ("crops", (82,)), ("woody_wetland", (90,)),
           ("emergent_wetland", (95,))]


def aggregate(path, grid, block=30, rows_per_read=1500):
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
    H, W = grid.height, grid.width
    lut = np.full(256, len(CLASSES), np.int32)                       # unknown / nodata -> dropped bucket
    for k, (_, codes) in enumerate(CLASSES):
        lut[list(codes)] = k
    K = len(CLASSES) + 1
    acc = np.zeros((H * W, K), "float64")
    inv = ~grid.transform
    with rasterio.open(path) as s:
        tr = Transformer.from_crs(s.crs, grid.crs, always_xy=True)
        Wt = (s.width // block) * block
        nbx = Wt // block
        px_per_cell = abs(grid.transform.a * grid.transform.e) / abs(s.transform.a * s.transform.e)
        rows_per_read = (rows_per_read // block) * block
        for r0 in range(0, (s.height // block) * block, rows_per_read):
            nr = min(rows_per_read, (s.height // block) * block - r0)
            a = s.read(1, window=Window(0, r0, Wt, nr))
            cls = lut[a]
            nby = nr // block
            bi = ((np.arange(nr, dtype=np.int32) // block)[:, None] * nbx + (np.arange(Wt, dtype=np.int32) // block)[None, :])
            cnt = np.bincount((bi * K + cls).ravel(), minlength=nby * nbx * K).reshape(nby * nbx, K)
            has = cnt[:, :-1].sum(1) > 0
            if not has.any():
                continue
            by, bx = np.divmod(np.nonzero(has)[0], nbx)
            xs, ys = s.transform * ((bx + 0.5) * block, r0 + (by + 0.5) * block)
            gx, gy = tr.transform(xs, ys)
            col, row = inv * (np.asarray(gx), np.asarray(gy))
            col, row = np.floor(col).astype(int), np.floor(row).astype(int)
            ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
            np.add.at(acc, row[ok] * W + col[ok], cnt[has][ok].astype("float64"))
    frac = (acc[:, :-1] / px_per_cell).reshape(H, W, K - 1)
    return np.moveaxis(frac, -1, 0).astype("float32")                 # (K-1, H, W)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid")
    ap.add_argument("--out")
    ap.add_argument("--tif", action="append", help="YEAR=path", required=True)
    ap.add_argument("--y0", type=int, default=1985)
    ap.add_argument("--y1", type=int, default=2025)
    a = ap.parse_args()
    import rasterio
    grid = rasterio.open(a.grid)
    snaps = {}
    for t in a.tif:
        y, p = t.split("=", 1)
        snaps[int(y)] = aggregate(p, grid)
        f = snaps[int(y)]
        print(f"[nlcd] {y}: CONUS means " + ", ".join(f"{n} {float(f[k][f.sum(0) > 0.5].mean()):.3f}"
                                                     for k, (n, _) in enumerate(CLASSES)), flush=True)
    sy = np.array(sorted(snaps))
    years = np.arange(a.y0, a.y1 + 1)
    pick = [sy[np.argmin(np.abs(sy - y) + 1e-6 * (sy > y))] for y in years]
    arr = np.stack([snaps[int(p)] for p in pick])                       # (Y, K, H, W)
    np.savez(a.out, years=years, nlcd=arr, nlcd_names=np.array([n for n, _ in CLASSES]), snapshots=sy,
             snapshot_of_year=np.array(pick))
    print(f"[nlcd] -> {a.out}: snapshots {sy.tolist()}, years {years[0]}-{years[-1]}", flush=True)


if __name__ == "__main__":
    main()

"""Within-cell urbanisation strata for the age model's sub-cell ideal-free settlement.

The age model's dispersal and its covariates (DESK's Z) live on the 27 km grid. Within a cell,
House Finch habitat is far from uniform -- towns versus countryside -- and early colonists
cluster in the towns, so mate-finding and crowding act on LOCAL density, not the cell mean.
``population_model.subcell_strata`` represents each cell as S area slices of its built-up
intensity distribution; the model redistributes each cell's birds among them every year
(fitness ideal-free) and applies the Allee factor and Beverton-Holt per slice.

Source: the existing ``bui_grid`` product (src/data/preprocess/bui.py) -- per-cell QUANTILES of
HISDAC-US built-up intensity from a ~1 km sub-grid, 5-yearly 1885-2020, CONUS only. Quantile
q_k stands for the area slice around it:

    q05 q25 q50 q75 q90 q99  ->  area weights 0.15 0.225 0.25 0.2 0.12 0.055 (sum 1)

(slice boundaries at the midpoints between neighbouring quantile levels, 0 and 1 at the ends).

Output ``subcell_strata.npz`` next to the model inputs:
    u   (time, N_land, S) float32  standardised log1p(BUI) per slice, model years
    a   (S,)                       area weights
    years, u_mean, u_sd, quantiles
u is standardised by the pooled covered-cell mean/sd of log1p(BUI), so the model's tilt
coefficients are per sd. Cells without BUI coverage (bui_avail < 0.5: Canada, Mexico, ocean)
get u = 0 in EVERY slice -- homogeneous, so the mean-preserving tilt is exactly 1 there and
they behave as before. Years before the first snapshot hold it; after the last, hold the last;
in between, linear interpolation in log1p space.

    python -m src.data.preprocess.subcell_strata
"""
import json
import os
import pickle

import numpy as np
import rasterio

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.9, 0.99)


def slice_weights(qs=QUANTILES):
    """Area weight of the slice each quantile stands for (boundaries at midpoints)."""
    q = np.asarray(qs, dtype=float)
    edges = np.concatenate([[0.0], 0.5 * (q[:-1] + q[1:]), [1.0]])
    return np.diff(edges)


def build(bui_dir, input_dir, out_path, avail_min=0.5):
    with open(os.path.join(bui_dir, "manifest.json")) as f:
        manifest = json.load(f)
    qs = tuple(manifest["quantiles"])
    if qs != QUANTILES:
        raise ValueError(f"bui_grid quantiles {qs} != expected {QUANTILES}")
    with open(os.path.join(input_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    rows, cols = np.asarray(meta["land_rows"]), np.asarray(meta["land_cols"])
    years = np.asarray(meta["years"]).astype(int)

    snaps = sorted({int(n.split("_")[2]) for n in os.listdir(bui_dir)
                    if n.startswith("bui_avail_") and n.endswith("_grid.tif")})
    names = [f"bui_q{int(round(q * 100)):02d}" for q in qs]
    vals = np.zeros((len(snaps), rows.size, len(qs)), dtype=np.float64)
    covered = np.zeros((len(snaps), rows.size), dtype=bool)
    for i, y in enumerate(snaps):
        with rasterio.open(os.path.join(bui_dir, f"bui_avail_{y}_grid.tif")) as src:
            covered[i] = src.read(1)[rows, cols] >= avail_min
        for k, n in enumerate(names):
            with rasterio.open(os.path.join(bui_dir, f"{n}_{y}_grid.tif")) as src:
                vals[i, :, k] = src.read(1)[rows, cols]
    lg = np.log1p(np.maximum(vals, 0.0))
    cov = covered.all(axis=0)                 # coverage is a fixed footprint; demand it throughout
    pooled = lg[:, cov, :]
    mu, sd = float(pooled.mean()), float(pooled.std())

    # interpolate snapshots -> model years in log1p space, holding the ends
    out = np.zeros((years.size, rows.size, len(qs)), dtype=np.float32)
    sy = np.asarray(snaps, dtype=float)
    for j, y in enumerate(years):
        if y <= sy[0]:
            v = lg[0]
        elif y >= sy[-1]:
            v = lg[-1]
        else:
            i = int(np.searchsorted(sy, y)) - 1
            f = (y - sy[i]) / (sy[i + 1] - sy[i])
            v = (1 - f) * lg[i] + f * lg[i + 1]
        z = (v - mu) / sd
        z[~cov] = 0.0
        out[j] = z
    np.savez_compressed(out_path, u=out, a=slice_weights(qs), years=years, u_mean=mu, u_sd=sd,
                        quantiles=np.asarray(qs), covered=cov)
    print(f"[subcell_strata] {out_path}: u {out.shape}, covered cells {int(cov.sum())}/{cov.size}, "
          f"log1p(BUI) mean {mu:.3f} sd {sd:.3f}, snapshots {snaps[0]}-{snaps[-1]}")
    return out_path


def main():
    from src.config_utils import load_age_model_config
    bui_dir = os.path.join(os.environ["HOUFIN_DATA"], "bui_grid")
    input_dir = load_age_model_config()["input_dir"]
    build(bui_dir, input_dir, os.path.join(input_dir, "subcell_strata.npz"))


if __name__ == "__main__":
    main()

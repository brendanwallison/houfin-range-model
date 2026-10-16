"""Read a loglik_decompose.py output dir locally: where does B fit better/worse than A?

    python research/age_model/exp/plot_decomp.py <decomp_dir> [--title ...]

Uses <dir>/per_obs_A.npz and <dir>/B/per_obs_A.npz (same observation order). Writes
<dir>/delta_map.png (per-cell sum of dll, zeros and positives separately) and
<dir>/delta_by_year.png (dll by year per longitude region), and prints the top cells.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REGIONS = [("west", -180, -105), ("plains", -105, -95), ("midwest", -95, -85), ("east", -85, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--title", default="")
    ap.add_argument("--meta", default=None, help="meta_subset.npz (land cells, disease onset, release sites)")
    a = ap.parse_args()
    A = np.load(os.path.join(a.dir, "per_obs_A.npz")); B = np.load(os.path.join(a.dir, "B", "per_obs_A.npz"))
    assert np.array_equal(A["y"], B["y"]) and np.array_equal(A["rows"], B["rows"]), "observation order differs"
    d = B["ll"] - A["ll"]; pos = A["y"] > 0; q = A["quality"] == 0
    print(f"total dll {d.sum():+.1f}  (zeros {d[~pos].sum():+.1f}, positives {d[pos].sum():+.1f}; mexico {d[~q].sum():+.1f})")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, mask, lab in ((axes[0], ~pos & q, "zero counts"), (axes[1], pos & q, "positive counts")):
        cells = {}
        for r, c, v in zip(A["rows"][mask], A["cols"][mask], d[mask]):
            cells[(r, c)] = cells.get((r, c), 0.0) + v
        if not cells:
            continue
        rc = np.array(list(cells)); v = np.array(list(cells.values()))
        lim = np.quantile(np.abs(v), 0.98) or 1.0
        lon = {(r, c): x for r, c, x in zip(A["rows"], A["cols"], A["lon"])}
        lat = {(r, c): y for r, c, y in zip(A["rows"], A["cols"], A["lat"])}
        sc = ax.scatter([lon[tuple(k)] for k in rc], [lat[tuple(k)] for k in rc], c=v, s=14,
                        cmap="RdBu", vmin=-lim, vmax=lim, marker="s")
        ax.set_title(f"{lab}: sum dll per cell (blue = B fits better)"); plt.colorbar(sc, ax=ax)
        worst = np.argsort(v)[:5]
        print(f"  {lab}: worst cells (lon, lat, dll):",
              [(round(lon[tuple(rc[i])], 1), round(lat[tuple(rc[i])], 1), round(v[i], 1)) for i in worst])
    fig.suptitle(a.title); fig.tight_layout(); fig.savefig(os.path.join(a.dir, "delta_map.png"), dpi=110)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    yrs = np.unique(A["year"])
    for name, lo, hi in REGIONS:
        for mask, ls in ((~pos, ":"), (pos, "-")):
            m = q & mask & (A["lon"] >= lo) & (A["lon"] < hi)
            ax.plot(yrs, [d[m & (A["year"] == y)].sum() for y in yrs], ls, label=f"{name} {'zeros' if ls == ':' else 'pos'}")
    ax.axhline(0, color="k", lw=0.5); ax.set(xlabel="year", ylabel="dll (B - A)", title=a.title)
    ax.legend(ncol=4, fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(a.dir, "delta_by_year.png"), dpi=110)
    if a.meta:
        plot_clocks(A, d, pos, q, a.meta, a.dir, a.title)


def clocks(A, meta_path):
    """Per-observation clocks matching the hypotheses: years since the cell's first observed
    positive count (H2 fill-in), years relative to local disease arrival (H6), and distance from
    the nearest release cell in km (H5). NaN where undefined."""
    m = np.load(meta_path)
    rows, cols, year, y = A["rows"], A["cols"], A["year"], A["y"]
    first = {}
    for r, c, yr, v in zip(rows, cols, year, y):
        if v > 0 and yr < first.get((r, c), 1e9):
            first[(r, c)] = yr
    since_first = np.array([yr - first.get((r, c), np.nan) for r, c, yr in zip(rows, cols, year)], float)
    lookup = np.full((int(m["Ny"]), int(m["Nx"])), -1)
    lookup[m["land_rows"], m["land_cols"]] = np.arange(m["land_rows"].size)
    cell = lookup[rows, cols]
    onset_year = np.where(cell >= 0, m["years"][0] + m["disease_onset"][np.maximum(cell, 0)], np.nan)
    rel_disease = year - onset_year
    inv = m["inv_locations"]
    dist = 27.0 * np.min(np.hypot(rows[:, None] - inv[None, :, 0], cols[:, None] - inv[None, :, 1]), axis=1)
    return {"years since first positive in cell": (since_first, np.arange(-10, 45, 5)),
            "years since local disease arrival": (rel_disease, np.arange(-20, 35, 5)),
            "km from release site": (dist, np.array([0, 100, 250, 500, 1000, 1500, 2500, 5000]))}


def plot_clocks(A, d, pos, q, meta_path, out, title):
    cl = clocks(A, meta_path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (name, (x, edges)) in zip(axes, cl.items()):
        mids = 0.5 * (edges[:-1] + edges[1:])
        for mask, lab, ls in ((~pos & q, "zeros", ":"), (pos & q, "positives", "-")):
            vals = [d[mask & (x >= lo) & (x < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])]
            ax.plot(mids, vals, ls, marker="o", label=lab)
        ax.axhline(0, color="k", lw=0.5); ax.set(xlabel=name, ylabel="dll (B - A)"); ax.legend(fontsize=8)
        print(f"  by {name}: " + ", ".join(f"[{lo:g},{hi:g}): {d[q & (x >= lo) & (x < hi)].sum():+.1f}"
                                          for lo, hi in zip(edges[:-1], edges[1:])))
    fig.suptitle(title); fig.tight_layout(); fig.savefig(os.path.join(out, "delta_by_clock.png"), dpi=110)


if __name__ == "__main__":
    main()

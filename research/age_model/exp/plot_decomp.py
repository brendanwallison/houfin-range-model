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


if __name__ == "__main__":
    main()

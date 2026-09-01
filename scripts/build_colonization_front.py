"""First year a House Finch was DETECTED in each cell — the observed colonization front.

WHY. Every validation map so far can say the model is wrong somewhere; none can say the error sits
at the expansion front, which is the one place a range model is actually being asked to work. The
front is the obvious spatial covariate for reading an error map and it has never existed as a
product: `plot_invasion_progression` maps the SIMULATED front, and `visualize_bbs_observed_spread`
animates the observed one and persists only a GIF.

TWO TRAPS, both of which make a plausible-looking and wrong raster.

1. THE PSEUDO-ZERO BLOCK. `bbs_data_for_python.npz` concatenates synthetic 1902-1939 zeros AHEAD
   of the real observations (`bbs.py` `save`), so a bare `min(obs_year)` per cell returns 1902
   everywhere. The npz records `N_pseudo`, so the block is sliced off by provenance rather than
   inferred from the counts -- and then `observed_results > 0` additionally drops real surveyed
   absences, because the question is first DETECTION, not first survey.

2. THE WEST HAS NO FRONT. Pseudo-zeros are only asserted outside a 700 km halo around the native
   hull, and the native population was already there in 1902. A western cell's first detection is
   therefore 1966 -- BBS's launch year, an artifact of when counting started, not a colonization
   date. Those cells are MASKED, not emitted: a front raster that quietly reports 1966 across the
   native range would put a bright artificial edge exactly where the Great Plains analysis looks.

The output is deliberately narrow: eastern expansion only, with the native hull and never-detected
cells as nodata, and the mask recorded in the raster tags so a consumer cannot mistake its scope.

    python scripts/build_colonization_front.py [--out PATH] [--dilate-hull 2]
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.config_utils import load_age_model_config, load_data_config  # noqa: E402

NODATA = -9999.0


def first_detection(obs_rows, obs_cols, obs_year, counts, shape, n_pseudo=0):
    """``(H, W)`` first year with a positive count per cell; NaN where never detected. Pure.

    ``n_pseudo`` slices the synthetic pre-invasion zeros off the front of the arrays. It is a
    count, not a year cutoff, because the pseudo block's own years (1902-1939) overlap nothing
    real and a year-based filter would silently also drop any genuine early record if the record
    ever gained one.
    """
    r = np.asarray(obs_rows)[n_pseudo:]
    c = np.asarray(obs_cols)[n_pseudo:]
    y = np.asarray(obs_year)[n_pseudo:]
    v = np.asarray(counts)[n_pseudo:]
    seen = v > 0
    out = np.full(shape, np.nan)
    if not seen.any():
        return out
    lin = r[seen] * shape[1] + c[seen]
    order = np.argsort(y[seen], kind="stable")            # earliest first, then take the first
    lin, yy = lin[order], y[seen][order]
    flat = np.full(shape[0] * shape[1], np.nan)
    _, first = np.unique(lin, return_index=True)
    flat[lin[first]] = yy[first]
    return flat.reshape(shape)


def native_hull_mask(initpop_rows, initpop_cols, shape, dilate=0):
    """Cells the native western population already occupied, optionally dilated.

    Dilation exists because the hull is a CONVEX hull of pre-1970 presences and so is a lower
    bound on the real native fringe; a cell one step outside it is not a colonization event
    either. Chebyshev, matching `blocked_holdout`'s buffer so the two dilations are one idea.
    """
    m = np.zeros(shape, bool)
    m[np.asarray(initpop_rows, int), np.asarray(initpop_cols, int)] = True
    for _ in range(int(dilate)):
        p = np.pad(m, 1, constant_values=False)
        m = np.zeros_like(m)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                m |= p[1 + dy:1 + dy + shape[0], 1 + dx:1 + dx + shape[1]]
    return m


def build(npz_path, dilate=2):
    z = np.load(npz_path, allow_pickle=True)
    shape = (int(z["Ny"]), int(z["Nx"]))
    n_pseudo = int(z["N_pseudo"]) if "N_pseudo" in z else 0
    front = first_detection(z["obs_rows"], z["obs_cols"], z["obs_year"],
                            z["observed_results"], shape, n_pseudo=n_pseudo)
    hull = native_hull_mask(z["initpop_rows"], z["initpop_cols"], shape, dilate=dilate)
    masked = np.where(hull, np.nan, front)
    bbs_start = int(np.asarray(z["obs_year"])[n_pseudo:].min()) if n_pseudo < len(z["obs_year"]) \
        else None
    stats = {"n_pseudo_dropped": n_pseudo,
             "n_cells_detected": int(np.isfinite(front).sum()),
             "n_native_hull_masked": int((hull & np.isfinite(front)).sum()),
             "n_emitted": int(np.isfinite(masked).sum()),
             "bbs_first_year": bbs_start,
             "year_range": ([float(np.nanmin(masked)), float(np.nanmax(masked))]
                            if np.isfinite(masked).any() else None)}
    # The launch-year artifact, measured rather than assumed: if a large share of what survives
    # the hull mask is still exactly the first BBS year, the mask is too tight and those cells are
    # "already there when counting started", not colonized in 1966.
    if bbs_start is not None and np.isfinite(masked).any():
        at_start = int(np.nansum(masked == bbs_start))
        stats["n_at_bbs_first_year"] = at_start
        stats["share_at_bbs_first_year"] = at_start / max(stats["n_emitted"], 1)
    return masked, hull, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=None, help="bbs_data_for_python.npz (default: config)")
    ap.add_argument("--out", default=None, help="output GeoTIFF (default: processed/regions)")
    ap.add_argument("--dilate-hull", type=int, default=2,
                    help="Chebyshev dilation of the native hull, in cells (default 2 = 54 km)")
    args = ap.parse_args()

    npz = args.npz or load_age_model_config()["bbs_npz"]
    if not os.path.exists(npz):
        raise SystemExit(f"{npz} not found — this is built on HPC; run there or scp it down")
    dcfg = load_data_config()
    out = args.out or os.path.join(dcfg["processed_root"], "regions",
                                   "colonization_front_27km.tif")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    front, hull, stats = build(npz, dilate=args.dilate_hull)
    print(json.dumps(stats, indent=2))
    if stats.get("share_at_bbs_first_year", 0) > 0.25:
        print(f"[front] WARNING {100 * stats['share_at_bbs_first_year']:.0f}% of emitted cells "
              f"are exactly {stats['bbs_first_year']}, the first BBS year. Those are 'already "
              f"present when counting started', not colonizations — widen --dilate-hull.")

    import rasterio
    with rasterio.open(dcfg["grid"]["ref_raster"]) as ref:
        profile = ref.profile
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(front), front, NODATA).astype("float32"), 1)
        dst.update_tags(
            method="first BBS year with a positive House Finch count",
            scope="EASTERN EXPANSION ONLY — native western hull is nodata",
            native_hull_dilate_cells=str(args.dilate_hull),
            pseudo_zeros_dropped=str(stats["n_pseudo_dropped"]),
            **{k: str(v) for k, v in stats.items()})
    print(f"[front] wrote {out}")


if __name__ == "__main__":
    main()

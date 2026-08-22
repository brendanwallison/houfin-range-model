"""Is the early-era sample a spatially biased subset, and does the epoch gate make it worse?

A cell's FIRST SURVEY YEAR is not random -- it is structured in space, and BBS is biased toward
the COASTS. Nothing in the encoder conditions on it (`first_year_weight` is the OBSERVER's first
year on a route, a different effect). Two consequences worth measuring rather than assuming:

1. Any "early era" statistic is computed on whichever cells existed early, so an early-vs-modern
   difference confounds era with geography. Note this concerns the REFERENCE COMMUNITY -- DESK
   encodes 96 species with the focal House Finch EXCLUDED -- so what is at stake is which
   communities the basis and the validation represent, not anything about the focal species'
   range.

   The axis must be DISTANCE TO COAST. An east-west split cannot see a coastal bias: it puts the
   Pacific coast in the same bin as the interior mountain west, so a U-shape reads as a trend.

2. The epoch gate requires >=3 distinct surveyed years in BOTH eras. That is a data-quality
   filter in form and a SPATIAL filter in effect, and it compounds (1).

Also reports the ENDPOINT AVERAGING asymmetry: the modern reference averages all rows in a 16-year
window (~11.4 surveys) while the early endpoint uses +/-2 years (~4.2). For a DIFFERENCE, unequal
endpoint noise biases the result per era -- which is the very axis the temporal sweep varies.

Run: cd $HOUFIN_REPO && python scripts/diagnostics/coverage_asymmetry.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.community_encoder.train_DESK.config_utils import load_config          # noqa: E402

EARLY_GATE = (1966, 1986)
MODERN_GATE = (2005, 2025)


def main():
    cfg = load_config(os.environ.get("ESK_DESK_CONFIG") or None)
    from src.community_encoder.train_DESK.validate_bbs_routes import load_observed
    X_log, keys, meta, X_raw = load_observed(cfg)
    rows, cols, yrs = keys[:, 0].astype(int), keys[:, 1].astype(int), keys[:, 2].astype(int)
    cell = rows * 100000 + cols

    first, nyr, by_cell = {}, {}, {}
    for i in range(len(keys)):
        c = int(cell[i])
        first[c] = min(first.get(c, 9999), int(yrs[i]))
        by_cell.setdefault(c, []).append(int(yrs[i]))
    cids = np.array(sorted(by_cell))
    fy = np.array([first[c] for c in cids])
    ccol = np.array([c % 100000 for c in cids])          # grid column ~ easting
    crow = np.array([c // 100000 for c in cids])

    print(f"{len(cids):,} cells, first-survey year {fy.min()}-{fy.max()}")

    # DISTANCE TO COAST, not easting. BBS is coast-biased, and a monotonic east-west split cannot
    # see that -- it lumps the Pacific coast in with the interior mountain west, so a real U-shape
    # in coverage reads as a spurious trend. The axis has to be distance from the coastline.
    from scipy import ndimage
    from src.data.preprocess import bbs
    land_mask, _, transform, crs, nx, ny = bbs.load_grid_reference(bbs.MASK_PATH)
    land = np.asarray(land_mask, bool)
    # Distance (in grid cells) from each land cell to the nearest non-land cell = the coastline.
    dist_coast = ndimage.distance_transform_edt(land)
    dc = np.array([dist_coast[r, c] if (0 <= r < land.shape[0] and 0 <= c < land.shape[1]) else
                   np.nan for r, c in zip(crow, ccol)])
    good = np.isfinite(dc)
    print(f"\n=== coverage vs DISTANCE TO COAST (grid cells; 1 cell ~ 27 km) ===")
    print(f"  Pearson r(first_year, dist_to_coast) = "
          f"{np.corrcoef(fy[good], dc[good])[0, 1]:+.3f}   "
          f"(positive => interior cells entered the survey LATER)")
    edges = np.nanquantile(dc[good], [0, .25, .5, .75, 1.0])
    print(f"  {'coast->interior':<20}{'cells':>7}{'med dist':>10}{'median first yr':>17}"
          f"{'% by 1970':>11}")
    coast_bins = []
    for i in range(4):
        m = good & (dc >= edges[i]) & ((dc <= edges[i + 1]) if i == 3 else (dc < edges[i + 1]))
        coast_bins.append(m)
        if m.sum():
            print(f"  Q{i + 1}{'':<17}{int(m.sum()):>7}{np.median(dc[m]):>10.1f}"
                  f"{np.median(fy[m]):>17.0f}{100 * np.mean(fy[m] <= 1970):>10.1f}%")

    # --- the epoch gate, and who it keeps -------------------------------------------------
    def n_distinct(c, lo, hi):
        return len({y for y in by_cell[c] if lo <= y <= hi})

    early_ok = np.array([n_distinct(c, *EARLY_GATE) >= 3 for c in cids])
    modern_ok = np.array([n_distinct(c, *MODERN_GATE) >= 3 for c in cids])
    both = early_ok & modern_ok
    print(f"\n=== the epoch gate (>=3 distinct years in BOTH {EARLY_GATE} and {MODERN_GATE}) ===")
    print(f"  passes early gate : {early_ok.sum():>6,} / {len(cids):,}")
    print(f"  passes modern gate: {modern_ok.sum():>6,} / {len(cids):,}")
    print(f"  passes BOTH       : {both.sum():>6,} / {len(cids):,}  "
          f"({100 * both.mean():.1f}% of cells)")
    print(f"\n  {'coast->interior':<20}{'all cells':>10}{'pass both':>11}{'pass rate':>11}")
    for i, m in enumerate(coast_bins):
        if m.sum():
            print(f"  Q{i + 1}{'':<17}{int(m.sum()):>10}{int((m & both).sum()):>11}"
                  f"{100 * (m & both).sum() / m.sum():>10.1f}%")
    if both.sum() and good.sum():
        print(f"\n  mean distance to coast: all cells {np.nanmean(dc[good]):.2f} vs gated "
              f"{np.nanmean(dc[good & both]):.2f} grid cells "
              f"(shift {np.nanmean(dc[good & both]) - np.nanmean(dc[good]):+.2f} "
              f"~ {27 * (np.nanmean(dc[good & both]) - np.nanmean(dc[good])):+.0f} km)")
        print("  A NEGATIVE shift means the gate keeps coastal cells preferentially and thins the")
        print("  interior -- the known BBS bias, now quantified on the axis that can see it.")

    # --- endpoint averaging asymmetry -----------------------------------------------------
    print(f"\n=== endpoint averaging, per gated cell ===")
    e_n, m_n = [], []
    for c in cids[both] if both.sum() else []:
        ys = by_cell[c]
        # the early endpoint is a +/-2yr window around a row's own year; take the densest
        e_best = max((sum(1 for y in ys if abs(y - y0) <= 2)
                      for y0 in ys if EARLY_GATE[0] <= y0 <= EARLY_GATE[1]), default=0)
        e_n.append(e_best)
        m_n.append(sum(1 for y in ys if MODERN_GATE[0] <= y <= MODERN_GATE[1]))
    if e_n:
        e_n, m_n = np.array(e_n), np.array(m_n)
        print(f"  early  (+/-2 yr window) : median {np.median(e_n):.1f} surveys")
        print(f"  modern (16 yr window)   : median {np.median(m_n):.1f} surveys")
        print(f"  ratio                   : {np.median(m_n) / max(np.median(e_n), 1e-9):.2f}x more "
              "averaging on the modern side")
        print("  -> noise variance scales ~1/n, so the EARLY endpoint carries roughly that factor")
        print("     more noise. In a difference the noisier endpoint dominates, and the imbalance")
        print("     is itself era-dependent -- confounded with the temporal sweep's own axis.")


if __name__ == "__main__":
    main()

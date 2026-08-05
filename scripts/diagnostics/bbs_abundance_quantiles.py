#!/usr/bin/env python
"""Recompute the BBS abundance anchors that two priors depend on.

WHY THIS EXISTS. ``capacity_level_prior.target_level_mean_route_counts`` (2.892) and
``invasion_pulse_prior.global_q50_route_counts`` (0.64) both trace to an occupied-cell and an
all-cell median from an earlier quantile pass over the BBS data. That pass is not in the repo --
the numbers appear only in config comments, and the config itself says to "RECONFIRM 0.64 by
re-running ... against the current dataset/grid". So the definitions behind two live priors were
unrecoverable. This script is that pass, written down.

DEFINITIONS, stated explicitly because the previous ones were lost:

* MODERN ERA -- route-years with ``year >= --modern-start`` (default 1990). Every quantity below
  is computed from those records only, not the whole 1966-2025 history.
* PER-CELL VALUE -- the mean of ``observed_results`` (SpeciesTotal, a 50-stop route count) over
  that grid cell's modern route-year records. A cell surveyed by three routes in each of ten
  years contributes 30 records to its own mean.
* OCCUPIED -- a cell with at least ``--min-positive-years`` (default 2) DISTINCT modern years in
  which a positive count was recorded. Distinct years, not records: two routes both reporting
  birds in 1994 is one year of evidence, and one positive record ever is not occupancy.

Three populations are reported, because "all cells" is ambiguous and the choice moves the median:

1. ``occupied``       -- cells meeting the occupancy rule above.
2. ``surveyed``       -- every cell with at least one modern record, including cells whose counts
                         are all zero. These are genuine observed absences.
3. ``all_land``       -- every land cell, with never-surveyed cells counted as 0. This conflates
                         "surveyed and empty" with "never looked", so it is the lowest of the
                         three and the one to treat with most suspicion.

The 2.892 in config is not a median: ``_solve_softplus_loc`` matches a prior MEAN, and 2.892 is
the occupied median mapped through the lognormal mean/median ratio at ``alpha_k_scale`` = 0.8,
i.e. ``median * exp(0.8**2 / 2)``. That conversion is printed so the comparison is like-for-like.

    python scripts/diagnostics/bbs_abundance_quantiles.py
    python scripts/diagnostics/bbs_abundance_quantiles.py --modern-start 2000 --min-positive-years 3
    python scripts/diagnostics/bbs_abundance_quantiles.py --self-test   # no data needed
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

QUANTILES = [5, 10, 25, 50, 75, 90, 95, 99]


def cell_stats(rows, cols, years, counts, nx, modern_start, min_positive_years):
    """Per-cell modern-era summaries (pure).

    Returns a dict of flat arrays keyed by cell id (``row * nx + col``) order:
    ``cell_id``, ``mean_count``, ``n_records``, ``n_years``, ``n_positive_years``, ``occupied``.
    """
    rows = np.asarray(rows, dtype="int64")
    cols = np.asarray(cols, dtype="int64")
    years = np.asarray(years, dtype="int64")
    counts = np.asarray(counts, dtype="float64")

    keep = years >= int(modern_start)
    rows, cols, years, counts = rows[keep], cols[keep], years[keep], counts[keep]
    if rows.size == 0:
        raise ValueError(f"no BBS records at or after {modern_start}")

    cid = rows * int(nx) + cols
    uniq, inv = np.unique(cid, return_inverse=True)

    n_records = np.bincount(inv, minlength=uniq.size)
    total = np.bincount(inv, weights=counts, minlength=uniq.size)
    mean_count = total / np.maximum(n_records, 1)

    # Distinct years per cell, and distinct years with a POSITIVE count. Done on the
    # (cell, year) pair so repeated routes in one year collapse to a single year.
    pair = np.stack([inv, years], 1)
    uniq_pair = np.unique(pair, axis=0)
    n_years = np.bincount(uniq_pair[:, 0], minlength=uniq.size)
    pos = counts > 0
    if pos.any():
        uniq_pos = np.unique(np.stack([inv[pos], years[pos]], 1), axis=0)
        n_pos_years = np.bincount(uniq_pos[:, 0], minlength=uniq.size)
    else:
        n_pos_years = np.zeros(uniq.size, dtype="int64")

    return {"cell_id": uniq, "mean_count": mean_count, "n_records": n_records,
            "n_years": n_years, "n_positive_years": n_pos_years,
            "occupied": n_pos_years >= int(min_positive_years)}


def _report(label, values, note=""):
    values = np.asarray(values, dtype="float64")
    if values.size == 0:
        print(f"  {label:<12} (empty)")
        return {}
    qs = np.percentile(values, QUANTILES)
    out = {f"q{q}": float(v) for q, v in zip(QUANTILES, qs)}
    out["n"] = int(values.size)
    out["mean"] = float(values.mean())
    print(f"  {label:<12} n={values.size:<7} mean={values.mean():8.4f}  "
          + "  ".join(f"q{q}={v:.4f}" for q, v in zip(QUANTILES, qs)) + (f"   {note}" if note else ""))
    pos = values[values > 0]
    if pos.size > 1:
        out["log_sd_positive"] = float(np.std(np.log(pos)))
        out["geometric_mean_positive"] = float(np.exp(np.mean(np.log(pos))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modern-start", type=int, default=1990)
    ap.add_argument("--min-positive-years", type=int, default=2)
    ap.add_argument("--quality", type=int, default=0,
                    help="max obs_quality tier to keep (0 = QC'd US/Canada only; "
                         "1 also admits the unscreened Mexico tier). -1 keeps all.")
    ap.add_argument("--npz", default=None, help="override bbs_npz path")
    ap.add_argument("--self-test", action="store_true",
                    help="run cell_stats on synthetic data and exit (no BBS data needed)")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    from src.config_utils import load_age_model_config
    cfg = load_age_model_config()
    path = args.npz or cfg["bbs_npz"]
    if not os.path.exists(path):
        raise SystemExit(f"bbs_npz not found: {path}\n"
                         "This must run where the BBS npz lives (TACC $HOUFIN_DATA).")
    d = np.load(path, allow_pickle=True)

    rows, cols = d["obs_rows"], d["obs_cols"]
    years, counts = d["obs_year"], d["observed_results"]
    nx, ny = int(d["Nx"]), int(d["Ny"])
    land = np.asarray(d["land"]).astype(bool)

    # The npz concatenates PSEUDO-ZEROS (pre-invasion uninvaded-east zeros) ahead of the real
    # observations. They are all pre-1940, so a modern filter excludes them -- but assert it
    # rather than assume, because silently averaging fabricated zeros into an abundance anchor
    # is exactly the kind of error this script exists to prevent.
    n_pseudo = int(d["N_pseudo"]) if "N_pseudo" in d.files else 0
    if n_pseudo:
        pseudo_modern = int((years[:n_pseudo] >= args.modern_start).sum())
        if pseudo_modern:
            raise SystemExit(f"{pseudo_modern} pseudo-zero records fall at/after "
                             f"{args.modern_start}; they would corrupt the anchor")
        print(f"[bbs-quantiles] {n_pseudo} pseudo-zero records, all pre-{args.modern_start}: excluded")

    if args.quality >= 0 and "obs_quality" in d.files:
        q = np.asarray(d["obs_quality"])
        keep = q <= args.quality
        print(f"[bbs-quantiles] quality filter <= {args.quality}: keeping {int(keep.sum())}"
              f"/{keep.size} records (tiers present: {sorted(set(q.tolist()))})")
        rows, cols, years, counts = rows[keep], cols[keep], years[keep], counts[keep]

    st = cell_stats(rows, cols, years, counts, nx, args.modern_start, args.min_positive_years)
    n_land = int(land.sum())

    print(f"\nMODERN ERA >= {args.modern_start};  OCCUPIED = >= {args.min_positive_years} "
          f"distinct modern years with a positive count")
    print(f"per-cell value = mean route count over that cell's modern records\n")
    print(f"  land cells {n_land};  surveyed in era {st['cell_id'].size};  "
          f"occupied {int(st['occupied'].sum())}")
    print(f"  median distinct modern years per surveyed cell: {np.median(st['n_years']):.0f}\n")

    res = {}
    res["occupied"] = _report("occupied", st["mean_count"][st["occupied"]])
    res["surveyed"] = _report("surveyed", st["mean_count"], "(incl. observed-zero cells)")
    all_land = np.zeros(n_land, dtype="float64")
    all_land[:st["cell_id"].size] = st["mean_count"]      # unsurveyed land cells contribute 0
    res["all_land"] = _report("all_land", all_land, "(never-surveyed counted as 0)")

    occ_med = res["occupied"].get("q50")
    print(f"\nAgainst the two live priors:")
    if occ_med:
        scale = 0.8                                       # capacity_level_prior.alpha_k_scale
        print(f"  capacity_level_prior.target_level_mean_route_counts = 2.892 in config")
        print(f"    occupied median {occ_med:.4f} -> lognormal mean at alpha_k_scale={scale}: "
              f"{occ_med * math.exp(scale ** 2 / 2):.4f}")
        lsd = res["occupied"].get("log_sd_positive")
        if lsd:
            print(f"    occupied log sd {lsd:.4f}  (config comment cites 1.72)")
    print(f"  invasion_pulse_prior.global_q50_route_counts = 0.64 in config")
    print(f"    surveyed median {res['surveyed'].get('q50', float('nan')):.4f}   "
          f"all-land median {res['all_land'].get('q50', float('nan')):.4f}")
    print("\n  Pick the population deliberately: 'all cells' was ambiguous in the original and "
          "the three medians above differ.")


def _self_test():
    """Verify cell_stats' occupancy and mean logic on hand-built records."""
    nx = 10
    # cell (0,0): positive in 1995 and 2001 -> 2 positive years -> OCCUPIED
    # cell (0,1): two positive records in ONE year -> 1 positive year -> NOT occupied
    # cell (0,2): surveyed twice, both zero -> surveyed, not occupied
    # cell (0,3): positive only in 1980 -> outside the modern era entirely -> not surveyed
    rows = [0, 0, 0, 0, 0, 0, 0]
    cols = [0, 0, 1, 1, 2, 2, 3]
    years = [1995, 2001, 1994, 1994, 1996, 1999, 1980]
    counts = [4, 8, 5, 7, 0, 0, 9]
    st = cell_stats(rows, cols, years, counts, nx, modern_start=1990, min_positive_years=2)
    ids = st["cell_id"].tolist()
    assert ids == [0, 1, 2], ids                          # (0,3) excluded: pre-1990
    assert st["occupied"].tolist() == [True, False, False], st["occupied"]
    assert st["n_positive_years"].tolist() == [2, 1, 0]
    assert st["n_records"].tolist() == [2, 2, 2]
    assert np.allclose(st["mean_count"], [6.0, 6.0, 0.0])
    # (0,1) has the same mean as (0,0) but fails occupancy -- the rule is about YEARS of
    # evidence, not abundance, which is the distinction the old "SpeciesTotal > 0" lacked.
    print("self-test OK: modern filter, distinct-positive-year occupancy, per-cell means")


if __name__ == "__main__":
    main()

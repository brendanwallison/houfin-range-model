"""Occupancy of the weighting strata, swept over binning resolution. CPU only, seconds to run.

WHY THIS EXISTS. The rebalance in `desk.balance` needs two numbers chosen from the data: how
finely to bin, and where to put the `n_min` floor. The first attempt guessed `n_min=200` while the
median stratum held 41 rows and the largest held ~570 -- so the floor sat above almost every
stratum, all of them received the floor weight, and the realised weight range collapsed to
0.60-1.00: a mild downweight of the few dense strata with no uplift anywhere. The correction was
inert and the run looked like it had worked.

The table and the weights were computed in the same execution, so there was no opportunity to set
one from the other. This separates them. Run it, read it, then set `spatial_bins` and either
`n_min` or `n_min_quantile` in `desk.balance`.

WHAT TO LOOK FOR, in order:

* `median` per stratum -- a weighting scheme cannot be supported by cells of a few dozen rows. The
  thin tail of an over-binned pool is mostly fragmentation, and upweighting it chases noise rather
  than correcting bias. Prefer the coarsest resolution that still separates the regions you care
  about.
* `max/median` -- the actual severity of the coast/present bias at that resolution. Measured at
  8x8 tiles WITH an abundance axis it was only 14-15x, well below what "BBS is coast-heavy" had
  been taken to imply, and much of even that was binning artifact.
* `n_singleton` -- strata holding one observation. These are the ones a naive inverse-frequency
  weight would boost hardest, and they cannot support an estimate at all.

    python scripts/diagnostics/stratum_occupancy.py
    python scripts/diagnostics/stratum_occupancy.py --exclude-years 1966-1995
"""
import argparse
import os
import sys

import numpy as np

# Same line as scripts/diagnostics/bbs_abundance_quantiles.py: run by path rather than as a module,
# so the repo root is not on sys.path and `import src` fails without it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.community_encoder.train_DESK.esk_kernel import spacetime_strata  # noqa: E402
from src.config_utils import load_config, target_points_dir                # noqa: E402


def _summary(labels, pidx):
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    return {"n_strata": len(counts), "median": float(np.median(counts)),
            "p10": float(np.quantile(counts, 0.10)),
            "p25": float(np.quantile(counts, 0.25)),
            "max": int(counts.max()),
            "ratio": float(counts.max() / max(np.median(counts), 1)),
            "n_singleton": int((counts == 1).sum()),
            "frac_rows_in_thin": float(counts[counts < 20].sum() / counts.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclude-years", default="",
                    help="e.g. 1966-1995, to mirror a temporal holdout's pool")
    ap.add_argument("--tiles", default="2,3,4,6,8",
                    help="spatial bins per axis to sweep")
    args = ap.parse_args()

    cfg = load_config()
    zt = target_points_dir(cfg)
    pidx = np.load(os.path.join(zt, "point_index.npy"))
    X = np.nan_to_num(np.load(os.path.join(zt, "X_points.npy"))).astype("float32")

    if args.exclude_years:
        lo, hi = (int(v) for v in args.exclude_years.split("-"))
        keep = ~((pidx[:, 2] >= lo) & (pidx[:, 2] <= hi))
        print(f"excluding {lo}-{hi}: {int((~keep).sum()):,} of {len(pidx):,} rows dropped, "
              f"mirroring that holdout's training pool")
        pidx, X = pidx[keep], X[keep]

    print(f"\n{len(pidx):,} pool rows, years {pidx[:, 2].min()}-{pidx[:, 2].max()}\n")
    hdr = (f"{'tiles':>6} {'abund':>6} {'strata':>7} {'median':>7} {'p25':>6} {'p10':>6} "
           f"{'max':>7} {'max/med':>8} {'single':>7} {'%rows<20':>9}")
    print(hdr); print("-" * len(hdr))
    for t in (int(v) for v in args.tiles.split(",")):
        for inc_ab, ab in ((False, 1), (True, 4)):
            lab, _k = spacetime_strata(pidx, X, spatial_bins=t, abundance_bins=ab,
                                       include_abundance=inc_ab)
            s = _summary(lab, pidx)
            print(f"{t:>4}x{t:<2} {(ab if inc_ab else 0):>6} {s['n_strata']:>7} "
                  f"{s['median']:>7.0f} {s['p25']:>6.0f} {s['p10']:>6.0f} {s['max']:>7} "
                  f"{s['ratio']:>7.0f}x {s['n_singleton']:>7} {100*s['frac_rows_in_thin']:>8.1f}%")
    print("\nabund=0 is the WEIGHTING stratification (geography x time only); abund=4 is what the")
    print("ESK landmark coverage uses, where spanning the community manifold is the point.")
    print("Pick the coarsest tiling whose median stratum is large enough to weight on, then set")
    print("desk.balance.spatial_bins, and n_min (or n_min_quantile) from that row.")


if __name__ == "__main__":
    main()

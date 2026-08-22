"""Is the early-era sample a spatially biased subset, and does the epoch gate make it worse?

BBS began in 1966 in the east and expanded west and into Canada over following decades, so a
cell's FIRST SURVEY YEAR is not random -- it is structured in space. Nothing in the encoder
conditions on it (`first_year_weight` is the OBSERVER's first year on a route, a different
effect). Two consequences worth measuring rather than assuming:

1. Any "early era" statistic is computed on whichever cells existed early, which is an eastern,
   long-record subset. For House Finch specifically that is the introduced/invading population,
   not a continental sample -- so an early-vs-modern difference confounds era with geography.

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
    print(f"\n=== first survey year vs EASTING (grid column; higher = further east) ===")
    print(f"  Pearson r(first_year, column) = {np.corrcoef(fy, ccol)[0, 1]:+.3f}")
    qs = np.quantile(ccol, [0, .25, .5, .75, 1.0])
    print(f"  {'easting quartile':<20}{'cells':>7}{'median first yr':>17}{'% by 1970':>11}")
    for i in range(4):
        m = (ccol >= qs[i]) & (ccol <= qs[i + 1] if i == 3 else ccol < qs[i + 1])
        if m.sum():
            print(f"  Q{i + 1} (west->east){'':<6}{int(m.sum()):>7}{np.median(fy[m]):>17.0f}"
                  f"{100 * np.mean(fy[m] <= 1970):>10.1f}%")

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
    print(f"\n  {'easting quartile':<20}{'all cells':>10}{'pass both':>11}{'pass rate':>11}")
    for i in range(4):
        m = (ccol >= qs[i]) & (ccol <= qs[i + 1] if i == 3 else ccol < qs[i + 1])
        if m.sum():
            print(f"  Q{i + 1} (west->east){'':<6}{int(m.sum()):>10}{int((m & both).sum()):>11}"
                  f"{100 * (m & both).sum() / m.sum():>10.1f}%")
    if both.sum():
        print(f"\n  mean easting: all cells {ccol.mean():.1f}  vs gated {ccol[both].mean():.1f}  "
              f"(shift {ccol[both].mean() - ccol.mean():+.1f} grid cells)")
        print(f"  mean northing: all cells {crow.mean():.1f}  vs gated {crow[both].mean():.1f}  "
              f"(shift {crow[both].mean() - crow.mean():+.1f})")

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

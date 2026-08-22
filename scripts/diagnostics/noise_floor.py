"""How much of the measured community change is real turnover, and how much is observation noise?

The temporal question grades predictors against Ruzicka between a cell's early community and its
modern one -- measured at 0.62, i.e. 38% apparent change. But the endpoints are AVERAGES of few
surveys (mean 4.2 surveyed years early, 11.4 modern, at ~1.08 BBS routes per cell-year), and noise
pushes two observations of the SAME unchanged community apart. So 0.62 is depressed by noise, and
an unknown share of that 38% is not turnover at all.

This matters because the predictors are denoised to DIFFERENT degrees. DESK is a smooth function of
covariates with a ~10.8 yr output EMA, so it is denoised by construction and predicts 0.955 -- near
stasis. The oracle projects the same noisy averages and gives 0.857. Grading a denoised prediction
against a noisy target penalises it for failing to reproduce noise, which would make a correct
model look flat. Without the floor below, "DESK predicts stasis" cannot be distinguished from
"DESK correctly declines to predict noise".

THE MEASUREMENT. Split one cell's surveys WITHIN a single era into two disjoint halves, average
each the same way the real endpoints are averaged, and take the Ruzicka between them. Same place,
same era, so no real turnover is possible: whatever falls below 1.0 is noise. Reported at matched
sample sizes, because a 2-survey average is noisier than a 6-survey one and the real endpoints
differ (4.2 vs 11.4), which by itself biases the early endpoint downward relative to the modern.

READING IT:
  floor ~= 0.62 (the cross-era value) -> the apparent change is all noise, there is no detectable
      turnover at this sample size, and the temporal test is underpowered. DESK's 0.955 is not
      evidence of anything wrong.
  floor ~= 0.85+ -> real turnover is large, and DESK's 0.955 is genuinely too flat.
  in between -> the honest statement is a noise-corrected change, and every predictor should be
      compared against the floor rather than against 1.0.

Run: cd $HOUFIN_REPO && python scripts/diagnostics/noise_floor.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.community_encoder.train_DESK.config_utils import load_config          # noqa: E402

EARLY = (1966, 1986)
MODERN = (2005, 2025)


def ruzicka_rows(A, B):
    mn = np.minimum(A, B).sum(1)
    mx = np.maximum(A, B).sum(1)
    return np.where(mx > 0, mn / np.maximum(mx, 1e-12), np.nan)


def main():
    cfg = load_config(os.environ.get("ESK_DESK_CONFIG") or None)
    from src.community_encoder.train_DESK.validate_bbs_routes import load_observed
    from src.data.preprocess.bbs_community import log1p_community
    X_log, keys, meta, X_raw = load_observed(cfg)
    rows, cols, yrs = keys[:, 0], keys[:, 1], keys[:, 2]
    cell = rows.astype(np.int64) * 100000 + cols
    rng = np.random.default_rng(0)

    print(f"{len(keys):,} surveyed cell-years, {len(np.unique(cell)):,} cells, "
          f"years {yrs.min()}-{yrs.max()}")

    # index surveys by (cell, era)
    for era_name, (lo, hi) in (("EARLY " + str(EARLY), EARLY), ("MODERN " + str(MODERN), MODERN)):
        m = (yrs >= lo) & (yrs <= hi)
        by = {}
        for i in np.flatnonzero(m):
            by.setdefault(int(cell[i]), []).append(i)
        # need >=2 surveys to split; report by how many each half gets
        print(f"\n=== noise floor within {era_name} "
              f"-- same cell, same era, so NO real turnover is possible")
        print(f"  {'per half':<10}{'cells':>8}{'median R':>11}{'q25':>8}{'q75':>8}")
        for per_half in (1, 2, 3, 4, 6):
            vals = []
            for c, idx in by.items():
                if len(idx) < 2 * per_half:
                    continue
                pick = rng.permutation(idx)[:2 * per_half]
                a = log1p_community(X_raw[pick[:per_half]].mean(0, keepdims=True))
                b = log1p_community(X_raw[pick[per_half:]].mean(0, keepdims=True))
                vals.append(ruzicka_rows(a, b)[0])
            if len(vals) >= 20:
                v = np.asarray(vals, "float64")
                v = v[np.isfinite(v)]
                print(f"  {per_half:<10}{len(v):>8}{np.median(v):>11.4f}"
                      f"{np.quantile(v, .25):>8.4f}{np.quantile(v, .75):>8.4f}")
            else:
                print(f"  {per_half:<10}{len(vals):>8}   too few cells")

    # The comparison that matters: cross-era at the SAME per-side sample sizes as the floor.
    print("\n=== cross-era (real turnover PLUS noise), at matched sample sizes ===")
    be, bm = {}, {}
    for i in np.flatnonzero((yrs >= EARLY[0]) & (yrs <= EARLY[1])):
        be.setdefault(int(cell[i]), []).append(i)
    for i in np.flatnonzero((yrs >= MODERN[0]) & (yrs <= MODERN[1])):
        bm.setdefault(int(cell[i]), []).append(i)
    shared = sorted(set(be) & set(bm))
    print(f"  {len(shared):,} cells surveyed in both eras")
    print(f"  {'per side':<10}{'cells':>8}{'median R':>11}{'floor*':>9}{'excess':>9}")
    for per in (1, 2, 3, 4):
        cross, floor = [], []
        for c in shared:
            if len(be[c]) < 2 * per or len(bm[c]) < per:
                continue
            pe = rng.permutation(be[c]); pm = rng.permutation(bm[c])
            a = log1p_community(X_raw[pe[:per]].mean(0, keepdims=True))
            b = log1p_community(X_raw[pm[:per]].mean(0, keepdims=True))
            cross.append(ruzicka_rows(a, b)[0])
            # the floor on the SAME cells, from the early era's other half
            a2 = log1p_community(X_raw[pe[per:2 * per]].mean(0, keepdims=True))
            floor.append(ruzicka_rows(a, a2)[0])
        if len(cross) >= 20:
            cr = np.asarray(cross, "float64"); fl = np.asarray(floor, "float64")
            ok = np.isfinite(cr) & np.isfinite(fl)
            print(f"  {per:<10}{int(ok.sum()):>8}{np.median(cr[ok]):>11.4f}"
                  f"{np.median(fl[ok]):>9.4f}{np.median(fl[ok]) - np.median(cr[ok]):>+9.4f}")
    print("\n  * floor computed on the SAME cells, from two halves of the EARLY era only.")
    print("    excess = floor - cross-era: how much LESS similar the cross-era pair is than two")
    print("    same-era observations of the same place. That excess is the only part that can be")
    print("    real turnover. If it is ~0, the temporal signal is not measurable at this n and no")
    print("    predictor can be graded on it -- including DESK's 0.955.")


if __name__ == "__main__":
    main()

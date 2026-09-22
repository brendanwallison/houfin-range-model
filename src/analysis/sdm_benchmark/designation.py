"""Binary designations and the niche x occupancy 2x2.

THE COMPARISON IS BETWEEN DESIGNATIONS, NOT MAPS. The dynamic model emits
``lam_fundamental >= 1`` -- can a population sustain itself here WITHOUT
immigration. A correlative SDM emits a probability surface which, thresholded, is
a suitability call inferred FROM occurrence. Whether either map "looks like" it
has a Great Plains gap is not the test; whether the two designations agree is.

The 2x2 uses the codes already fixed by scripts/viz/hypothesis_scenarios.py:

    0  unoccupied sink      (not occupied, not niche)
    1  unoccupied source    (not occupied, niche)
    2  occupied source      (occupied, niche)
    3  OCCUPIED SINK        (occupied, NOT niche)  <- the diagnostic category

A correlative model cannot populate cells 1 and 3 other than by threshold noise:
it infers the niche axis FROM the occupancy axis, so its table is diagonal by
construction (Pulliam 2000). Quantifying that emptiness -- against a dynamic
model that puts real mass in cell 3 over the Great Plains -- is the result.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.stats import rankdata

# Codes shared with scripts/viz/hypothesis_scenarios.py.
CELL_UNOCC_SINK, CELL_UNOCC_SOURCE, CELL_OCC_SOURCE, CELL_OCC_SINK = 0, 1, 2, 3
CELL_NAMES = {
    CELL_UNOCC_SINK: "unoccupied_sink",
    CELL_UNOCC_SOURCE: "unoccupied_source",
    CELL_OCC_SOURCE: "occupied_source",
    CELL_OCC_SINK: "occupied_sink",
}


def load_dynamic_fields(run_dir):
    """Load the dynamic model's source/sink fields from a completed MAP run.

    Prefers the .npz (named keys) over the .tif (positional bands). Returns the
    dict of arrays plus ``window_years``; every field is (ny, nx) with NaN off
    land.
    """
    base = os.path.join(run_dir, "map_diagnostics", "07_source_sink_fields")
    npz = base + ".npz"
    if os.path.exists(npz):
        z = np.load(npz)
        return {k: z[k] for k in z.files}
    tif = base + ".tif"
    if not os.path.exists(tif):
        raise FileNotFoundError(
            f"No 07_source_sink_fields.{{npz,tif}} under {run_dir}. Run the map "
            f"diagnostics pass for that run first.")
    import rasterio
    with rasterio.open(tif) as src:               # band order per _write_source_sink_fields
        return {"lam_realized_modern": src.read(1),
                "lam_fundamental_modern": src.read(2),
                "K_modern": src.read(3)}


def niche_from_lambda(lam_fundamental, threshold=1.0):
    """The dynamic model's niche designation. lam >= 1 is the meaningful cut."""
    out = np.zeros(lam_fundamental.shape, dtype=bool)
    finite = np.isfinite(lam_fundamental)
    out[finite] = lam_fundamental[finite] >= threshold
    return out, finite


def cell_occupancy(df, ny, nx, min_routes=1):
    """Aggregate route-years to per-cell occupancy over the analysis window.

    Returns ``(occupied, surveyed, mean_count)``, each (ny, nx). A cell is
    occupied if ANY route-year in it recorded the species; surveyed records where
    there is evidence at all, so unsurveyed cells never enter a contingency table
    as absences.
    """
    occupied = np.zeros((ny, nx), dtype=bool)
    surveyed = np.zeros((ny, nx), dtype=bool)
    total = np.zeros((ny, nx), dtype=float)
    n = np.zeros((ny, nx), dtype=float)

    r = df["row"].to_numpy()
    c = df["col"].to_numpy()
    y = df["count"].to_numpy(dtype=float)
    np.add.at(total, (r, c), y)
    np.add.at(n, (r, c), 1.0)
    np.logical_or.at(occupied, (r, c), y > 0)
    surveyed[r, c] = True

    if min_routes > 1:
        surveyed &= n >= min_routes
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_count = np.where(n > 0, total / np.maximum(n, 1), np.nan)
    return occupied, surveyed, mean_count


def contingency(niche, occupied, mask):
    """The niche x occupancy 2x2 over ``mask``. Returns counts, fractions, codes."""
    m = mask & np.isfinite(mask.astype(float))
    nb = niche[m]
    ob = occupied[m]
    codes = np.where(ob, np.where(nb, CELL_OCC_SOURCE, CELL_OCC_SINK),
                     np.where(nb, CELL_UNOCC_SOURCE, CELL_UNOCC_SINK))
    counts = {CELL_NAMES[k]: int((codes == k).sum()) for k in CELL_NAMES}
    tot = max(int(m.sum()), 1)
    return {
        "n_cells": tot,
        "counts": counts,
        "fractions": {k: round(v / tot, 4) for k, v in counts.items()},
        "occupied_sink_fraction": round(counts["occupied_sink"] / tot, 4),
    }


def cohens_kappa(a, b, mask):
    """Agreement between two boolean designations, chance-corrected."""
    x, y = a[mask], b[mask]
    n = x.size
    if n == 0:
        return float("nan")
    po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return float("nan") if pe == 1.0 else (po - pe) / (1 - pe)


def auc(scores, labels):
    """Mann-Whitney AUC. No sklearn dependency so this stays testable standalone."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _sens_spec(scores, labels, thr):
    pred = scores >= thr
    tp = float((pred & labels).sum()); fn = float((~pred & labels).sum())
    tn = float((~pred & ~labels).sum()); fp = float((pred & ~labels).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def threshold_max_sss(scores, labels):
    """maxSSS: the threshold maximizing sensitivity + specificity.

    The standard binarization rule for presence/absence SDMs (Liu et al. 2005,
    2013). Note what it means for this benchmark: the cut is calibrated AGAINST
    observed occurrence, so the resulting "suitable" designation is occupancy-
    derived by construction. That is the point being measured, not a flaw to fix.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    cands = np.unique(scores)
    if cands.size > 2000:                       # cap the sweep on big surfaces
        cands = np.quantile(scores, np.linspace(0, 1, 2000))
    best, best_thr = -np.inf, float("nan")
    for t in cands:
        sens, spec = _sens_spec(scores, labels, t)
        if np.isfinite(sens) and np.isfinite(spec) and sens + spec > best:
            best, best_thr = sens + spec, float(t)
    return best_thr


def threshold_p10(scores, labels):
    """10th-percentile training presence: the standard sensitivity alternative."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    pres = scores[labels & np.isfinite(scores)]
    return float(np.percentile(pres, 10)) if pres.size else float("nan")


def tss(scores, labels, thr):
    """True Skill Statistic at a threshold: sensitivity + specificity - 1."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    ok = np.isfinite(scores)
    sens, spec = _sens_spec(scores[ok], labels[ok], thr)
    return sens + spec - 1.0

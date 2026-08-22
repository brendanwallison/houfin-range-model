"""Route-level BBS validation: DESK vs a no-change null, in similarity space.

WHY THIS EXISTS. Every other DESK validation grades against an IDW-interpolated target: the
BBS trend product is USGS's published inverse-distance surface (see ``bbs_trend.py``), further
smoothed by ``trend.smooth_sigma_cells``. So the IDW baseline that beats DESK on held-out cells
is approximately the target's own construction operator -- the metric rewards mimicking a
smoother, and no number computed against it can settle whether DESK has real temporal skill.

Raw BBS route counts escape that circularity: they are the observations the IDW surface was
interpolated FROM. Restricting evaluation to cell-years that were *actually surveyed* cannot be
won by reproducing an interpolator.

WHAT IT MEASURES. One question: does DESK beat a no-change null at reproducing observed
community turnover at surveyed sites -- overall, and separately in a modern and an early window.

    S_true = Ruzicka(log1p observed community)          <- raw BBS, routes averaged per cell-year
    S_desk = Z_ema @ Z_ema.T                            <- the DOT product: the kernel contract
    S_nc   = same dot, but DESK's z at each cell's MODERN year copied to all its years

    PRIMARY   rmse_skill = 1 - rmse(S_desk, S_true) / rmse(S_nc, S_true)
    SECONDARY cka_gain   = CKA(S_true, S_desk) - CKA(S_true, S_nc)

WHY THE DOT PRODUCT. ESK/DESK promise ``Z(x) dot Z(x') ~= uncentered Ruzicka(x, x')`` (recorded as
``kernel_contract`` in every ESK/cube meta), and ``desk_training.true_kernel_loss`` is the MSE
between that dot product and the Ruzicka similarity of the raw communities. The dot product is
therefore both the model's stated output and the quantity it was optimized for. Because the
contract makes the two directly comparable rather than merely correlated, they can be compared
ELEMENTWISE -- which is strictly more informative than CKA, and catches a norm-calibration error
that CKA is provably blind to (see tests).

The cosine columns are NOT a secondary curiosity, which is how they were previously labelled. They
are the ANGULAR HALF of an exact decomposition of the dot-product error::

    ||a - b||^2 = (||a|| - ||b||)^2  +  2 ||a|| ||b|| (1 - cos t)
                  |-- magnitude --|     |------- angular -------|

so dot and cosine together account for the whole error with no residual, and the magnitude half is
exactly the ``||z||^2`` deficit (measured ~0.60 against a contract of 1.0). Reading one without the
other hides half the error -- and hides that the two TRADE OFF: minimising the total at a fixed
direction cosine ``rho`` gives ``||a|| = rho*||b||``, so shrinking is the MSE-optimal response to a
poor angle. See ``validate_baselines.error_decomposition``.

BOTH MODEL-SIDE MATRICES USE THE SAME FUNCTIONAL, so the ONLY difference between them is temporal
variation. That is load-bearing. An earlier version built the null as Ruzicka on the observed
modern community; because truth is also Ruzicka, that null was scored in truth's own metric while
DESK was not, and the difference between the two similarity functions swamped the temporal signal
(measured: a temporally-neutral model scored -0.28 purely from the mismatch). The observed-space
null is still reported, as ``cka_nochange_observed`` / ``rmse_nochange_observed``, but only as the
achievable-ceiling reference -- never differenced against the DESK columns.

THE GAIN IS THE READOUT, NOT ABSOLUTE CKA. A mixed similarity matrix is dominated by
between-cell (spatial) structure -- the same reason ``Val`` MSE says little about temporal skill.
The no-change null has ZERO temporal variation by construction, so differencing against it
isolates exactly the temporal component. A large ``cka_desk`` with ``cka_gain <= 0`` means DESK
reproduced the map and nothing about change.

TRUTH STAYS OUT OF Z SPACE; THE BARS DO NOT. ``S_true`` is Ruzicka on raw counts and owes nothing
to any fitted object -- that is what makes this module uncircular, and it does not change.

This section used to say z-space projection was unavailable here, because "the ESK landmarks are
frozen at grid-product magnitude, so projecting route counts would require inventing a per-species
BBS->eBird scale factor". That was true of the retired trend-product basis. It is stale:
``run_spacetime_esk`` now fits the basis on ``target_points_dir`` (= ``bbs_points``, raw route-level
BBS), so the landmarks are already at raw-BBS magnitude, there is no scale factor to invent, and
``validate_spacetime`` performs this projection on this same point set every run.

Two things depend on the projection, and both are BARS rather than truth:

* the spacetime-IDW bar -- an actual alternative predictor, needed because ``S_nc`` is a
  decomposition device and nearly free to beat (measured: on ``self_change`` the null already
  correlates 0.53-0.63 with observed change while carrying zero temporal content);
* the ESK oracle on ``self_change`` -- the ceiling, which says whether the basis can represent
  same-cell temporal change at all, and so whether a weak DESK row is the model's fault.

The residual cost, stated rather than left implicit: those two carry basis dependence, the truth
they are scored against does not. A bar must also use DESK's OWN functional -- a dot product of
ESK-space vectors -- for the reason recorded below: scoring a bar in Ruzicka while DESK is scored
in dot product was worth -0.28 on a temporally-neutral model, from the metric mismatch alone.

READ ``rmse_skill``, NOT the absolute ``rmse_desk``. Bare Ruzicka is invariant when both arguments
are rescaled, but ``log1p`` is NOT (``log1p(c*x) != c*log1p(x)``), so the absolute count scale does
survive into ``S_true``. Training applied log1p to grid-scale abundances; this applies it to
per-route mean counts, and those magnitudes differ. That inflates ``rmse_desk`` and ``bias_desk``
by an unknown offset, so their ABSOLUTE values are diagnostic only.

``rmse_skill`` is unaffected: it is a ratio of two errors measured against the SAME ``S_true``, so
whatever the units mismatch does, it does to DESK and the null alike. Same for ``cka_gain``. The
comparison between the two models is sound; the calibration of either one against observed
Ruzicka in absolute terms is not.

    python -m src.community_encoder.train_DESK.validate_bbs_routes
"""
import argparse
import json
import os
import time

import numpy as np

from src.config_utils import load_config

from .desk_training import apply_output_ema
from .validate_spacetime import linear_cka, mantel_r, ruzicka_rect

# Window definitions. 1966, not 1965: BBS begins in 1966, so a 1965 bound would silently be a
# 1966 bound and misreport the window in the output.
MODERN_WINDOW = (2010, 2025)
EARLY_WINDOW = (1966, 1980)


# ----------------------------- pure: observed community -----------------------------

# densify_community / log1p_community moved to src/data/preprocess/bbs_community.py: they are
# now the shared core of BOTH the raw-BBS training target and this validation, and the two have
# to densify identically or they are not measuring the same community. Re-exported here so this
# module's public surface (and its tests) are unchanged.
from src.data.preprocess.bbs_community import (            # noqa: E402  (deliberate: see above)
    densify_community, log1p_community,
)


def modern_reference_rows(keys, modern_window=MODERN_WINDOW):
    """Map each cell to its most recent surveyed row inside ``modern_window`` (pure).

    Returns ``(nc_src (N,) int64, keep (N,) bool)`` where ``nc_src[i]`` is the row index holding
    cell ``i``'s modern observed community and ``keep[i]`` is False for every row of a cell with
    NO surveyed year in the window. Those cells are dropped from EVERY matrix (truth included) --
    the same ``has_rec`` gate ``zspace_reconstruction`` applies, at cell granularity. Without the
    gate a no-change null cannot be defined for those sites, and silently keeping them in truth
    while excluding them from the null would compare two different row sets.
    """
    keys = np.asarray(keys)
    n = keys.shape[0]
    lo, hi = int(modern_window[0]), int(modern_window[1])
    best = {}
    for i in range(n):
        r, c, y = int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])
        if lo <= y <= hi:
            cell = (r, c)
            if cell not in best or y > best[cell][0]:
                best[cell] = (y, i)
    nc_src = np.full(n, -1, dtype="int64")
    for i in range(n):
        hit = best.get((int(keys[i, 0]), int(keys[i, 1])))
        if hit is not None:
            nc_src[i] = hit[1]
    return nc_src, nc_src >= 0


def window_groups(keys, half_width):
    """``[rows, ...]`` per row -- that cell's rows within +/-half_width YEARS of the row's own year.

    Indices are in the SAME row space as ``keys``, which must be the UNFILTERED key array: the
    caller averages ``X_raw`` with these, and ``X_raw`` is never site-gated (see the note at the
    gate). Handing in post-gate keys would index the wrong rows -- the failure mode already
    recorded there as "IndexError at 110878 vs size 110839".

    ``half_width=0`` returns each row alone, so the windowed path reduces exactly to the
    historical per-row behaviour.
    """
    keys = np.asarray(keys)
    hw = int(half_width)
    by_cell = {}
    for i in range(keys.shape[0]):
        by_cell.setdefault((int(keys[i, 0]), int(keys[i, 1])), []).append(i)
    out = []
    for i in range(keys.shape[0]):
        y = int(keys[i, 2])
        out.append(tuple(j for j in by_cell[(int(keys[i, 0]), int(keys[i, 1]))]
                         if abs(int(keys[j, 2]) - y) <= hw))
    return out


def modern_reference_groups(keys, modern_window=MODERN_WINDOW):
    """``(groups, keep)`` -- ALL of each cell's rows inside ``modern_window``, not just the last.

    ``modern_reference_rows`` picks each cell's most recent surveyed row, which uses a 16-year
    window to select a single route-year. At ~1.08 routes per cell-year that reference is one
    observer on one morning, and it appears in every no-change null and every observed-space
    ceiling in this report -- so its noise propagates everywhere, attenuating the measured
    similarity toward zero.

    Averaging the window instead cuts that noise by ~sqrt(n_years). ``groups[i]`` is the tuple of
    row indices to average for row ``i``; ``keep`` is the same site gate as
    ``modern_reference_rows`` (a cell with no survey in the window has no definable null), so the
    row population and the gate are unchanged and only the reference VALUE differs.

    The caller must average the observed community and the model's z over the SAME group. Doing
    one and not the other would compare a smoothed truth against an unsmoothed prediction, which
    is the asymmetry this module exists to avoid.
    """
    keys = np.asarray(keys)
    lo, hi = int(modern_window[0]), int(modern_window[1])
    by_cell = {}
    for i in range(keys.shape[0]):
        y = int(keys[i, 2])
        if lo <= y <= hi:
            by_cell.setdefault((int(keys[i, 0]), int(keys[i, 1])), []).append(i)
    groups = [tuple(by_cell.get((int(keys[i, 0]), int(keys[i, 1])), ()))
              for i in range(keys.shape[0])]
    keep = np.array([len(g) > 0 for g in groups], dtype=bool)
    return groups, keep


def dot_gram(Z):
    """``Z @ Z.T`` → ``(n, n)``. THE prediction, per the kernel contract.

    ESK/DESK promise ``Z(x) dot Z(x') ~= uncentered Ruzicka(x, x')`` (recorded as
    ``kernel_contract`` in every ESK/cube meta), and ``desk_training.true_kernel_loss`` is
    literally the MSE between this dot product and the Ruzicka similarity of the raw communities.
    So the dot product is what the model was optimized to produce and what must be graded.

    Using cosine here instead would discard the vector norms, which are part of the prediction --
    the contract fixes the SCALE, not just the direction. Cosine is the right choice for
    per-cell TURNOVER (``1 - similarity``), where a norm deficit masquerades as change; it is the
    wrong choice for asking whether the kernel reproduces observed similarity.
    """
    Z = np.asarray(Z, "float64")
    return Z @ Z.T


def cosine_gram(Z):
    """Row-normalized ``Z @ Z.T`` -- SECONDARY diagnostic only, not the graded quantity.

    Reported alongside the dot product because the norm deficit is real and measured (``||Z||^2``
    medians 0.73-0.81 rather than the contract's 1.0). If the dot-product result is poor while
    cosine is good, the failure is calibration of ``||Z||``; if both are poor, it is the angular
    structure. Keeping both separates those two explanations.
    """
    Z = np.asarray(Z, "float64")
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    Zn = Z / np.where(nrm > 0, nrm, 1.0)
    return Zn @ Zn.T


def _offdiag(S):
    """Upper-triangle off-diagonal entries. The diagonal is self-similarity (trivially 1 for
    Ruzicka and ``||z||^2`` for the dot) and would flatter both models identically."""
    return np.asarray(S)[np.triu_indices_from(np.asarray(S), k=1)]


def vector_error(t, p):
    """Elementwise agreement between flat arrays of observed and predicted similarity (pure).

    The shared core of ``kernel_error`` (which feeds it matrix off-diagonals) and the epoch
    analysis (which feeds it focal-to-neighbour pair vectors), so both report identical
    definitions. See ``kernel_error`` for what each field means.
    """
    t, p = np.asarray(t, "float64").ravel(), np.asarray(p, "float64").ravel()
    d = p - t
    sd = float(t.std())
    r = float(np.corrcoef(t, p)[0, 1]) if t.size > 1 and sd > 0 and p.std() > 0 else 0.0
    rmse = float(np.sqrt(np.mean(d ** 2))) if t.size else float("nan")
    return {"rmse": rmse, "bias": float(np.mean(d)) if t.size else float("nan"),
            "pearson_r": r, "r2": (1.0 - (rmse / sd) ** 2) if sd > 0 else 0.0,
            "sd_true": sd, "mean_true": float(t.mean()) if t.size else float("nan"),
            "n": int(t.size)}


#: WHAT THIS SUITE TESTS, and why -- the map. These four pairings were previously implicit in where
#: the code happened to call things, which made it possible (and it happened, repeatedly) to add a
#: predictor or a quantity to some of them and not others, and to propose building a test that
#: already existed. Every question is evaluated against every predictor in PREDICTOR_ROLES, and
#: ``assert_complete`` requires a result or a stated reason for each combination.
QUESTIONS = {
    "same_cell_over_time": {
        "pairs": "one cell, early era vs modern era",
        "observed": "Ruzicka(Xe_i, Xm_i)",
        "why": ("the temporal question proper -- did THIS place change the way the model says. The "
                "only pairing where 'did it change correctly' is directly interpretable, and the "
                "one the downstream kernel has to get right."),
    },
    "cross_cell_same_era": {
        "pairs": "two cells, both early or both modern",
        "observed": "Ruzicka(Xe_i, Xe_j) / Ruzicka(Xm_i, Xm_j)",
        "why": ("the spatial control. `spatial_modern` is identical to its own no-change null by "
                "construction, so its skill must be EXACTLY 0 -- the harness's built-in zero "
                "check. If it ever moves, no other row can be trusted."),
    },
    "cross_cell_cross_time": {
        "pairs": "historic cell i against modern cell j",
        "observed": "Ruzicka(Xe_i, Xm_j)",
        "why": ("the analog question: is a historic community like some modern community "
                "elsewhere. Distance-resolved, so it also says at what range the resemblance "
                "holds. Needs the ANGULAR form especially, since the dot form carries the ~34% "
                "self-similarity deficit and this pairing spans both eras."),
    },
    "absolute_position": {
        "pairs": "one cell-year against its own observation",
        "observed": "||z_pred - z_obs|| at that cell-year",
        "why": ("where a place IS, not how it moved. Separating this from "
                "`same_cell_over_time` is what revealed that DESK can have the direction of "
                "change right while the absolute position is wrong -- a cell-specific offset "
                "hurts position and cancels in a difference."),
    },
}

#: The predictors every question is graded against. Names are the report's addressing, so they must
#: not drift: adding one here adds a row to every question automatically, which is the property the
#: `compare_predictors` refactor exists to buy.
PREDICTOR_ROLES = {
    "desk": "the model under test",
    "spacetime_idw": ("the honest bar -- interpolate observed z in space AND time from training "
                      "rows only. The no-change null is a decomposition device and nearly free to "
                      "beat; this is an actual alternative predictor."),
    "no_change": ("frozen-modern null: the model's own z held at each cell's modern year. Shares "
                  "DESK's functional, so their difference isolates temporal variation. NOT a "
                  "competitor."),
    "esk_oracle": ("ceiling -- the observed community's own projection in place of the model's. "
                   "Says whether the BASIS can represent the pairing at all, so a weak DESK row "
                   "can be attributed to the model rather than inherited."),
}


#: Reasons a predictor may be missing from a question. NAMED and specific, because a single
#: catch-all defeats the completeness check that consumes them.
#:
#: The catch-all this replaced read "inputs not supplied to this run (bar unbuilt, or oracle refused
#: its representability gate)" -- a DISJUNCTION. It could not distinguish a refusal, which is a
#: real finding about the basis, from a wiring omission, which is a bug. The route-level buckets
#: never passed the oracle in at all and stamped that reason over the hole, so a code gap read as a
#: measured decision, `assert_complete` returned zero gaps, and the missing ceiling went unnoticed
#: across several runs. A reason is only as good as its specificity.
UNAVAILABLE_NOT_WIRED = ("not wired into this question -- a CODE GAP, not a property of the data. "
                         "The inputs exist; nothing passes them here.")
UNAVAILABLE_NOT_SUPPLIED = "caller passed None for this predictor"
UNAVAILABLE_BAR_UNBUILT = "interpolation bar could not be built for this run"
UNAVAILABLE_ORACLE_GATE = "oracle refused its representability gate"


def canonical_question(name):
    """Map an emitted question key to its registry entry, or None if it belongs to no question.

    A question may be emitted as several INSTANCES -- `cross_cell_same_era` runs once within the
    early era and once within the modern -- and they share one registry entry because they share
    one rationale. Matching is by longest registry prefix on a `_` boundary rather than by
    stripping the last token, so a question whose own name contains underscores
    (`same_cell_over_time`) cannot be mangled into a prefix that happens to exist.
    """
    if name in QUESTIONS:
        return name
    cands = [q for q in QUESTIONS if name.startswith(q + "_")]
    return max(cands, key=len) if cands else None


def _predictor_leaves(node, path):
    """Yield every (path, table) that carries a predictor table, at whatever depth it sits.

    Questions differ in shape -- the neighbour questions nest split/distance-bin/form, while
    `absolute_position` nests only populations -- so the walk finds leaves BY SHAPE rather than by
    a fixed depth. Depth-coupling is what silently disabled the first version of this check: it
    read `results[q]["predictors"]`, a level the analysis never emits, and so inspected nothing
    while reporting green.
    """
    if not isinstance(node, dict):
        return
    if "predictors" in node or "unavailable" in node:
        yield path, node
    if "skipped" in node:
        return                                     # a stated reason, which is what we require
    for k, v in node.items():
        if isinstance(v, dict) and k not in ("predictors", "unavailable"):
            yield from _predictor_leaves(v, f"{path}/{k}")


def assert_complete(results, predictors=tuple(PREDICTOR_ROLES), questions=tuple(QUESTIONS)):
    """Every (question x predictor) must have a result or a stated reason. Returns the gaps.

    This is the check that makes a silently-missing comparison impossible. Every gap that
    accumulated in this suite -- the bar reaching the pooled matrices but not the per-distance
    rows, the decomposition computed for DESK and not the bar, the cosine form present for the
    pooled matrices and absent from all four epoch types -- would have been caught here, because
    each was an absent key rather than a wrong number and nothing looked for absences.

    `questions` is the scope the report is RESPONSIBLE for, and callers must pass their own: this
    module does not produce `absolute_position` (that is `zspace_reconstruction`'s), and a check
    that flagged it on every run would train the reader to ignore the output -- the exact habit
    this exists to break. Scope is declared by the report as `covers`, never inferred from what it
    emitted, which would make the check compare the output to itself and always pass.

    Descends to the leaves. A predictor present in the pooled row but missing from `heldout`, or
    from one distance bin, is precisely the class of gap that kept recurring, so checking only the
    top level would reproduce the blind spot.
    """
    gaps = []
    seen = {}
    for emitted, r in (results or {}).items():
        q = canonical_question(emitted)
        if q is None:
            gaps.append(f"{emitted}: emitted but in no question registry")
            continue
        seen.setdefault(q, []).append(emitted)
        leaves = list(_predictor_leaves(r, emitted))
        if not leaves:
            gaps.append(f"{emitted}: question present but carries no predictor table")
        for where, cell in leaves:
            have = set(cell.get("predictors", {})) | set(cell.get("unavailable", {}))
            for pname in predictors:
                if pname not in have:
                    gaps.append(f"{where} x {pname}: neither a result nor a reason")
    for q in questions:
        if q not in seen:
            gaps.append(f"{q}: question absent entirely")
    return gaps


def compare_positions(z_obs, predictors, reference="no_change", populations=None):
    """Grade every predictor's ABSOLUTE POSITION against the observed z, on identical terms.

    The `absolute_position` question asks where a place IS, not how it moved, so its truth is a
    matrix of observed latents rather than a similarity vector -- `compare_predictors` does not
    fit. This shares that function's OUTPUT vocabulary exactly (`predictors` / `skill_vs` /
    `unavailable` / `reference`) so both questions address the same way and one completeness
    check covers both.

    WHY THIS REPLACED THE HAND-WRITTEN BLOCKS in `zspace_reconstruction`. That function grew one
    key pair per baseline -- `frac_desk_beats_nochange`, `frac_desk_beats_idw`,
    `frac_desk_beats_spacetime_idw` -- so DESK was the only possible SUBJECT of a comparison.
    "Does the spacetime bar beat the same-year bar", which says whether borrowing across time
    helps at all, could not be asked. The error decomposition was computed for DESK alone, so
    "does the bar win on magnitude while DESK wins on direction" had no answer for the bar. And
    the populations were applied unevenly -- the null got train/heldout, the same-year bar got
    heldout only, the spacetime bar got heldout plus withheld-years -- because each was bolted on
    where it was written.

    `populations` is `{name: mask}` over the same rows; every predictor is scored on every one.

    Predictors may carry NaN rows (an interpolation bar reaches only where it has neighbours).
    Each predictor's own summaries use its own finite rows, but a WIN RATE uses the intersection
    of the pair's finite rows -- scoring a bar's easy subset against a reference's full set would
    otherwise flatter whichever predictor declined the hardest rows.
    """
    from .validate_baselines import error_decomposition

    z_obs = np.asarray(z_obs, "float64")
    n_o = np.linalg.norm(z_obs, axis=1)

    def _summary(P, mask):
        P = np.asarray(P, "float64")
        err = np.linalg.norm(P - z_obs, axis=1)
        fin = np.isfinite(err) & mask
        if fin.sum() < 4:
            return None
        tot, mag, ang, cos = error_decomposition(P[fin], z_obs[fin])
        n_p = np.linalg.norm(P[fin], axis=1)
        return {
            "n_scored": int(fin.sum()),
            "median_err": float(np.median(err[fin])),
            "err_total_sq": float(np.median(tot)),
            "err_magnitude_sq": float(np.median(mag)),
            "err_angular_sq": float(np.median(ang)),
            # Shares of the MEAN, not of the medians: medians of two terms do not add to the
            # median of their sum, and a reader comparing the three medians would otherwise
            # conclude the identity is broken. It holds per point; these are its expectation.
            "magnitude_share": float(np.mean(mag) / max(np.mean(tot), 1e-12)),
            "angular_share": float(np.mean(ang) / max(np.mean(tot), 1e-12)),
            "median_cos_vs_obs": float(np.nanmedian(cos)),
            "median_z_norm2": float(np.median(n_p ** 2)),
            # The Ruzicka similarity the error distance actually implies. The naive 1 - d^2/2
            # assumes both norms are 1 and so flatters by exactly the norm deficit.
            "implied_ruzicka": float(np.median((n_p ** 2 + n_o[fin] ** 2 - tot) / 2.0)),
        }

    def _block(mask):
        rows, errs = {}, {}
        for name, P in predictors.items():
            if P is None:
                continue
            e = np.linalg.norm(np.asarray(P, "float64") - z_obs, axis=1)
            errs[name] = e
            r = _summary(P, mask)
            if r is not None:
                rows[name] = r
        out = {"n": int(mask.sum()), "reference": reference, "predictors": rows,
               "median_z_obs_norm2": float(np.median(n_o[mask] ** 2)) if mask.any() else None,
               "unavailable": {}}
        ref = errs.get(reference)
        if ref is not None:
            wins, skill = {}, {}
            for name, e in errs.items():
                # the intersection, so neither predictor is graded on rows the other declined
                both = mask & np.isfinite(e) & np.isfinite(ref)
                if both.sum() < 4:
                    continue
                wins[name] = float(np.mean(e[both] < ref[both]))
                skill[name] = float(1.0 - np.median(e[both]) / max(np.median(ref[both]), 1e-12))
            out["win_rate_vs"], out["skill_vs"] = wins, skill
        return out

    n = z_obs.shape[0]
    res = _block(np.ones(n, bool))
    res["populations"] = {}
    for pname, m in (populations or {}).items():
        m = np.asarray(m, bool)
        if m.sum() >= 4:
            res["populations"][pname] = _block(m)
    # A predictor that produced too few finite rows anywhere is a STATED reason, not an absence.
    for name, P in predictors.items():
        if name not in res["predictors"]:
            res["unavailable"][name] = (UNAVAILABLE_NOT_SUPPLIED if P is None else
                                        "fewer than 4 finite rows to score")
    return res


def compare_predictors(obs, predictors, reference="no_change", grams=False):
    """Grade EVERY predictor against one observed truth, on identical terms. Pure.

    ``obs`` is a flat pair vector, or a square Gram when ``grams=True``. ``predictors`` is
    ``{name: values}`` in the same shape. Returns::

        {"n", "observed_mean", "observed_sd", "reference",
         "predictors": {name: {rmse, bias, bias_share, pearson_r, r2, calibration_loss,
                               [cka, mantel]}},
         "skill_vs": {name: 1 - rmse[name]/rmse[reference]},
         "unavailable": {name: reason}}

    WHY THIS REPLACED ``pair_metrics`` / ``bucket_metrics``. Both took exactly two predictors and
    baked their names into the OUTPUT KEYS -- ``rmse_desk``, ``r2_nochange``, ``pearson_desk``. A
    third predictor could then only be bolted on at each call site, and the vocabulary made
    "DESK vs the no-change null" the one first-class comparison. The result was a line reading
    ``pair_metrics(sc_obs, sc_desk, sc_idw)  # null slot := the bar`` -- three different
    predictors pushed through two slots with their meaning carried in a comment, and a suite where
    the interpolation bar reached some questions and not others, the magnitude/angular split was
    computed for DESK and never for the bar, and the cosine form existed for the pooled matrices
    and for none of the epoch types. Every one of those was fixed individually and the class came
    back, because the primitive made asymmetry the cheapest path.

    Here no predictor is privileged. ``reference`` is a NAME, so "DESK vs no-change" and "DESK vs
    spacetime-IDW" are the same call with a different argument rather than one built-in comparison
    plus an afterthought, and adding a predictor adds a row everywhere with no call-site edit.

    ``unavailable`` carries a REASON for any predictor that could not run, so a gap is stated in
    the report rather than inferred from a missing key.
    """
    obs_a = np.asarray(obs, "float64")
    flat = (lambda A: _offdiag(np.asarray(A, "float64"))) if grams else (
        lambda A: np.asarray(A, "float64").ravel())
    t = flat(obs_a)
    rows, unavailable = {}, {}
    for name, val in (predictors or {}).items():
        if val is None:
            unavailable[name] = "predictor not supplied"
            continue
        if isinstance(val, str):                 # a reason string in the predictor slot
            unavailable[name] = val
            continue
        e = vector_error(t, flat(val))
        row = {"rmse": e["rmse"], "bias": e["bias"], "pearson_r": e["pearson_r"], "r2": e["r2"],
               "bias_share": ((e["bias"] ** 2) / (e["rmse"] ** 2)) if e["rmse"] > 0 else 0.0,
               # pearson^2 - r2 is the calibration loss: scale-free ranking minus absolute
               # accuracy, so the gap is exactly what a wrong scale or a bias costs.
               "calibration_loss": e["pearson_r"] ** 2 - e["r2"]}
        if grams:
            row["cka"] = linear_cka(obs_a, np.asarray(val, "float64"))
            row["mantel"] = mantel_r(obs_a, np.asarray(val, "float64"))
        rows[name] = row

    ref_rmse = rows.get(reference, {}).get("rmse")
    skill = {}
    for name, row in rows.items():
        skill[name] = (1.0 - row["rmse"] / ref_rmse) if (ref_rmse and ref_rmse > 0) else float("nan")
    return {"n": int(t.size),
            "observed_mean": float(t.mean()) if t.size else float("nan"),
            "observed_sd": float(t.std()) if t.size else float("nan"),
            "reference": reference, "predictors": rows,
            "skill_vs": skill, "unavailable": unavailable}


def kernel_error(S_true, S_pred):
    """Elementwise agreement between a predicted kernel and observed Ruzicka (pure).

    Only meaningful for the DOT product, where the contract says the two should be EQUAL rather
    than merely correlated -- so this is a direct calibration check that CKA cannot give.

    Returns RMSE, mean signed bias (positive = predicting MORE similarity than observed, i.e.
    under-predicting turnover), Pearson r, and ``r2``: the fraction of the variance in observed
    pairwise similarity that the prediction explains, measured against the honest know-nothing
    floor -- predicting the same average similarity for every pair, whose RMSE is exactly
    ``sd(S_true)``. So ``r2 = 1 - (rmse / sd_true)^2``, zero for that trivial model and negative
    for anything worse.

    ``r2`` and ``pearson_r ** 2`` are BOTH reported because their gap is the calibration loss:
    Pearson is scale-free and only asks whether the prediction ranks pairs correctly, while r2
    also charges for bias and wrong scale. Measured on real data the two diverge sharply
    (pearson^2 0.78 vs r2 0.55), and that gap is where the model's miscalibration lives.
    """
    return vector_error(_offdiag(S_true), _offdiag(S_pred))


def stratified_sample(keys, n_sample, rng, windows=(MODERN_WINDOW, EARLY_WINDOW), is_heldout=None,
                      spatial_regions=0):
    """Sample row indices, giving each (year-window x holdout) cell a guaranteed share (pure).

    Two separate imbalances make a uniform sample useless here, and BOTH have to be corrected or
    the decisive bucket is estimated from a handful of rows:

    - BBS coverage grows roughly monotonically, so a uniform sample is swamped by the modern
      decades and the early window ends up too thin to report.
    - Held-out cells are a small fraction of rows (measured: ~7%), so they land in the sample by
      luck. In the first run this left ``heldout/early`` with 94 rows and ``heldout/modern`` with
      73 -- and held-out is the ONLY split that answers whether the model generalizes, so the
      table's most important cells were its least reliable.

    Crossing the two axes fixes the second: strata are (modern, early, other) x (train, heldout),
    each drawing an equal share of the budget, with short strata backfilled from the remainder.
    Because the similarity matrices are n^2, raising ``n_sample`` is a quadratic-cost way to buy
    held-out rows; stratifying buys them for free.

    ``spatial_regions`` > 0 adds a THIRD axis, and it is not symmetry for its own sake. BBS is
    coast-heavy, so a population-weighted metric scores the model where the survey is dense -- and
    once the objective is rebalanced (``desk.balance``) that metric would report an IMPROVEMENT as a
    regression, because the rebalanced model trades coastal accuracy for interior accuracy. The
    deficiency the rebalance exists to fix is exactly the one a population-weighted metric cannot
    see, so without this axis the training change is not assessable.

    The regions are COARSE (``spatial_regions`` per axis, so 2 gives quadrants) and derived by
    grouping the same tiles ``esk_kernel.spacetime_strata`` uses, so the two are one definition at
    two resolutions. The fine labels cannot be used here: 3 year-windows x 2 holdout x 64 tiles is
    384 strata against a ~4,000-row budget, which is ~10 rows each with most empty.

    ``is_heldout`` (``(N,)`` bool, optional) omitted reduces this to year-window stratification.
    """
    keys = np.asarray(keys)
    n = keys.shape[0]
    if n <= n_sample:
        return np.arange(n)
    yrs = keys[:, 2]

    year_strata, claimed = [], np.zeros(n, bool)
    for lo, hi in windows:
        m = (yrs >= lo) & (yrs <= hi) & ~claimed
        year_strata.append(m)
        claimed |= m
    year_strata.append(~claimed)

    if is_heldout is None:
        splits = [np.ones(n, bool)]
    else:
        ho = np.asarray(is_heldout, bool)
        splits = [~ho, ho]
    if int(spatial_regions) > 0:
        from .esk_kernel import coarse_spatial
        reg = coarse_spatial(keys, regions=int(spatial_regions))
        regions = [reg == r for r in np.unique(reg)]
    else:
        regions = [np.ones(n, bool)]
    strata = [np.where(y & s & g)[0]
              for y in year_strata for s in splits for g in regions]
    strata = [ix for ix in strata if ix.size]
    if not strata:
        return np.arange(min(n, n_sample))

    per = max(1, n_sample // len(strata))
    picked = [rng.choice(ix, size=min(per, ix.size), replace=False) for ix in strata]
    out = np.concatenate(picked)
    if out.size < n_sample:                                  # backfill from whatever is left
        rest = np.setdiff1d(np.arange(n), out, assume_unique=False)
        if rest.size:
            extra = rng.choice(rest, size=min(n_sample - out.size, rest.size), replace=False)
            out = np.concatenate([out, extra])
    return np.sort(np.unique(out))


# ------------------- epoch x local-neighbourhood analysis -------------------
# Motivation: the pooled analysis above compares every site-year pair, which spends nearly all its
# power on two things we do not care about -- year-to-year sampling noise, and cross-continental
# pairs (Arizona vs Maine differ enormously in any year, which is why cka_desk was 0.857 while the
# gain over the null was 0.017). This section removes both: average each cell over two 21-year
# epochs, and compare each cell only against its nearest neighbours.

EPOCH_EARLY = (1966, 1986)
EPOCH_MODERN = (2005, 2025)
MIN_EPOCH_YEARS = 3
GRID_RES_M = 27000.0


def epoch_gate(keys, early=EPOCH_EARLY, modern=EPOCH_MODERN, min_years=MIN_EPOCH_YEARS):
    """Cells with >= ``min_years`` DISTINCT surveyed years in BOTH epochs (pure).

    Distinct YEARS, not route-years: three routes run in one calendar year say nothing about the
    other twenty, and the whole point of epoch-averaging is to beat down year-to-year sampling
    noise. (``raw_bbs_turnover.py`` thresholds summed route coverage instead -- a different, weaker
    criterion.) Both epochs are required because a cell missing either one has no usable
    early-vs-modern contrast.

    Returns ``(cells (M,2) int32 sorted, early_rows, modern_rows, stats)`` where ``early_rows[m]``
    / ``modern_rows[m]`` are lists of row indices into ``keys`` for cell ``m``'s surveyed years in
    that epoch. Epoch bounds are INCLUSIVE on both ends.
    """
    keys = np.asarray(keys)
    e_lo, e_hi = int(early[0]), int(early[1])
    m_lo, m_hi = int(modern[0]), int(modern[1])
    per = {}
    for i in range(keys.shape[0]):
        r, c, y = int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])
        if e_lo <= y <= e_hi:
            per.setdefault((r, c), ({}, {}))[0].setdefault(y, []).append(i)
        elif m_lo <= y <= m_hi:
            per.setdefault((r, c), ({}, {}))[1].setdefault(y, []).append(i)

    kept, e_rows, m_rows = [], [], []
    n_fail_early = n_fail_modern = n_fail_both = 0
    for cell in sorted(per):
        e_by_year, m_by_year = per[cell]
        ok_e, ok_m = len(e_by_year) >= min_years, len(m_by_year) >= min_years
        if ok_e and ok_m:
            kept.append(cell)
            # Flatten year -> rows. A cell-year appears once (densify dedups), so this is just the
            # row list, but going through the year dict is what makes the COUNT a distinct-year one.
            e_rows.append([i for y in sorted(e_by_year) for i in e_by_year[y]])
            m_rows.append([i for y in sorted(m_by_year) for i in m_by_year[y]])
        elif not ok_e and not ok_m:
            n_fail_both += 1
        elif not ok_e:
            n_fail_early += 1
        else:
            n_fail_modern += 1

    stats = {"min_years": int(min_years), "early_window": [e_lo, e_hi],
             "modern_window": [m_lo, m_hi], "cells_kept": len(kept),
             "cells_seen": len(per), "cells_failed_early_only": n_fail_early,
             "cells_failed_modern_only": n_fail_modern, "cells_failed_both": n_fail_both}
    cells = (np.array(kept, dtype="int32") if kept
             else np.zeros((0, 2), dtype="int32"))
    return cells, e_rows, m_rows, stats


def epoch_mean_observed(X_raw, row_lists):
    """Mean RAW counts over each cell's surveyed years, then log1p (pure).

    Order matters: ``log1p`` is concave, so ``mean(log1p(x)) < log1p(mean(x))`` whenever abundance
    varies across years, and the gap grows with the spread. The epoch summary we want is "the
    average community over this period", so average the counts and transform once.
    """
    X_raw = np.asarray(X_raw)
    out = np.zeros((len(row_lists), X_raw.shape[1]), dtype="float64")
    for m, rows in enumerate(row_lists):
        out[m] = X_raw[np.asarray(rows, dtype="int64")].mean(axis=0)
    return log1p_community(out)


def epoch_mean_z(Z, row_lists):
    """Mean of ``z_ema`` over each cell's rows (pure). Averages z itself, per the request.

    NaN-aware: a cell-year outside the covariate footprint comes back as NaN from
    ``encode_points``, and one such year must not wipe out the whole epoch mean. A cell with no
    finite year at all yields NaN and is caught by the caller's finite filter.
    """
    Z = np.asarray(Z, dtype="float64")
    out = np.full((len(row_lists), Z.shape[1]), np.nan, dtype="float64")
    for m, rows in enumerate(row_lists):
        block = Z[np.asarray(rows, dtype="int64")]
        if np.isfinite(block).any():
            with np.errstate(invalid="ignore"):
                out[m] = np.nanmean(block, axis=0)
    return out.astype("float32")


def knn_neighbours(xy, k=99):
    """``k`` nearest OTHER cells for each row of ``xy`` (pure) → ``(idx (N,k), dist (N,k))``.

    ``xy`` is in metres (Albers), so distances come out in metres. Queries ``k+1`` and drops the
    self hit rather than assuming it is column 0 -- with exactly coincident points (two cells at
    the same centre should be impossible, but a duplicated row would do it) the self is not
    guaranteed to sort first, and silently keeping it would inject a similarity-1 pair.

    Returns ``k_eff = min(k, N-1)`` columns; fewer cells than requested is not an error.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, dtype="float64")
    n = xy.shape[0]
    k_eff = int(min(k, max(n - 1, 0)))
    if k_eff == 0:
        return np.zeros((n, 0), dtype="int64"), np.zeros((n, 0), dtype="float64")
    dist, idx = cKDTree(xy).query(xy, k=k_eff + 1)
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    out_i = np.empty((n, k_eff), dtype="int64")
    out_d = np.empty((n, k_eff), dtype="float64")
    for i in range(n):
        keep = idx[i] != i                                  # drop self wherever it landed
        take = np.where(keep)[0][:k_eff]
        if take.size < k_eff:                               # self absent (duplicate coords)
            take = np.arange(idx.shape[1])[:k_eff]
        out_i[i], out_d[i] = idx[i][take], dist[i][take]
    return out_i, out_d


def quantile_distance_bins(dist, n_bins=10):
    """Quantile edges over all finite distances (pure) → ``(edges_m, labels_km)``.

    Quantile rather than fixed-width: k-NN distances are strongly right-skewed (dense East, sparse
    West), so equal-width bins would pile most pairs into the first one or two and leave the rest
    with too few pairs to estimate anything. Equal-n bins give comparably stable estimates.
    Degenerate input (all distances equal) collapses to a single bin instead of raising.
    """
    d = np.asarray(dist, dtype="float64").ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.array([0.0, np.inf]), ["all"]
    edges = np.unique(np.quantile(d, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if edges.size < 2:
        edges = np.array([edges[0], edges[0] + 1.0])
    edges[-1] = np.nextafter(edges[-1], np.inf)             # make the last bin right-inclusive
    labels = [f"{edges[i] / 1000:.0f}-{edges[i + 1] / 1000:.0f}km" for i in range(edges.size - 1)]
    return edges, labels


def _pairwise_ruzicka_neighbours(A, B, idx, device=None, chunk=None):
    """Ružička between each focal row of ``A`` and its neighbours ``B[idx]`` (batched, GPU).

    ``A (N,S)``, ``B (N,S)``, ``idx (N,k)`` → ``(N,k)``. Computes only the focal-to-neighbour
    entries; routing this through ``ruzicka_rect`` would build a full N x N matrix and then throw
    away all but k columns per row.

    Sizing note (measured, not guessed): the gathered ``(N,k,S)`` tensor at N=3000, k=99, S=100 is
    ~119 MB in fp32 and a handful of those are live, so a single pass fits comfortably. ``chunk``
    exists only as a guard for much larger N; the real cost of this stage is ``desk_z_ema``'s
    whole-grid forwards, not this arithmetic.
    """
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N, k = idx.shape
    if k == 0:
        return np.zeros((N, 0), dtype="float64")
    if chunk is None:
        chunk = N if dev == "cpu" else max(1, min(N, int(4e8 // max(k * A.shape[1], 1))))
    ta = torch.as_tensor(np.asarray(A), dtype=torch.float32, device=dev)
    tb = torch.as_tensor(np.asarray(B), dtype=torch.float32, device=dev)
    ti = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=dev)
    out = np.empty((N, k), dtype="float64")
    with torch.no_grad():
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            a = ta[s:e].unsqueeze(1)                        # (n,1,S)
            b = tb[ti[s:e]]                                 # (n,k,S)
            mn = torch.minimum(a, b).sum(-1)
            mx = torch.maximum(a, b).sum(-1)
            r = torch.where(mx > 0, mn / mx, torch.ones_like(mx))
            out[s:e] = r.double().cpu().numpy()
    return out


def _pairwise_dot_neighbours(A, B, idx, device=None, chunk=None):
    """``A[i] . B[idx[i,j]]`` for each focal/neighbour pair (batched, GPU) → ``(N,k)``.

    The dot product, per the kernel contract -- same reasoning as ``dot_gram``.
    """
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N, k = idx.shape
    if k == 0:
        return np.zeros((N, 0), dtype="float64")
    if chunk is None:
        chunk = N if dev == "cpu" else max(1, min(N, int(4e8 // max(k * A.shape[1], 1))))
    ta = torch.as_tensor(np.asarray(A), dtype=torch.float32, device=dev)
    tb = torch.as_tensor(np.asarray(B), dtype=torch.float32, device=dev)
    ti = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=dev)
    out = np.empty((N, k), dtype="float64")
    with torch.no_grad():
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            out[s:e] = (ta[s:e].unsqueeze(1) * tb[ti[s:e]]).sum(-1).double().cpu().numpy()
    return out


def _rowwise_ruzicka(A, B):
    """Ružička between matching rows of ``A`` and ``B`` (pure) → ``(N,)``. For self-change."""
    A, B = np.asarray(A, "float64"), np.asarray(B, "float64")
    mn, mx = np.minimum(A, B).sum(1), np.maximum(A, B).sum(1)
    return np.where(mx > 0, mn / mx, 1.0)


def bootstrap_skill_ci(obs, pred, null, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI on ``1 - rmse(pred,obs)/rmse(null,obs)``, resampling ELEMENTS. Pure.

    Call this on per-FOCAL-CELL vectors (``self_change``), never on pair matrices. Pairs drawn
    from N sampled rows number ~N^2/2 but are massively correlated, so an interval computed over
    pairs would be far too narrow -- the effective sample size is the number of independent
    cells, not the number of pairs. Reporting ``n_pairs`` without an interval is how a difference
    of a few thousandths came to be read as a finding.

    Returns ``{"skill", "lo", "hi", "n"}``; the point estimate is the unresampled value.
    """
    obs = np.asarray(obs, "float64"); pred = np.asarray(pred, "float64")
    null = np.asarray(null, "float64")
    fin = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(null)
    obs, pred, null = obs[fin], pred[fin], null[fin]
    n = obs.size

    def skill(o, p, q):
        rn = np.sqrt(np.mean((q - o) ** 2))
        return float(1.0 - np.sqrt(np.mean((p - o) ** 2)) / rn) if rn > 0 else float("nan")

    if n < 8:
        return {"skill": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(n)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(int(n_boot), n))
    vals = np.array([skill(obs[d], pred[d], null[d]) for d in draws])
    vals = vals[np.isfinite(vals)]
    if vals.size < 8:
        return {"skill": skill(obs, pred, null), "lo": float("nan"), "hi": float("nan"),
                "n": int(n)}
    return {"skill": skill(obs, pred, null),
            "lo": float(np.quantile(vals, alpha / 2.0)),
            "hi": float(np.quantile(vals, 1.0 - alpha / 2.0)), "n": int(n)}


def epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=99, n_bins=10, is_heldout=None,
                                device=None, sc_esk=None, sc_idw=(None, None)):
    """Focal-to-neighbour comparisons across space AND time, resolved by distance (pure-ish).

    Four comparison types per focal/neighbour pair, all graded against the SAME frozen-modern null
    (``Zm_i . Zm_j``) so differences isolate what varies:

    ==================  ===========================  ====================
    type                observed                     DESK
    ==================  ===========================  ====================
    ``spatial_early``   Ruzicka(Xe_i, Xe_j)          Ze_i . Ze_j
    ``spatial_modern``  Ruzicka(Xm_i, Xm_j)          Zm_i . Zm_j
    ``cross_time``      Ruzicka(Xe_i, Xm_j)          Ze_i . Zm_j
    ``self_change``     Ruzicka(Xe_i, Xm_i)  (d=0)   Ze_i . Zm_i
    ==================  ===========================  ====================

    ``spatial_modern`` is IDENTICAL to its null by construction, so its skill must be exactly 0.
    That is deliberate: it is this analysis's built-in null test. If it ever comes back non-zero
    the harness is wrong and no other row in the table can be trusted.

    Returns ``(report, per_cell)``.
    """
    idx, dist = knn_neighbours(xy, k=k)
    edges, labels = quantile_distance_bins(dist, n_bins=n_bins)
    bin_of = np.clip(np.digitize(dist, edges) - 1, 0, len(labels) - 1) if idx.size else dist

    # --- one predictor table, every question, dot AND cosine -------------------------------------
    # Each predictor supplies its (early, modern) z; the pairings below are then generated
    # identically for all of them. That is the whole point: a predictor added here appears in every
    # question and every distance bin with no call-site edit, which is what previously had to be
    # remembered separately at fourteen places and repeatedly was not.
    Ze_i, Zm_i = sc_idw if sc_idw is not None else (None, None)
    sources = {"desk": (Ze, Zm),
               "no_change": (Zm, Zm),          # frozen modern: shares DESK's functional
               "spacetime_idw": (Ze_i, Zm_i) if Ze_i is not None else None,
               "esk_oracle": sc_esk}
    missing = {n: UNAVAILABLE_NOT_SUPPLIED for n, v in sources.items() if v is None}

    def _unit(A):
        """Row-normalised, so a dot becomes a cosine. The ANGULAR form is not a secondary
        curiosity: it is the half of an exact error decomposition whose partner is the norm
        deficit, and with ||z||^2 ~ 0.66 the dot form carries that deficit while this does not."""
        A = np.asarray(A, "float64")
        nrm = np.linalg.norm(A, axis=1, keepdims=True)
        return A / np.maximum(nrm, 1e-12)

    # (question -> (observed, {predictor -> (A, B)})) so dot and cosine differ only in _unit.
    pairings = {
        "cross_cell_same_era_early": (_pairwise_ruzicka_neighbours(Xe, Xe, idx, device=device),
                                      lambda ze, zm: (ze, ze)),
        "cross_cell_same_era_modern": (_pairwise_ruzicka_neighbours(Xm, Xm, idx, device=device),
                                       lambda ze, zm: (zm, zm)),
        "cross_cell_cross_time": (_pairwise_ruzicka_neighbours(Xe, Xm, idx, device=device),
                                  lambda ze, zm: (ze, zm)),
    }
    types, types_cos, types_obs = {}, {}, {}
    for qname, (obs_q, sel) in pairings.items():
        types_obs[qname] = obs_q
        types[qname], types_cos[qname] = {}, {}
        for pname, zs in sources.items():
            if zs is None:
                continue
            a, b = sel(*zs)
            types[qname][pname] = _pairwise_dot_neighbours(a, b, idx, device=device)
            types_cos[qname][pname] = _pairwise_dot_neighbours(_unit(a), _unit(b), idx,
                                                               device=device)

    n = Xe.shape[0]
    ho = np.zeros(n, bool) if is_heldout is None else np.asarray(is_heldout, bool)
    splits = {"pooled": np.ones(n, bool)}
    if is_heldout is not None:
        splits.update({"train": ~ho, "heldout": ho})

    report = {
        "config": {"k": int(idx.shape[1]), "n_bins": len(labels), "n_focal_cells": int(n),
                   "distance_bin_edges_km": [float(e / 1000.0) for e in edges],
                   "distance_bin_labels": labels},
        "distance_summary": {
            "min_km": float(dist.min() / 1000) if idx.size else None,
            "median_km": float(np.median(dist) / 1000) if idx.size else None,
            "max_km": float(dist.max() / 1000) if idx.size else None},
        "types": {},
    }

    for qname in pairings:
        report["types"][qname] = {}
        obs_q = types_obs[qname]
        for sname, smask in splits.items():
            rows = np.where(smask)[0]

            def _row(sub):
                """One row: every predictor, dot AND cosine, plus the reasons for any that could
                not run. `sub` indexes within `rows`, or None for all of them."""
                take = (lambda M: M[rows] if sub is None else M[rows][sub])
                dot = compare_predictors(take(obs_q),
                                         {p: take(M) for p, M in types[qname].items()})
                cos = compare_predictors(take(obs_q),
                                         {p: take(M) for p, M in types_cos[qname].items()})
                dot["unavailable"].update(missing)
                cos["unavailable"].update(missing)
                # DESK against the BAR as well as against the null. Both are just a named
                # reference over the same rows, so neither is privileged.
                if "spacetime_idw" in dot["predictors"]:
                    dot["skill_vs_spacetime_idw"] = compare_predictors(
                        take(obs_q), {p: take(M) for p, M in types[qname].items()},
                        reference="spacetime_idw")["skill_vs"]
                return {"dot": dot, "cosine": cos}

            per_bin = {"all_distances": _row(None)}
            for b, lab in enumerate(labels):
                sel_b = bin_of[rows] == b
                per_bin[lab] = ({"n_pairs": int(sel_b.sum()), "skipped": "fewer than 10 pairs"}
                                if sel_b.sum() < 10 else _row(sel_b))
            report["types"][qname][sname] = per_bin

    # same_cell_over_time: one value per focal cell, no distance axis (d=0 by definition).
    # This was three hand-written blocks -- self_change, self_change_vs_idw and
    # self_change_esk_oracle -- each pushing a different predictor through a two-slot signature,
    # one of them annotated "null slot := the bar". Now it is the same predictor table as every
    # other question, so the bar and the oracle are rows rather than special cases.
    sc_obs = _rowwise_ruzicka(Xe, Xm)
    sc = {}
    for pname, zs in sources.items():
        if zs is None:
            continue
        a, b = zs
        sc[pname] = (np.asarray(a, "float64") * np.asarray(b, "float64")).sum(1)
    sc_cos = {}
    for pname, zs in sources.items():
        if zs is None:
            continue
        a, b = zs
        sc_cos[pname] = (_unit(a) * _unit(b)).sum(1)

    report["types"]["same_cell_over_time"] = {}
    for sname, m in splits.items():
        w = np.where(m)[0]
        dot = compare_predictors(sc_obs[w], {p: v[w] for p, v in sc.items()})
        cos = compare_predictors(sc_obs[w], {p: v[w] for p, v in sc_cos.items()})
        dot["unavailable"].update(missing)
        cos["unavailable"].update(missing)
        if "spacetime_idw" in dot["predictors"]:
            dot["skill_vs_spacetime_idw"] = compare_predictors(
                sc_obs[w], {p: v[w] for p, v in sc.items()},
                reference="spacetime_idw")["skill_vs"]
        # An interval, resampling FOCAL CELLS. This is the one place in the module where a
        # bootstrap is honest: one value per cell over ~1,950 independent cells. The pair
        # matrices elsewhere have an effective n far below their n_pairs.
        dot["ci95"] = {p: bootstrap_skill_ci(sc_obs[w], v[w], sc["no_change"][w])
                       for p, v in sc.items() if p != "no_change"}
        report["types"]["same_cell_over_time"][sname] = {"all_distances": {"dot": dot,
                                                                           "cosine": cos}}

    # Per-focal-cell arrays, so the result can be mapped without recomputing.
    per_cell = {"neighbour_idx": idx.astype("int32"), "neighbour_dist_m": dist.astype("float32"),
                "same_cell_over_time_obs": sc_obs.astype("float32"), "is_heldout": ho}
    # One field per PREDICTOR, not just DESK: mapping where the bar beats the model is as useful
    # as mapping where the model beats the null, and it costs nothing now that they are symmetric.
    for pname, v in sc.items():
        per_cell[f"same_cell_over_time_{pname}"] = v.astype("float32")
    for qname in pairings:
        obs_q = types_obs[qname]
        rn = (np.sqrt(((types[qname]["no_change"] - obs_q) ** 2).mean(1)) if idx.size
              else np.full(n, np.nan))
        for pname, M in types[qname].items():
            rp = np.sqrt(((M - obs_q) ** 2).mean(1)) if idx.size else np.full(n, np.nan)
            per_cell[f"{qname}_rmse_{pname}"] = rp.astype("float32")
            with np.errstate(divide="ignore", invalid="ignore"):
                per_cell[f"{qname}_skill_{pname}"] = np.where(
                    rn > 0, 1.0 - rp / rn, 0.0).astype("float32")

    report["manifest"] = {
        # The registry questions this analysis OWNS. Declared, not derived from `types` -- a scope
        # read back off the output would make `assert_complete` compare the report to itself.
        # `absolute_position` is deliberately absent: it is `zspace_reconstruction`'s question.
        "covers": ["same_cell_over_time", "cross_cell_same_era", "cross_cell_cross_time"],
        "questions": {q: QUESTIONS.get(canonical_question(q), {}) for q in report["types"]},
        "predictors": {p: PREDICTOR_ROLES.get(p, "") for p in sources if sources[p] is not None},
        "unavailable": missing,
        "quantities": {"dot": "z.z' against Ruzicka -- the contract's own quantity",
                       "cosine": "the angular half; free of the ~34% self-similarity deficit"},
        "populations": {s: int(m.sum()) for s, m in splits.items()},
    }
    return report, per_cell


# ----------------------------- IO / driver -----------------------------

def assert_same_layout(trained, species, source="points_meta.json"):
    """Raise unless two species layouts agree POSITION BY POSITION. Pure.

    Order, not membership. The basis and this module once held identical SETS of 96 species, of
    identical length, while 94 of 96 POSITIONS held a different species -- so `matched 96/96, 0
    unmatched` printed, every count and coverage check passed, and Ruzicka silently compared one
    species' abundance to another's. No test of counts or sets can see a permutation; only a
    positional comparison can, which is why this exists as its own function with its own test.
    """
    a = [str(x).lower() for x in trained]
    b = [str(x).lower() for x in species]
    if a == b:
        return
    same_set = set(a) == set(b)
    n_bad = sum(1 for x, y in zip(a, b) if x != y)
    first = next(((i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    raise ValueError(
        f"species LAYOUT disagrees with {source}: {n_bad} of {min(len(a), len(b))} positions hold "
        f"a different species"
        + (f" (lengths {len(a)} vs {len(b)})" if len(a) != len(b) else "")
        + (" (same set, so it is a pure permutation)" if same_set else " (and the sets differ)")
        + (f"; first at column {first[0]}: trained={first[1]!r} validation={first[2]!r}"
           if first else "")
        + ". Ruzicka would compare one species' abundance to another's. Both sides must order "
          "columns by bbs_community_points.species_order(community_trend.csv).")


def load_observed(config):
    """Build the observed route-level community from RAW BBS → ``(X_log, keys, meta, X_raw)``.

    The community is ``community_trend_list`` (``community_trend.csv``) -- the same list
    ``trend_community.build_trend_points`` uses -- so truth is defined over exactly the community
    DESK trains on. Nothing here is reimplemented: the route QC filter, AOU->species_code
    crosswalk, route->cell mapping and per-cell-year averaging are all the existing functions,
    with the community passed in via ``build_crosswalk(community_codes=...)``.

    Reads no precomputed community artifact: it goes straight from the raw BBS release, so there
    is no separate build step and no stale intermediate to fall out of sync with the trend
    community.
    """
    from src.config_utils import load_data_config
    from src.data.identify.bbs_crosswalk import build_crosswalk
    from src.data.preprocess import bbs
    from src.data.preprocess.bbs_community import build_community_matrix, route_grid_map
    import pandas as pd

    dcfg = load_data_config()
    tc = config.get("trend", {}) or {}
    community_csv = tc.get("community_trend_list") or dcfg["community_trend_list"]
    codes = [str(c) for c in pd.read_csv(community_csv)["species_code"].tolist()]
    if not codes:
        raise ValueError(f"no species_code rows in {community_csv}")

    dr = dcfg["datasets_root"]
    bbs_species = config.get("bbs", {}).get("species_list") or \
        os.path.join(bbs.BBS_PARENT_DIR, "SpeciesList.csv")
    crosswalk, _ = build_crosswalk(
        bbs_species, os.path.join(dr, "avonet", "eBird_taxonomy.csv"),
        os.path.join(dr, "avonet", "reference_community_ranked.csv"),
        community_codes=codes)                      # <- DESK's community, not the weekly stack
    # THE COLUMN LAYOUT IS PINNED TO community_trend.csv, via the SAME function the basis uses.
    #
    # This used to be `list(dict.fromkeys(crosswalk["species_code"]))` -- the order species happen
    # to appear in the crosswalk, which is TAXONOMIC (mutswa, wooduc, motduc, bargol, ...). The
    # basis builds its layout from `species_order(community_csv)`, which is the CSV's own
    # rank-ordering (casfin, houspa, allhum, ...). Both had 96 species and the same SET, so every
    # count and coverage check passed, while 94 of 96 COLUMNS held a different species. Ruzicka
    # then compared one species' abundance to another's: measured best-landmark similarity 0.17
    # against 0.65 for like-against-like, and ||z_obs||^2 = 0.15 against a contract of 1.0.
    #
    # Calling species_order here rather than re-deriving the order is the point: one function
    # owns the layout, so the two sides cannot drift apart again.
    from .bbs_community_points import species_order
    species = species_order(community_csv)
    matched = {str(c).lower() for c in crosswalk["species_code"]}
    n_matched = sum(1 for c in species if c in matched)
    if n_matched < 3:
        raise ValueError(
            f"only {n_matched} of {len(codes)} community_trend species crosswalked to a BBS "
            f"AOU (from {bbs_species}); cannot build a route-level community")
    # A species BBS cannot survey keeps its column, all zeros. Compacting it out would shift every
    # later column and reintroduce exactly the misalignment above -- the layout must depend on the
    # community definition alone, never on what BBS happens to match this release.

    obs_all, coverage = bbs.load_usca_observations(aou_filter=None, return_coverage=True)
    routes = bbs.load_routes()
    land_mask, _, transform, crs, nx, ny = bbs.load_grid_reference(bbs.MASK_PATH)
    route_cells = route_grid_map(routes, transform, crs, nx, ny, land_mask)
    mean_df, cov_df = build_community_matrix(obs_all, coverage, crosswalk, route_cells)

    code_ix = {c: i for i, c in enumerate(species)}
    # species_order lowercases; the crosswalk's codes may not, and a case mismatch here would
    # silently drop every row of an affected species rather than misplace it.
    sp_lower = mean_df["species_code"].astype(str).str.lower()
    mean_df = mean_df[sp_lower.isin(code_ix)]
    X_raw, keys, dropped = densify_community(
        mean_df["row"].to_numpy(), mean_df["col"].to_numpy(), mean_df["year"].to_numpy(),
        mean_df["species_code"].astype(str).str.lower().map(code_ix).to_numpy(),
        mean_df["mean_count"].to_numpy(),
        cov_df["row"].to_numpy(), cov_df["col"].to_numpy(), cov_df["year"].to_numpy(),
        len(species))
    if X_raw.shape[1] != len(species):
        raise ValueError(f"community matrix has {X_raw.shape[1]} columns for {len(species)} "
                         "species; the pinned layout was not honoured")

    # points_meta.json records the log1p flag and the species DESK trained on. Neither is in the
    # ESK meta.json. Cross-check rather than assume: a raw-count basis would make a log1p
    # similarity structure the wrong quantity, and a species set that does not match
    # community_trend.csv means this module and the trainer disagree about the community.
    # WHICH point set. `target_points_dir` prefers target.points_dir (= bbs_points, the live raw-BBS
    # target the basis is actually fitted on) and falls back to trend.points_dir (= esk_spacetime,
    # the retired trend-products set). This function read config["trend"]["points_dir"] DIRECTLY,
    # so it looked in the retired directory, found no points_meta.json there, and warned instead
    # of checking -- every run. Every other consumer (esk_kernel, desk_training,
    # validate_spacetime, pipeline_manifest, the diagnostics) goes through the helper, and
    # config_utils' own docstring warns about this exact mistake.
    #
    # The cost of that skipped check was the column permutation: 94 of 96 species columns
    # disagreed with the basis, and this is the check that compares the two species lists.
    from src.config_utils import target_points_dir
    zt = target_points_dir(config)
    pm_path = os.path.join(zt, "points_meta.json") if zt else ""
    log1p_flag, trained = True, None
    if not (pm_path and os.path.exists(pm_path)):
        # HARD FAILURE, not a warning. A warning is what let the permutation through: it printed
        # once per run into a log nobody diffs, and the grading continued on scrambled columns.
        raise FileNotFoundError(
            f"no points_meta.json at {pm_path or '<no points_dir configured>'}; the basis/BBS "
            "species-layout cross-check cannot run, and grading without it is what produced 94 "
            "of 96 misaligned columns. Point target.points_dir at the point set the basis was "
            "fitted on, or re-run the target-build stage.")
    with open(pm_path, "r", encoding="utf-8") as fh:
        pm = json.load(fh)
    log1p_flag = bool(pm.get("ruzicka_log1p", True))
    trained = [str(s) for s in (pm.get("species") or [])]
    if not log1p_flag:
        raise ValueError(
            f"{pm_path} reports ruzicka_log1p=false; the ESK basis was fit on RAW counts. "
            "Comparing a log1p similarity structure against it is not like-for-like.")
    if trained:
        assert_same_layout(trained, species, pm_path)
    print(f"[bbs-routes] layout cross-check PASSED against {pm_path}: {len(species)} species, "
          f"same order as the basis (ruzicka_log1p={log1p_flag})")

    sp = {"n_community_trend": len(codes), "n_columns": len(species),
          "n_bbs_matched": n_matched}
    if trained:
        # Both sides now derive from community_trend.csv, so a shortfall here is only species BBS
        # cannot survey -- not a definitional mismatch. A LARGE shortfall means the trained points
        # were built from a different community list and the comparison is not like-for-like.
        shared = [s for s in species if s in set(trained)]
        sp.update({"n_trained": len(trained), "n_shared_with_trained": len(shared)})
        print(f"[bbs-routes] community: {len(species)} columns pinned to community_trend.csv "
              f"order (species_order), {n_matched} observable in BBS; "
              f"{len(shared)}/{len(trained)} of the trained community observable")
        if len(shared) < 0.5 * len(species):
            print(f"[bbs-routes] WARNING: only {len(shared)}/{len(species)} BBS-matched species "
                  f"appear in {pm_path}; verify the trained points used {community_csv}")
    else:
        print(f"[bbs-routes] community: {len(species)} columns pinned to community_trend.csv "
              f"order (species_order), {n_matched} observable in BBS")

    meta = {"n_species": len(species), "species": list(species),
            "n_surveyed_cell_years": int(keys.shape[0]),
            "presence_triples_outside_coverage": int(dropped),
            "ruzicka_log1p": log1p_flag, "community_csv": community_csv, **sp,
            "year_range": [int(keys[:, 2].min()), int(keys[:, 2].max())] if keys.size else []}
    # RAW counts returned alongside the log1p version: the epoch analysis must average raw counts
    # and THEN take log1p (log1p is concave, so mean-of-log1p understates abundant species), and it
    # cannot recover the raw values from the transformed ones without an expm1 round trip.
    return log1p_community(X_raw), keys, meta, X_raw


def desk_z_ema(config, keys):
    """DESK ``z_ema`` for every row of ``keys`` → ``(N, latent)``.

    The EMA is a CAUSAL scan, so a single (cell, year) cannot be smoothed in isolation. Encodes
    the involved cells over the contiguous span ``ema_warmup_start .. max(year)``, runs the scan
    along the year axis, then gathers the requested (cell, year) rows. Grades ``z_ema`` because
    that is the quantity the trainer supervised -- the cube deliberately exports raw z (the
    population model below it supplies demographic lag), so raw is the wrong comparand here.
    """
    from .validate_spacetime import encode_points

    out_dir = config["paths"]["desk_output_dir"]
    dm = np.load(os.path.join(out_dir, "desk_meta.npz"), allow_pickle=True)
    ema_on = bool(dm["output_ema"]) if "output_ema" in dm.files else False
    hl = float(dm["ema_half_life"]) if "ema_half_life" in dm.files else float("nan")
    warm = int(dm["ema_warmup_start"]) if "ema_warmup_start" in dm.files else 1940

    keys = np.asarray(keys)
    cells = np.unique(keys[:, :2], axis=0)
    y_max = int(keys[:, 2].max())
    years = list(range(min(warm, int(keys[:, 2].min())), y_max + 1))

    # One (cell, year) request per cell per year in the span; encode_points batches by year, so
    # this costs one whole-grid forward per year regardless of how many cells we ask for.
    grid = np.empty((len(cells) * len(years), 3), dtype="int32")
    for t, y in enumerate(years):
        s = t * len(cells)
        grid[s:s + len(cells), :2] = cells
        grid[s:s + len(cells), 2] = y
    t0 = time.perf_counter()
    Z, ok = encode_points(config, grid)
    print(f"[bbs-routes] encoded {len(cells)} cells x {len(years)} yr "
          f"({years[0]}-{years[-1]}) in {time.perf_counter() - t0:.1f}s", flush=True)

    L = Z.shape[1]
    stack = Z.reshape(len(years), len(cells), L)
    valid = ok.reshape(len(years), len(cells))
    if ema_on and np.isfinite(hl):
        stack = apply_output_ema(stack, hl, valid=valid)
        print(f"[bbs-routes] applied output-EMA (half-life={hl:.2f} yr, "
              f"alpha={1.0 - 2.0 ** (-1.0 / hl):.3f}) over {len(years)} years")
    else:
        print(f"[bbs-routes] WARNING: desk_meta has output_ema={ema_on}, ema_half_life={hl}; "
              "grading RAW z (no EMA available in this checkpoint)")

    cell_ix = {(int(r), int(c)): i for i, (r, c) in enumerate(cells)}
    year_ix = {y: t for t, y in enumerate(years)}
    Zout = np.full((keys.shape[0], L), np.nan, dtype="float32")
    for i in range(keys.shape[0]):
        Zout[i] = stack[year_ix[int(keys[i, 2])], cell_ix[(int(keys[i, 0]), int(keys[i, 1]))]]
    return Zout, {"output_ema_applied": bool(ema_on and np.isfinite(hl)),
                  "ema_half_life": hl if np.isfinite(hl) else None,
                  "ema_warmup_start": warm, "encode_years": [years[0], years[-1]]}


def build_spacetime_bar(config, keys, X_raw_all, latent_dim):
    """Spacetime-IDW stand-in for z at every row of ``keys``, or None. ``(N, latent_dim)``.

    WHY A BAR AT ALL. Every skill figure in this module is scored against ``S_nc`` -- DESK's own z
    frozen at the modern year. That null is a DECOMPOSITION device (both sides share the
    functional, so their difference isolates temporal variation), not a competitor, and it is
    nearly free to beat: measured, on ``self_change`` the null already correlates 0.53-0.63 with
    observed change while carrying ZERO temporal content, because how much a cell appears to
    change is largely a static property of the cell. An honest bar has to be a real alternative
    PREDICTOR, so this interpolates the observed z in space AND time from training rows only.

    Uses DESK's OWN functional -- a dot product of ESK-space vectors -- because scoring a bar in
    Ruzicka while DESK is scored in dot product was measured at -0.28 on a temporally-neutral
    model, from the metric mismatch alone. That needs the observed communities projected into the
    pinned basis, which is admissible now that ``run_spacetime_esk`` fits the basis on
    ``target_points_dir`` (= raw route-level BBS); see the module docstring.

    Returns None (with a printed reason) whenever the projection or the holdout mask is missing,
    so a run without a bar degrades to the old report rather than failing.
    """
    try:
        from .esk_kernel import project_points_to_z
        from .validate_baselines import spacetime_idw_baseline, spacetime_idw_z
        zd = config["desk"]["z_dir"]
        z_rows = project_points_to_z(log1p_community(X_raw_all), zd, latent_dim)
        ho_p = os.path.join(config["paths"]["desk_output_dir"], "holdout_cells.npy")
        bf_p = os.path.join(config["paths"]["desk_output_dir"], "buffer_cells.npy")
        if z_rows is None or not os.path.exists(ho_p):
            print(f"[bbs-routes] spacetime-IDW bar unavailable (no projection in {zd} "
                  f"or no holdout mask); skill will be reported against the no-change null only")
            return None
        ho = np.load(ho_p)
        bf = np.load(bf_p) if os.path.exists(bf_p) else np.zeros_like(ho)
        hy = [int(y) for y in (config["desk"].get("trend", {}).get("holdout_years") or [])]
        # Anisotropy fitted on TRAINING rows only, long-gap probe when years are withheld, so
        # the bar is tuned for the reach it is judging rather than for interpolation.
        _e, ratio = spacetime_idw_baseline(keys, z_rows, ho, np.zeros(len(keys), bool),
                                           buffer_mask=bf, exclude_years=hy, verbose=True)
        bar = spacetime_idw_z(keys, z_rows, ho, float(ratio), buffer_mask=bf, exclude_years=hy)
        print(f"[bbs-routes] spacetime-IDW bar at ratio={ratio:g} cells/yr; "
              f"{int(np.isfinite(bar).all(1).sum()):,}/{len(keys):,} rows reachable")
        return bar
    except Exception as exc:                      # a missing bar must not sink the analysis
        print(f"[bbs-routes] spacetime-IDW bar unavailable ({exc})")
        return None


def _run_epoch_analysis(config, keys, X_raw_all, cells, e_rows, m_rows, gate_stats, gather_z,
                        zmeta, out_dir, k=99, n_bins=10, idw_rows=None):
    """Epoch means -> kNN neighbourhoods -> distance-resolved comparisons. Writes json + npz."""
    if cells.shape[0] < 5:
        print(f"[bbs-routes] epoch analysis SKIPPED: only {cells.shape[0]} cells passed the gate")
        return None

    Xe = epoch_mean_observed(X_raw_all, e_rows)               # mean raw counts THEN log1p
    Xm = epoch_mean_observed(X_raw_all, m_rows)
    Ze = epoch_mean_z(gather_z(keys[np.array([i for r in e_rows for i in r])]),
                      _reindex_row_lists(e_rows))
    Zm = epoch_mean_z(gather_z(keys[np.array([i for r in m_rows for i in r])]),
                      _reindex_row_lists(m_rows))

    # --- the spacetime-IDW BAR (shared builder; see build_spacetime_bar) -------------------
    # Why this exists: every skill figure in this module is scored against S_nc, DESK's own z
    # frozen at the modern year. That null is a DECOMPOSITION device (it isolates temporal
    # variation because both sides share the functional), not a competitor -- and it is nearly
    # free to beat. Measured: on self_change the null already correlates 0.53-0.63 with observed
    # change while carrying ZERO temporal content, because how much a cell appears to change is
    # largely a static property of the cell. An honest bar has to be an actual alternative
    # PREDICTOR, so: interpolate the observed z in space AND time from training rows only.
    #
    # It must use DESK's functional (a dot product of ESK-space vectors). A Ruzicka-based bar
    # would be scored in truth's own metric while DESK is not, and that mismatch alone was
    # measured at -0.28 on a temporally-neutral model.
    Z_idw_rows = idw_rows if idw_rows is not None else build_spacetime_bar(
        config, keys, X_raw_all, Ze.shape[1])

    Ze_idw = Zm_idw = None
    if Z_idw_rows is not None:
        # Averaged over the SAME epoch rows as DESK, through the same epoch_mean_z, so a
        # difference between model and bar is never an averaging artifact.
        Ze_idw = epoch_mean_z(Z_idw_rows[np.array([i for r in e_rows for i in r])],
                              _reindex_row_lists(e_rows))
        Zm_idw = epoch_mean_z(Z_idw_rows[np.array([i for r in m_rows for i in r])],
                              _reindex_row_lists(m_rows))

    # The ceiling: project the OBSERVED epoch communities through the same pinned ESK basis.
    # Now admissible on raw BBS -- run_spacetime_esk fits the basis on target_points_dir
    # (= bbs_points), so the landmarks are already at raw-BBS magnitude and there is no
    # BBS->eBird scale factor to invent. project_points_to_z is the single source of truth for
    # z_obs, shared with the trainer and with validate_spacetime.
    # SELF-GATED. This oracle was published once and withdrawn, because it fed the basis 20-year
    # averaged communities whose projections sit at ||z||^2 = 0.15 against 0.672 for single
    # cell-years -- the Ruzicka feature map is nonlinear, so phi(mean x) lies off the span of
    # {phi(x_i)} even though mean phi(x_i) does not. It therefore measured that mismatch and not a
    # ceiling. A diagnostic whose reading assumes the kernel contract has to REFUSE when the
    # contract does not hold on its own inputs, so the representability check is now a gate rather
    # than a printed aside.
    sc_esk = None
    try:
        from .esk_kernel import project_points_to_z
        _zd = config["desk"]["z_dir"]
        Ze_o = project_points_to_z(Xe, _zd, Ze.shape[1])
        Zm_o = project_points_to_z(Xm, _zd, Zm.shape[1])
        if Ze_o is None or Zm_o is None:
            print("[bbs-routes] ESK oracle unavailable: no saved projection in "
                  f"{_zd}; the self_change ceiling cannot be measured this run")
        else:
            n_avg = float(np.median(np.concatenate([(Ze_o ** 2).sum(1), (Zm_o ** 2).sum(1)])))
            # The reference: ANNUAL communities, which are what the basis was fitted on.
            _rs = np.random.default_rng(0).permutation(len(X_raw_all))[:4000]
            z_ann = project_points_to_z(log1p_community(X_raw_all[_rs]), _zd, Ze.shape[1])
            n_ann = float(np.median((z_ann ** 2).sum(1))) if z_ann is not None else float("nan")
            tol = float(config.get("bbs_routes", {}).get("oracle_norm_tol", 0.5))
            ok = np.isfinite(n_ann) and n_ann > 0 and (n_avg / n_ann) >= tol
            if ok:
                sc_esk = (Ze_o, Zm_o)
                print(f"[bbs-routes] ESK oracle ENABLED: averaged-community ||z_obs||^2 = "
                      f"{n_avg:.4f} against {n_ann:.4f} annual (ratio {n_avg / n_ann:.2f} "
                      f">= tol {tol:g}); the basis spans the objects the oracle projects")
            else:
                print(f"[bbs-routes] ESK oracle REFUSED: ceiling not measurable -- the basis does "
                      f"not span averaged communities (||z_obs||^2 = {n_avg:.4f} against "
                      f"{n_ann:.4f} annual, ratio {n_avg / max(n_ann, 1e-9):.2f} < tol {tol:g}). "
                      f"Any number computed here would measure that mismatch, not a ceiling. "
                      f"Widen the ESK landmark support to cover averaged communities first.")
    except Exception as exc:                      # diagnostic only; never block the analysis
        print(f"[bbs-routes] ESK oracle unavailable ({exc})")

    finite = np.isfinite(Ze).all(1) & np.isfinite(Zm).all(1)
    if sc_esk is not None:
        finite &= np.isfinite(sc_esk[0]).all(1) & np.isfinite(sc_esk[1]).all(1)
    if not finite.all():
        print(f"[bbs-routes] epoch: dropping {int((~finite).sum())} cells with non-finite z")
        cells, Xe, Xm, Ze, Zm = cells[finite], Xe[finite], Xm[finite], Ze[finite], Zm[finite]
        if sc_esk is not None:
            sc_esk = (sc_esk[0][finite], sc_esk[1][finite])
        if Ze_idw is not None:
            Ze_idw, Zm_idw = Ze_idw[finite], Zm_idw[finite]
    if cells.shape[0] < 5:
        print("[bbs-routes] epoch analysis SKIPPED: too few cells after the finite-z filter")
        return None

    # cell_xy reopens the ref raster on every call, so call it ONCE for all cells.
    from src.config_utils import load_data_config
    from .validate_spacetime import cell_xy
    xy = cell_xy(cells[:, 0], cells[:, 1], load_data_config()["grid"]["ref_raster"])

    ho_path = os.path.join(config["paths"]["desk_output_dir"], "holdout_cells.npy")
    is_ho = np.load(ho_path)[cells[:, 0], cells[:, 1]] if os.path.exists(ho_path) else None

    t0 = time.perf_counter()
    rep, per_cell = epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=k, n_bins=n_bins,
                                                sc_esk=sc_esk, sc_idw=(Ze_idw, Zm_idw),
                                                is_heldout=is_ho)
    rep["gate"] = gate_stats
    rep["desk"] = zmeta
    rep["epochs"] = {"early": list(EPOCH_EARLY), "modern": list(EPOCH_MODERN),
                     "min_distinct_years": MIN_EPOCH_YEARS,
                     "desk_averaging": "matched (only each cell's surveyed years)",
                     "observed_averaging": "mean RAW counts then log1p (log1p is concave)"}

    # all_years variant: average DESK over EVERY year of each epoch rather than only the surveyed
    # ones. Better-estimated but NOT matched to what the observed mean summarizes, so if BBS
    # sampling clusters inside a window the two diverge -- and that divergence measures the
    # sampling bias. Reported at all_distances only; the matched variant stays primary.
    try:
        Ze_all, Zm_all = _epoch_mean_z_all_years(cells, gather_z)
        fin2 = np.isfinite(Ze_all).all(1) & np.isfinite(Zm_all).all(1)
        if fin2.sum() >= 5:
            rep_all, _ = epoch_neighborhood_analysis(
                Xe[fin2], Xm[fin2], Ze_all[fin2], Zm_all[fin2], xy[fin2], k=k, n_bins=1,
                is_heldout=None if is_ho is None else is_ho[fin2])
            rep["variant_all_years"] = {
                t: v["pooled"]["all_distances"] for t, v in rep_all["types"].items()}
            rep["variant_all_years"]["_divergence"] = {
                "mean_abs_diff_Ze": float(np.nanmean(np.abs(Ze_all[fin2] - Ze[fin2]))),
                "mean_abs_diff_Zm": float(np.nanmean(np.abs(Zm_all[fin2] - Zm[fin2]))),
                "note": "large divergence => BBS sampling years are not representative of the "
                        "epoch, so the matched variant is the one to trust"}
    except Exception as exc:                                  # diagnostic only, never fatal
        print(f"[bbs-routes] all_years variant skipped: {type(exc).__name__}: {exc}")
    print(f"[bbs-routes] epoch neighbourhood analysis: {cells.shape[0]} cells x "
          f"{rep['config']['k']} neighbours in {time.perf_counter() - t0:.1f}s")

    with open(os.path.join(out_dir, "bbs_epoch_neighborhood.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    np.savez_compressed(os.path.join(out_dir, "bbs_epoch_neighborhood.npz"),
                        cells=cells, rows=cells[:, 0], cols=cells[:, 1],
                        Xe=Xe.astype("float32"), Xm=Xm.astype("float32"),
                        Ze=Ze, Zm=Zm, **per_cell)

    # --- console summary
    d = rep["distance_summary"]
    print(f"\nEPOCH x NEIGHBOURHOOD -- {cells.shape[0]} focal cells, k={rep['config']['k']}, "
          f"neighbour distance {d['min_km']:.0f}/{d['median_km']:.0f}/{d['max_km']:.0f} km "
          f"(min/median/max)")
    # One column per PREDICTOR, generated from the row itself, so a predictor added to the
    # registry appears in the printed table too rather than only in the JSON.
    for tname, per_split in rep["types"].items():
        print(f"\n  {tname}  -- {QUESTIONS.get(tname, {}).get('why', '')[:96]}")
        for sname, per_bin in per_split.items():
            for bname, mm in per_bin.items():
                if "skipped" in mm:
                    continue
                for form in ("dot", "cosine"):
                    blk = mm.get(form)
                    if not blk or not blk.get("predictors"):
                        continue
                    names = sorted(blk["predictors"])
                    if bname == "all_distances" and form == "dot":
                        print(f"    {'split/form':<18}{'n':>8}" +
                              "".join(f"{p[:12]:>13}" for p in names))
                    print(f"    {sname + '/' + form:<18}{blk['n']:>8}" +
                          "".join(f"{blk['skill_vs'][p]:>+13.3f}" for p in names)
                          + ("   [skill vs no_change]" if form == "dot" else ""))
                    break
        for reason_p, reason in (mm.get("dot", {}) or {}).get("unavailable", {}).items():
            print(f"      unavailable: {reason_p} -- {reason}")
    sm = rep["types"]["cross_cell_same_era_modern"]["pooled"]["all_distances"]["dot"]["skill_vs"]["desk"]
    print(f"\n  [null test] cross_cell_same_era_modern skill = {sm:+.2e} -- MUST be ~0: DESK and "
          "the null are the same quantity there by construction. Non-zero means the harness is "
          "broken.")
    gaps = assert_complete(rep["types"])
    print(f"  [completeness] {'OK -- every question x predictor has a result or a reason' if not gaps else 'GAPS: ' + '; '.join(gaps[:6])}")
    print(f"[bbs-routes] epoch report -> {os.path.join(out_dir, 'bbs_epoch_neighborhood.json')}")
    return rep


def _epoch_mean_z_all_years(cells, gather_z):
    """DESK epoch means over EVERY year of each epoch (not just surveyed ones)."""
    e_years = list(range(EPOCH_EARLY[0], EPOCH_EARLY[1] + 1))
    m_years = list(range(EPOCH_MODERN[0], EPOCH_MODERN[1] + 1))
    out = []
    for years in (e_years, m_years):
        kk = np.array([[r, c, y] for r, c in cells for y in years], dtype="int32")
        Z = gather_z(kk)
        out.append(epoch_mean_z(Z, _reindex_row_lists([years] * len(cells))))
    return out[0], out[1]


def _reindex_row_lists(row_lists):
    """Row lists renumbered against their own concatenation (pure).

    ``gather_z`` is handed the flattened rows, so the per-cell groupings have to be expressed as
    offsets into that flat result rather than into the original ``keys``.
    """
    out, n = [], 0
    for rows in row_lists:
        out.append(list(range(n, n + len(rows))))
        n += len(rows)
    return out


def run(config=None, n_sample=4000, seed=0):
    """Driver: observed vs no-change vs DESK, bucketed, written to ``desk_output_dir``."""
    config = config or load_config()
    rng = np.random.default_rng(seed)

    X_log, keys, meta, X_raw_all = load_observed(config)
    print(f"[bbs-routes] {meta['n_surveyed_cell_years']} surveyed cell-years, "
          f"{meta['n_species']} species, years {meta['year_range']}")

    # ---- epoch analysis prep, done BEFORE the pooled site gate so it sees every surveyed row.
    # Its gate (>=3 distinct years in both epochs) is stricter and different from the pooled one,
    # so the two must be derived independently from the full row set.
    #
    # KEEP THE UNFILTERED KEYS. The pooled site gate below rebinds `keys = keys[keep]`, so the row
    # indices in ep_e_rows/ep_m_rows -- which index the FULL array -- would silently point into the
    # wrong rows afterwards (and did: IndexError at 110878 vs size 110839). X_raw_all is already
    # unfiltered, so the epoch analysis must be handed keys_all to match it.
    keys_all = keys
    ep_cells, ep_e_rows, ep_m_rows, ep_stats = epoch_gate(keys_all)
    print(f"[bbs-routes] epoch gate: {ep_stats['cells_kept']}/{ep_stats['cells_seen']} cells have "
          f">={MIN_EPOCH_YEARS} distinct surveyed years in BOTH "
          f"{EPOCH_EARLY[0]}-{EPOCH_EARLY[1]} and {EPOCH_MODERN[0]}-{EPOCH_MODERN[1]} "
          f"(failed: early-only {ep_stats['cells_failed_early_only']}, "
          f"modern-only {ep_stats['cells_failed_modern_only']}, "
          f"both {ep_stats['cells_failed_both']})")

    # Site gate: drop every row of any cell lacking a modern survey (no definable null).
    nc_src, keep = modern_reference_rows(keys, MODERN_WINDOW)
    # Averaging groups stay in FULL row space, indexed the same way X_raw_all is -- see
    # window_groups. Only the pooled row bookkeeping (X_log/keys/nc_src) is gated.
    nc_groups_full, _keep_g = modern_reference_groups(keys, MODERN_WINDOW)
    assert bool((_keep_g == keep).all()), "reference gate must not depend on the averaging mode"
    # The single shared point-denoising half-width; bbs_routes.window_half_width is honoured only
    # as a legacy fallback so old configs keep working.
    win_groups_full = window_groups(keys, int(
        (config.get("target", {}) or {}).get(
            "smooth_half_width",
            config.get("bbs_routes", {}).get("window_half_width", 2))))
    kept_to_full = np.where(keep)[0]
    n_cells_all = np.unique(keys[:, :2], axis=0).shape[0]
    X_log, keys, nc_src = X_log[keep], keys[keep], nc_src[keep]
    remap = -np.ones(len(keep), dtype="int64")
    remap[np.where(keep)[0]] = np.arange(int(keep.sum()))
    nc_src = remap[nc_src]                                   # reindex into the kept rows
    n_cells_kept = np.unique(keys[:, :2], axis=0).shape[0]
    print(f"[bbs-routes] site gate: kept {int(keep.sum())}/{len(keep)} rows, "
          f"{n_cells_kept}/{n_cells_all} cells "
          f"({n_cells_all - n_cells_kept} dropped for no {MODERN_WINDOW[0]}-{MODERN_WINDOW[1]} survey)")
    if keys.shape[0] < 3:
        raise SystemExit("[bbs-routes] fewer than 3 rows survive the site gate; nothing to compare")

    # Holdout must be resolved BEFORE sampling so it can be a stratum. Held-out cells are a small
    # minority of rows, and held-out is the only split that answers whether the model generalizes,
    # so leaving it to chance makes the table's most important cells its least reliable.
    ho_path = os.path.join(config["paths"]["desk_output_dir"], "holdout_cells.npy")
    has_ho = os.path.exists(ho_path)
    if has_ho:
        ho_grid = np.load(ho_path)
        is_ho_all = ho_grid[keys[:, 0], keys[:, 1]]
        print(f"[bbs-routes] holdout mask: {int(is_ho_all.sum())}/{len(is_ho_all)} rows "
              f"({100.0 * is_ho_all.mean():.1f}%) are held-out cells")
    else:
        print(f"[bbs-routes] WARNING: no {ho_path}; train/heldout split unavailable, "
              "reporting pooled only (every prior metric was pooled at ~83% training points)")
        is_ho_all = None

    _reg = int(config.get("bbs_routes", {}).get("spatial_regions", 0))
    sel = stratified_sample(keys, int(n_sample), rng, is_heldout=is_ho_all,
                            spatial_regions=_reg)
    if _reg:
        print(f"[bbs-routes] row sampling stratified over year-window x holdout x {_reg}x{_reg} "
              f"spatial regions, so a coast-heavy metric cannot hide an interior deficit")
    X_s, keys_s, nc_s = X_log[sel], keys[sel], nc_src[sel]
    keys_nc = keys[nc_s]                                     # each row's MODERN (cell, year)
    sel_full = kept_to_full[sel]                             # back into X_raw_all's row space

    # DENOISED endpoints. A cell-year is ~1.08 routes, i.e. one observer on one morning, so a
    # single-year community is a noisy realization of the decadal quantity we actually care
    # about. Averaging borrows power over time at a FIXED location, which is defensible in a way
    # spatial smoothing would not be (it borrows none across space).
    #
    # Averaging happens on RAW COUNTS, then log1p once -- via epoch_mean_observed, the same
    # helper and the same order the epoch analysis already uses. That order is not a preference:
    # BBS counts are Poisson (a count with mean B has log-variance ~1/B), so the mean of raw
    # counts is the minimum-variance unbiased estimator of the rate, while mean-of-log1p is
    # biased for it and understates abundant species. Because the transform is still applied
    # exactly once at the end, S_true remains Ruzicka of a log1p community -- the same functional
    # on the same quantity, from a better estimate of it. The kernel contract is untouched and
    # truth never passes through a fitted projection, so this module's whole reason for existing
    # (escaping the target's own construction operator) is preserved.
    avg = bool(config.get("bbs_routes", {}).get("average_windows", True))
    hw = int((config.get("target", {}) or {}).get(
        "smooth_half_width", config.get("bbs_routes", {}).get("window_half_width", 2)))
    grp_nc = [nc_groups_full[i] for i in sel_full]            # modern reference group per row
    grp_win = [win_groups_full[i] for i in sel_full]          # +/-hw window per row
    if avg:
        X_s = epoch_mean_observed(X_raw_all, grp_win)         # mean raw counts THEN log1p
        X_nc_s = epoch_mean_observed(X_raw_all, grp_nc)
        d_win = float(np.mean([len(g) for g in grp_win])) if grp_win else float("nan")
        d_nc = float(np.mean([len(g) for g in grp_nc])) if grp_nc else float("nan")
        # Bucket membership keys on the row's OWN year, so a window near a bucket edge reaches
        # across it. Reported rather than clamped: clamping would make averaging depth depend on
        # the bucket, which would reintroduce exactly the era-dependent smoothing this removes.
        yv = keys_s[:, 2]
        straddle = int(np.sum([
            (np.min(keys_all[list(g)][:, 2]) < EARLY_WINDOW[1] < np.max(keys_all[list(g)][:, 2]))
            or (np.min(keys_all[list(g)][:, 2]) < MODERN_WINDOW[0] <= np.max(keys_all[list(g)][:, 2]))
            for g in grp_win])) if len(yv) else 0
        print(f"[bbs-routes] DENOISED endpoints (mean raw counts then log1p): rows averaged over "
              f"+/-{hw} yr, mean {d_win:.1f} surveyed years each; modern reference averaged over "
              f"{MODERN_WINDOW[0]}-{MODERN_WINDOW[1]}, mean {d_nc:.1f} years "
              f"(single-year was 1.0 for both)")
        print(f"[bbs-routes] {straddle}/{len(yv)} sampled windows reach across an early/modern "
              f"bucket edge; buckets key on the row's own (centre) year")
    else:
        X_nc_s = X_log[nc_s]
        d_win = d_nc = 1.0
        print(f"[bbs-routes] single-year endpoints, modern reference = most recent survey in "
              f"{MODERN_WINDOW[0]}-{MODERN_WINDOW[1]} (historical behaviour)")
    is_ho = is_ho_all[sel] if is_ho_all is not None else np.zeros(len(sel), bool)
    print(f"[bbs-routes] sampled {len(sel)} rows ({int(is_ho.sum())} held-out) "
          f"for the {len(sel)}x{len(sel)} matrices")

    # ONE encode pass for BOTH analyses. encode_points batches by year, so desk_z_ema costs one
    # whole-grid forward PER YEAR regardless of how many cells are requested -- calling it once per
    # analysis would double the ~86 forwards for no benefit. So request the union and slice.
    # Sampled rows' modern references are usually not themselves in the sample, so they must be
    # asked for explicitly.
    ep_rows_flat = np.array([i for rows in (ep_e_rows + ep_m_rows) for i in rows], dtype="int64")
    need = [keys_s, keys_nc]
    if avg:
        # Every row the two averages touch, in FULL row space. Free in forwards: desk_z_ema
        # encodes every requested cell across every year of the span, so extra keys add gathers
        # only (same reasoning as the epoch rows below).
        avg_rows_full = np.unique(np.concatenate(
            [np.asarray(g, dtype="int64") for g in (grp_win + grp_nc) if len(g)]))
        need.append(keys_all[avg_rows_full])
    if ep_rows_flat.size:
        need.append(keys_all[ep_rows_flat])          # keys_all: epoch rows index the FULL array
        # The all_years DESK variant needs z at EVERY year of each epoch, not just surveyed ones.
        # Free: desk_z_ema already encodes every requested cell across every year of the span, so
        # these extra keys add gathers, not forwards.
        ep_all = np.array([[r, c, y] for r, c in ep_cells
                           for y in list(range(EPOCH_EARLY[0], EPOCH_EARLY[1] + 1))
                           + list(range(EPOCH_MODERN[0], EPOCH_MODERN[1] + 1))], dtype="int32")
        need.append(ep_all)
    want = np.unique(np.vstack(need), axis=0)
    Z_want, zmeta = desk_z_ema(config, want)
    want_ix = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(want)}

    def _gather(kk):
        return Z_want[[want_ix[(int(r), int(c), int(y))] for r, c, y in np.asarray(kk)]]

    # The kernel contract is a property of a SINGLE cell-year's z, so it must be checked on
    # unaveraged z: ||mean(z)||^2 < mean(||z||^2) whenever the z's differ (Jensen), so measuring
    # it on the averaged vectors would report a calibration failure that is really just averaging.
    Z_raw_rows = _gather(keys_s)
    if avg:
        # The SAME groups drive both sides -- one list, so symmetry is structural, not asserted.
        # Averaged over the OBSERVED years only, never over every year in the window: the model
        # is deterministic, so its averaging exists to estimate the same functional the
        # observation estimates ("mean over surveyed years"). Averaging all years instead would
        # shift the model's effective time centre wherever a cell's surveys sit asymmetrically in
        # the window -- a systematic offset, largest in the sparsely-surveyed early era, scored as
        # model error when it is survey timing.
        zc_rows = np.unique(np.concatenate(
            [np.asarray(g, dtype="int64") for g in (grp_win + grp_nc) if len(g)]))
        Zc = _gather(keys_all[zc_rows])
        cpos = {int(r): i for i, r in enumerate(zc_rows)}
        Z_s = epoch_mean_z(Zc, [[cpos[int(j)] for j in g] for g in grp_win])
        Z_nc = epoch_mean_z(Zc, [[cpos[int(j)] for j in g] for g in grp_nc])
    else:
        Z_s, Z_nc = Z_raw_rows, _gather(keys_nc)

    finite = np.isfinite(Z_s).all(1) & np.isfinite(Z_nc).all(1)
    if finite.sum() < 3:
        raise SystemExit("[bbs-routes] DESK returned non-finite z for nearly every row "
                         "(covariate footprint mismatch?); nothing to compare")
    if not finite.all():
        print(f"[bbs-routes] dropping {int((~finite).sum())} rows with non-finite DESK z")
        X_s, keys_s, X_nc_s = X_s[finite], keys_s[finite], X_nc_s[finite]
        Z_s, Z_nc, is_ho = Z_s[finite], Z_nc[finite], is_ho[finite]

    # ONE bar for the whole module: the pooled matrices below and the epoch analysis at the end
    # both take it, so they cannot end up scored against different alternatives.
    #
    # Built in FULL row space (keys_all), because that is the space X_raw_all lives in and the
    # space the epoch analysis indexes -- the same split the site gate creates and that has bitten
    # here before ("IndexError at 110878 vs size 110839"). The pooled path indexes down through
    # sel_full, then through `finite`, in that order, to land on Z_s's rows.
    idw_rows = build_spacetime_bar(config, keys_all, X_raw_all, Z_s.shape[1])
    Z_idw_s = None
    if idw_rows is not None:
        Z_idw_s = idw_rows[sel_full]
        if not finite.all():
            Z_idw_s = Z_idw_s[finite]
        fin_bar = np.isfinite(Z_idw_s).all(1)
        if fin_bar.sum() < max(8, int(0.2 * len(Z_idw_s))):
            print(f"[bbs-routes] spacetime-IDW bar reaches only {int(fin_bar.sum())}/"
                  f"{len(Z_idw_s)} sampled rows; too thin to score the pooled matrices against")
            Z_idw_s = None
        else:
            # Unreachable rows would poison a Gram matrix, and dropping them here would put the
            # bar on a different row set from DESK. Fill with the no-change vector instead: the
            # bar then degrades TOWARD the null exactly where it cannot reach, which is the
            # conservative direction (it can only make the bar weaker, never stronger).
            Z_idw_s = np.where(fin_bar[:, None], Z_idw_s, Z_nc)
            print(f"[bbs-routes] pooled IDW bar: {int((~fin_bar).sum())} unreachable rows filled "
                  f"with the no-change vector (conservative -- weakens the bar, never strengthens)")

    S_true = ruzicka_rect(X_s, X_s)
    S_desk, S_nc = dot_gram(Z_s), dot_gram(Z_nc)             # THE contract: dot ~= Ruzicka
    S_idw = dot_gram(Z_idw_s) if Z_idw_s is not None else None
    S_desk_cos, S_nc_cos = cosine_gram(Z_s), cosine_gram(Z_nc)   # the ANGULAR half; see docstring
    S_nc_obs = ruzicka_rect(X_nc_s, X_nc_s)                  # achievable-ceiling reference

    # THE ORACLE, as a predictor row rather than a separate scalar. It substitutes the observed
    # community's own projection for the model's z, so the gap between it and S_true is the
    # BASIS's representational limit and nothing to do with DESK. Without it a route-level skill
    # of ~0 cannot be attributed: unclear whether the model failed to learn the signal or the
    # basis cannot carry it. It was never passed in here, and the catch-all reason made that hole
    # look like a decision.
    # Degrades gracefully like the bar above, but records WHY -- a specific cause, never a
    # catch-all, since a vague reason is what hid this row's absence in the first place.
    from .esk_kernel import project_points_to_z
    Z_esk_s, oracle_why = None, None
    _zdir = (config.get("desk", {}) or {}).get("z_dir")
    if not _zdir:
        oracle_why = "desk.z_dir is not configured, so the observed community cannot be projected"
    else:
        Z_esk_s = project_points_to_z(np.asarray(X_s, "float32"), _zdir, Z_s.shape[1])
        if Z_esk_s is None:
            oracle_why = f"no saved ESK projection in {_zdir}"
    S_esk = dot_gram(Z_esk_s) if Z_esk_s is not None else None
    S_esk_cos = cosine_gram(Z_esk_s) if Z_esk_s is not None else None
    if Z_esk_s is not None:
        print(f"[bbs-routes] oracle wired into the route buckets: median ||z_obs||^2 = "
              f"{float(np.median((Z_esk_s ** 2).sum(1))):.4f} (contract 1.0); the gap from "
              "S_true is the BASIS limit, not DESK's")
    else:
        print(f"[bbs-routes] oracle NOT in the route buckets: {oracle_why}")

    # Is the kernel contract even holding on this data? ||z||^2 should be ~1 and the dot should
    # land on the same [0,1] scale as Ruzicka. A large gap here means the headline RMSE is
    # dominated by calibration, and the cosine columns are the ones to read.
    z2 = float(np.median((Z_raw_rows[finite] ** 2).sum(1)))
    print(f"[bbs-routes] contract check (UNAVERAGED z: averaging shrinks ||z||^2 by Jensen, so "
          f"the contract cannot be read off the denoised vectors): median ||z||^2 = {z2:.4f} "
          f"(contract 1.0); "
          f"median dot = {np.median(_offdiag(S_desk)):.4f} vs observed Ruzicka "
          f"{np.median(_offdiag(S_true)):.4f}")

    # Underpowered-comparison guard: if observed communities are all near-identical there is no
    # turnover to predict and every CKA will be ~1 regardless of model quality.
    iu = np.triu_indices_from(S_true, k=1)
    med_off = float(np.median(S_true[iu])) if iu[0].size else float("nan")
    print(f"[bbs-routes] median off-diagonal observed similarity {med_off:.4f} "
          f"(near 1.0 => underpowered)")

    yrs = keys_s[:, 2]
    windows = {"all": np.ones(len(yrs), bool),
               "modern": (yrs >= MODERN_WINDOW[0]) & (yrs <= MODERN_WINDOW[1]),
               "early": (yrs >= EARLY_WINDOW[0]) & (yrs <= EARLY_WINDOW[1])}
    splits = {"pooled": np.ones(len(yrs), bool)}
    if has_ho:
        splits.update({"train": ~is_ho, "heldout": is_ho})

    report = {"config": {"n_sample": int(n_sample), "seed": int(seed),
                         "modern_window": list(MODERN_WINDOW), "early_window": list(EARLY_WINDOW),
                         # Which reference produced these numbers. Load-bearing for comparing
                         # against any report written before the averaging existed.
                         "average_windows": bool(avg),
                         "window_half_width": hw,
                         "mean_window_depth_years": d_win,
                         "mean_reference_depth_years": d_nc},
              "observed": meta, "desk": zmeta,
              "site_gate": {"rows_kept": int(keep.sum()), "rows_total": int(len(keep)),
                            "cells_kept": int(n_cells_kept), "cells_total": int(n_cells_all),
                            "cells_dropped_no_modern": int(n_cells_all - n_cells_kept)},
              "median_offdiag_observed_similarity": med_off,
              "buckets": {}}

    for sname, smask in splits.items():
        for wname, wmask in windows.items():
            m = smask & wmask
            if m.sum() < 3:
                report["buckets"][f"{sname}/{wname}"] = {"n_rows": int(m.sum()),
                                                         "skipped": "fewer than 3 rows"}
                continue
            ix = np.where(m)[0]
            g = np.ix_(ix, ix)
            # Same predictor table as every other question, dot and cosine both. The bolt-on
            # form this replaced computed the bar's skill with two extra calls and reported one
            # scalar from each, so the bar never got the quantities DESK got.
            gram_preds = {"desk": S_desk[g], "no_change": S_nc[g]}
            gram_cos = {"desk": S_desk_cos[g], "no_change": S_nc_cos[g]}
            if S_idw is not None:
                gram_preds["spacetime_idw"] = S_idw[g]
                gram_cos["spacetime_idw"] = cosine_gram(Z_idw_s[ix])
            if S_esk is not None:
                gram_preds["esk_oracle"] = S_esk[g]
                gram_cos["esk_oracle"] = S_esk_cos[g]
            dot = compare_predictors(S_true[g], gram_preds, grams=True)
            cos = compare_predictors(S_true[g], gram_cos, grams=True)
            # Specific causes. A blanket reason here is what hid the oracle's absence for several
            # runs: the loop stamped "inputs not supplied" over a predictor nothing had ever
            # passed in, and the completeness check accepted it.
            for miss in set(PREDICTOR_ROLES) - set(gram_preds):
                why = (UNAVAILABLE_BAR_UNBUILT if miss == "spacetime_idw"
                       else (oracle_why or UNAVAILABLE_ORACLE_GATE) if miss == "esk_oracle"
                       else UNAVAILABLE_NOT_WIRED)
                dot["unavailable"][miss] = why
                cos["unavailable"][miss] = why
            if "spacetime_idw" in dot["predictors"]:
                dot["skill_vs_spacetime_idw"] = compare_predictors(
                    S_true[g], gram_preds, reference="spacetime_idw", grams=True)["skill_vs"]
            row = {"n_rows": int(len(ix)), "dot": dot, "cosine": cos,
                   # the achievable-ceiling reference: Ruzicka on the modern OBSERVED community,
                   # never differenced against the model columns (different functional).
                   "observed_ceiling_cka": linear_cka(S_true[g], S_nc_obs[g])}
            report["buckets"][f"{sname}/{wname}"] = row

    # BALANCED aggregate alongside the population-weighted one. Three numbers, not one: the
    # per-bucket rows are already reported above; this adds the unweighted mean over buckets (each
    # region-era counted once, however many rows it happens to contribute) and keeps the
    # population-weighted figure unchanged for comparability with every run reported so far.
    # A rebalanced model can move these two in OPPOSITE directions, which is the whole point.
    _bal_src = [(k, v) for k, v in report["buckets"].items()
                if isinstance(v, dict) and "dot" in v and k.startswith("heldout/")]
    if _bal_src:
        report["balanced_aggregate"] = {
            "note": ("unweighted mean over held-out buckets -- each era counted once regardless of "
                     "row count. Read WITH the population-weighted figure: on a rebalanced model "
                     "the population-weighted one can fall while the model improves, because BBS "
                     "row counts are concentrated on the coasts and in recent decades."),
            "n_buckets": len(_bal_src),
            "rmse_skill_balanced": float(np.mean([v["dot"]["skill_vs"]["desk"]
                                                  for _k, v in _bal_src])),
            "cka_gain_balanced": float(np.mean([v["dot"]["predictors"]["desk"]["cka"]
                                                - v["dot"]["predictors"]["no_change"]["cka"]
                                                for _k, v in _bal_src])),
            "per_bucket": {k: {"skill_vs_no_change": v["dot"]["skill_vs"]["desk"],
                               "n_rows": v.get("n_rows")} for k, v in _bal_src},
        }
        print(f"[bbs-routes] balanced aggregate over {len(_bal_src)} held-out buckets: "
              f"rmse_skill {report['balanced_aggregate']['rmse_skill_balanced']:+.4f} "
              f"(population-weighted heldout/all "
              f"{report['buckets'].get('heldout/all', {}).get('rmse_skill', float('nan')):+.4f})")

    out_dir = config["paths"]["desk_output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "bbs_route_validation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    np.savez_compressed(os.path.join(out_dir, "bbs_route_validation.npz"),
                        keys=keys_s, S_true=S_true.astype("float32"),
                        S_desk=S_desk.astype("float32"), S_nc=S_nc.astype("float32"),
                        S_nc_obs=S_nc_obs.astype("float32"), is_heldout=is_ho,
                        **({"S_idw": S_idw.astype("float32")} if S_idw is not None else {}))

    # ---- epoch x local-neighbourhood analysis (separate outputs, same encode pass)
    ep_report = _run_epoch_analysis(config, keys_all, X_raw_all, ep_cells, ep_e_rows, ep_m_rows,
                                    ep_stats, _gather, zmeta, out_dir, idw_rows=idw_rows)

    b0 = next((v for v in report["buckets"].values() if "observed_sd" in v), None)
    if b0:
        print(f"\nSCALE of the target: observed pairwise similarity mean {b0['observed_mean']:.4f}, "
              f"sd {b0['observed_sd']:.4f} (similarities are in [0,1]).")
        print(f"  sd is the RMSE of the know-nothing model (predict the mean for every pair), so "
              f"it is the r2=0 floor.")
    print(f"\nPRIMARY -- dot product vs observed Ruzicka (the kernel contract, and the quantity "
          f"true_kernel_loss trains on).")
    print(f"  r2 = variance of observed similarity explained; errVar% = share of the NULL's error "
          f"variance that DESK removes.")
    # Columns are generated from the predictor table, so a predictor added to the registry shows
    # up here too. Previously the header hardcoded desk/nochange and the bar could only appear as
    # an appended scalar.
    _first = next((v for v in report["buckets"].values() if "dot" in v), None)
    _names = sorted(_first["dot"]["predictors"]) if _first else []
    if _names:
        print(f"{'bucket':<18}{'n':>7}" + "".join(f"{p[:10]:>12}" for p in _names)
              + "   [skill vs no_change, dot form]")
        for k, v in report["buckets"].items():
            if "skipped" in v:
                print(f"{k:<18}{v['n_rows']:>7}  skipped ({v['skipped']})")
                continue
            print(f"{k:<18}{v['n_rows']:>7}"
                  + "".join(f"{v['dot']['skill_vs'].get(p, float('nan')):>+12.3f}" for p in _names))
        print(f"\n{'bucket':<18}{'n':>7}" + "".join(f"{p[:10]:>12}" for p in _names)
              + "   [same, COSINE form -- free of the ||z||^2 deficit]")
        for k, v in report["buckets"].items():
            if "skipped" in v:
                continue
            print(f"{k:<18}{v['n_rows']:>7}"
                  + "".join(f"{v['cosine']['skill_vs'].get(p, float('nan')):>+12.3f}"
                            for p in _names))
        print(f"\n  pearson^2 vs r2 per predictor -- their gap is calibration loss "
              f"(ranking right, scale wrong):")
        for k, v in report["buckets"].items():
            if "skipped" in v:
                continue
            for pn in _names:
                d = v["dot"]["predictors"][pn]
                print(f"  {k:<18}{pn:<15}pearson^2 {d['pearson_r'] ** 2:>6.3f}  "
                      f"r2 {d['r2']:>6.3f}  calibration loss {d['calibration_loss']:>+6.3f}")
        for pn, reason in (_first["dot"].get("unavailable") or {}).items():
            print(f"  unavailable: {pn} -- {reason}")
    print(f"\nSTRUCTURE-ONLY (CKA / Mantel), per predictor, gain over the no-change null:")
    if _names:
        print(f"{'bucket':<18}{'form':<8}" + "".join(f"{p[:10]:>12}" for p in _names))
        for k, v in report["buckets"].items():
            if "skipped" in v:
                continue
            for form in ("dot", "cosine"):
                base = v[form]["predictors"].get("no_change", {}).get("cka", float("nan"))
                print(f"{k:<18}{form:<8}"
                      + "".join(f"{v[form]['predictors'][p]['cka'] - base:>+12.4f}"
                                for p in _names))
    print(f"\n[bbs-routes] report -> {out}")
    print("[bbs-routes] rmse_skill > 0 means DESK's kernel is closer to observed Ruzicka than the "
          "frozen-modern null is, on GENUINELY OBSERVED data. <= 0 is a real negative result.")
    print("[bbs-routes] If the DOT rows are poor but the COSINE rows are healthy, the failure is "
          "||z|| calibration and not the angular structure -- check the contract line above. The "
          "two forms are the two halves of an exact error decomposition, so read them together.")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sample", type=int, default=4000,
                    help="rows in the similarity matrices (n^2 memory; default 4000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(n_sample=args.n_sample, seed=args.seed)


if __name__ == "__main__":
    main()

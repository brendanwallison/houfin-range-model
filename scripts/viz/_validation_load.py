"""Read the validation JSONs into flat rows, and compute the ONE number the figures compare on.

Every predictor table in this project -- the route buckets, the epoch x neighbourhood types, the
ceiling block, the z-space `absolute_position`, the baseline ladder -- is the same shape::

    {"n", "reference", "predictors": {name: {...}}, "skill_vs": {...}, "unavailable": {name: why}}

so one walker covers all of them. The walk is BY SHAPE, not by a fixed depth: the neighbour
questions nest split/distance-bin/form while `absolute_position` nests only populations, and
depth-coupling is exactly what silently disabled the first version of `assert_complete` (it read a
level the analysis never emits and reported green while inspecting nothing). ``_predictor_leaves``
in ``validate_bbs_routes`` already does this walk correctly, so it is imported rather than
rewritten.

THE RULES ARE IMPORTED, NOT RE-ENCODED. ``resolving_room``, ``NULL_IS_A_FLOOR_FOR``,
``PREDICTOR_DENOISING`` and ``canonical_question`` decide what counts as a floor and what counts as
a ceiling, and those decisions are load-bearing (a spatial question has NO floor, because the
frozen-modern null carries the whole modern spatial structure and is a strong competitor). A
plotting layer that re-stated any of that would drift from the validator on the first edit. All of
these are pure -- plain dicts in, no config, no file I/O, no GPU -- so importing them here costs
nothing.
"""
import json
import math
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.community_encoder.train_DESK.validate_baselines import (  # noqa: E402
    LADDER_ROLES,
    PREDICTOR_ALIASES,
)
from src.community_encoder.train_DESK.validate_bbs_routes import (  # noqa: E402
    NULL_IS_A_FLOOR_FOR,
    PREDICTOR_DENOISING,
    PREDICTOR_ROLES,
    QUESTIONS,
    _predictor_leaves,
    canonical_question,
    resolving_room,
)

REPORT = "validate_report.json"
ROUTES = "bbs_route_validation.json"
EPOCHS = "bbs_epoch_neighborhood.json"

#: Long `unavailable` reasons shortened for a hatch label. The full string still goes in the
#: caption -- the point of the short form is that a heatmap cell can carry a reason at all, not
#: that the reason gets truncated away.
UNAVAILABLE_SHORT = {
    "cell has too few surveys": "no split-half",
    "this question compares a cell with ITSELF": "no distance axis",
    "not wired into this question": "not wired (code gap)",
    "caller passed None": "not supplied",
    "interpolation bar could not be built": "bar unbuilt",
    "oracle refused its representability gate": "oracle gate",
}


def short_reason(text):
    for k, v in UNAVAILABLE_SHORT.items():
        if text and text.startswith(k):
            return v
    return (text or "n/a")[:24]


def canonical_predictor(name):
    """Current name for a predictor, following any rename. Pure.

    34 archived run directories carry `zspace_idw`, and they are the only record of runs that
    cannot be reproduced without the data tree. Canonicalising on READ means one name in every
    figure without rewriting a single one of them -- and without the figures carrying a rename
    they would then have to keep carrying.
    """
    return PREDICTOR_ALIASES.get(name, name)


def canonical_predictors(block):
    """A ``{name: ...}`` mapping with its keys canonicalised. Non-dicts pass through."""
    if not isinstance(block, dict):
        return block
    return {canonical_predictor(k): v for k, v in block.items()}


def _finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def load_run(run_dir):
    """``{name: parsed}`` for whichever of the three JSONs are present. Missing files are None
    rather than an error: a run that has not had `bbs-route-validate` run yet is a normal state,
    and the figures that need it should say so rather than the loader refusing the whole run."""
    out = {}
    for key, fname in (("report", REPORT), ("routes", ROUTES), ("epochs", EPOCHS)):
        path = os.path.join(run_dir, fname)
        out[key] = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None
        out[key + "_path"] = path if os.path.exists(path) else None
    out["run_dir"] = run_dir
    out["label"] = os.path.basename(run_dir.rstrip("/"))
    return out


def tidy(node, root=""):
    """Flatten any nested predictor tables into ``[{path, predictor, unavailable, **metrics}]``.

    One row per (leaf x predictor), including rows for predictors that could not run -- those
    carry ``unavailable`` and no metrics. An absent row and an unavailable row are different
    things and the figures render them differently, so they must not be collapsed here.
    """
    rows = []
    for path, leaf in _predictor_leaves(node, root):
        ref = leaf.get("reference")
        skill = leaf.get("skill_vs") or {}
        wins = leaf.get("win_rate_vs") or {}
        for name, vals in canonical_predictors(leaf.get("predictors") or {}).items():
            row = {"path": path, "predictor": name, "reference": ref, "unavailable": None,
                   "skill": skill.get(name), "win_rate": wins.get(name),
                   "n_leaf": leaf.get("n")}
            if isinstance(vals, dict):
                row.update({k: v for k, v in vals.items() if not isinstance(v, (dict, list))})
            rows.append(row)
        for name, why in canonical_predictors(leaf.get("unavailable") or {}).items():
            rows.append({"path": path, "predictor": name, "reference": ref,
                         "unavailable": why, "skill": None, "win_rate": None,
                         "n_leaf": leaf.get("n")})
    return rows


# --- the room join ------------------------------------------------------------------------------
#
# `resolving_room` prefers `esk_oracle_independent` and falls back to `esk_truncation`, stamping
# `ceiling_shares_target_noise: true` when it does. On the shipped runs it falls back for EVERY
# question under `types`, because the split-half oracle is not among that block's predictors.
#
# The honest ceiling is not missing -- it is computed, in the sibling `ceiling` block, which re-runs
# every question against a split-half truth. Its `desk` and `no_change` rows differ from the `types`
# block's (measured: desk skill -0.032 there against -0.065 here) because the truth is a different,
# noisier sample. So the join has to happen INSIDE `ceiling`: read baseline, model and ceiling from
# the one block where all three are scored against one truth. Reading a ceiling from `ceiling` and a
# model from `types` would put numerator and denominator on different targets.


def honest_room(epochs_json, question, split="heldout", form="dot", model="desk",
                baseline="no_change"):
    """Room and share-of-room for one epoch-neighbourhood question, on the independent ceiling.

    Returns ``{"room", "baseline_r", "ceiling_r", "model_r", "share_of_room", "ceiling",
    "narrow", "source"}`` or a dict carrying ``"refused"`` with the validator's own reason.

    ``share_of_room = (model - baseline) / (ceiling - baseline)`` in pearson-r units, which is the
    unit ``resolving_room`` itself works in -- so the room reported here is directly comparable to
    the ``room`` block in the file, with only the ceiling substituted.
    """
    q = canonical_question(question) or question
    types = ((epochs_json or {}).get("ceiling") or {}).get("types") or {}
    node = types.get(question) or types.get(q)
    if node is None:
        return {"refused": f"question {question!r} absent from the ceiling block"}
    leaf = node.get(split, node)
    leaf = leaf.get("all_distances", leaf)
    leaf = leaf.get(form, leaf)
    room = resolving_room(leaf, baseline=baseline, question=q)
    if room.get("baseline_is_a_floor") is False or "room" not in room:
        return {"refused": room.get("note", "room undefined"), "question": q}
    preds = leaf.get("predictors") or {}
    m = preds.get(model, {}).get("pearson_r")
    b, c = room["baseline_r"], room["ceiling_r"]
    share = None
    if _finite(m) and _finite(b) and _finite(c) and abs(c - b) > 1e-9:
        share = (float(m) - float(b)) / (float(c) - float(b))
    return {"question": q, "split": split, "form": form,
            "baseline_r": b, "ceiling_r": c, "model_r": m,
            "ceiling": room["ceiling"], "room": room["room"],
            "shares_target_noise": bool(room.get("ceiling_shares_target_noise")),
            "share_of_room": share, "narrow": bool(room["room"] < 0.15),
            "n": leaf.get("n"), "source": "ceiling-block (split-half truth)",
            "skill_vs": leaf.get("skill_vs") or {},
            "unavailable": leaf.get("unavailable") or {},
            "why": (QUESTIONS.get(q) or {}).get("why", "")}


def shipped_room(epochs_json, question, split="heldout", form="dot"):
    """The `room` block AS SHIPPED, for the side-by-side that shows why the join matters.

    Kept so a figure can draw both and let the overstatement be seen rather than asserted.
    """
    types = ((epochs_json or {}).get("types") or {})
    node = types.get(question)
    if node is None:
        return None
    leaf = node.get(split, node)
    leaf = leaf.get("all_distances", leaf)
    leaf = leaf.get(form, leaf)
    return leaf.get("room")


def epoch_direction_rooms(report_json, panel="windowed"):
    """The direction panel's own room, per epoch pair. Already computed by ``_ceiling_row``.

    Pairs are returned as a LIST in file order and never aggregated: they share cells and nest in
    time (1967->2025 contains 1985->2005), so a pooled figure would overstate the evidence.
    """
    ed = ((report_json or {}).get("epoch_directions") or {}).get(panel) or {}
    out = []
    for name, p in (ed.get("pairs") or {}).items():
        a, b = name.split("_")
        out.append({"pair": f"{a}→{b}", "key": name, **p})
    return out


def ladder_table(report_json, bucket=None):
    """``(columns, rows, cell)`` for the baseline ladder as a heatmap.

    ``cell[(rung, column)]`` is ``{"win_rate", "median_err", "n"}`` or ``{"unavailable": reason}``.
    A rung with ``n == 0`` or a non-finite median error is UNAVAILABLE, not zero -- under a spatial
    holdout ``cell_nearest_year``/``cell_trend`` cannot run at all, and under a temporal holdout
    ``borrowed_delta`` cannot. That pattern is structural and documented on ``baseline_panel``.
    """
    bl = (report_json or {}).get("baseline_ladder") or {}
    if bucket:
        bl = (bl.get("temporal_buckets") or {}).get(bucket) or {}
    by_era = bl.get("by_era") or {}
    columns = sorted(by_era) + (["overall"] if bl.get("overall") else [])
    rungs, cell = [], {}
    for col in columns:
        blk = bl["overall"] if col == "overall" else by_era[col]
        for name, vals in (blk.get("predictors") or {}).items():
            if name not in rungs:
                rungs.append(name)
            n = int(vals.get("n") or 0)
            wr = (blk.get("win_rate_vs") or {}).get(name)
            if n == 0 or not _finite(vals.get("median_err")):
                cell[(name, col)] = {"unavailable": "structurally n/a"}
            else:
                cell[(name, col)] = {"win_rate": wr, "median_err": vals.get("median_err"), "n": n}
    return columns, rungs, cell


def distance_curves(report_json, bucket):
    """``[{bin, lo, hi, predictors: {name: median_err}, wins: {name: rate}, n}]`` for one bucket."""
    tb = (((report_json or {}).get("baseline_ladder") or {}).get("temporal_buckets")
          or {}).get(bucket)
    if not tb:
        return []
    out = []
    for name, blk in (tb.get("by_distance") or {}).items():
        lo, hi = name.lstrip("d").split("-")
        # A by_distance entry is a whole baseline_panel result, so its predictor table sits under
        # `overall` -- reading it off the top level silently yields nothing.
        inner = blk.get("overall") or {}
        out.append({"bin": name, "lo": int(lo), "hi": int(hi),
                    "n": inner.get("n", blk.get("graded_rows")),
                    "predictors": {k: v.get("median_err")
                                   for k, v in (inner.get("predictors") or {}).items()
                                   if v.get("n")},
                    "wins": inner.get("win_rate_vs") or {}})
    return sorted(out, key=lambda r: r["lo"])


def glossary():
    """Every term the figures use, from the registries that define it.

    Returns ``[(name, [(stream, text)], shared)]``. The captions assumed this vocabulary and
    nothing in the output supplied it, which is a poor trade: the definitions already exist as
    ``LADDER_ROLES`` (the z-space bars) and ``PREDICTOR_ROLES`` (the similarity-space predictors),
    and both are imported here anyway. Restating them in the HTML would make a third copy to
    drift; reading them is free.

    THREE NAMES APPEAR IN BOTH, and for one of them that is a trap rather than a repetition.
    ``no_change`` is the cell's own OBSERVED community at the recent year in the z-space stream,
    and DESK's OWN z frozen at the modern year in the similarity stream. Different objects, one
    name. The similarity stream needs the model-frozen form on purpose: an observed-space null is
    scored in truth's own metric while DESK is not, and that mismatch alone was worth -0.28 on a
    temporally-neutral model. So both definitions are shown, labelled, rather than merged --
    merging them is precisely the confusion this exists to prevent.
    """
    names = list(LADDER_ROLES) + [n for n in PREDICTOR_ROLES if n not in LADDER_ROLES]
    out = []
    for name in names:
        defs = []
        if name in LADDER_ROLES:
            defs.append(("z-space (‖z_pred − z_obs‖)", LADDER_ROLES[name][0]))
        if name in PREDICTOR_ROLES:
            defs.append(("similarity space (kernel / Ružička)", PREDICTOR_ROLES[name]))
        out.append((name, defs, len(defs) > 1))
    return out


def question_glossary():
    """``[(name, pairs, observed, why)]`` for the five named questions the route stream asks."""
    return [(q, v.get("pairs", ""), v.get("observed", ""), v.get("why", ""))
            for q, v in QUESTIONS.items()]

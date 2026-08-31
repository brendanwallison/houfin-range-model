"""Draw the DESK validation report: one figure per question, plus an index.html.

The validation suite is thorough and entirely numeric -- four drivers emit ~2 MB of nested JSON per
run and nothing has ever been plotted from any of it. The cost shows up in the git history: several
conclusions here were withdrawn not because a number was wrong but because it was read without its
SCALE. A dir-cos of 0.48 beat its permutation null at 0.22 and still lost to plain inverse-distance
at 0.51. A pearson of 0.995 was quoted as a ceiling when it was the target passed through a filter.
A shared colour scale on a turnover panel "cost an entire investigation".

The suite has since grown the right guards -- `resolving_room`, `_ceiling_row`, `PREDICTOR_DENOISING`,
`assert_complete` -- but they are prose inside a 50 KB console log. These figures draw the scale
instead of stating it, so that "skill 0.011" and "captures 4% of the resolvable signal" are one
glance rather than two lookups.

    python scripts/viz/validation_report.py --run-dir results/crossed/processed/encoder/desk_tempho_1995
    python scripts/viz/validation_report.py --arms results/*/processed/encoder/desk_tempho_1995 --out /tmp/arms

Figures 1-8 need only the three JSONs a `validate` + `bbs-route-validate` run writes. The maps,
the covariate-R2 lane of the spectrum panel, and the training curves need artifacts that live
beside them on TACC (`validate_spacetime.npz`, `component_predictability.json`,
`train_trajectory.jsonl`); each says so rather than silently rendering an empty axis.
"""
import argparse
import base64
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _validation_load as L          # noqa: E402
import _validation_style as S         # noqa: E402
import matplotlib.pyplot as plt       # noqa: E402


def _fin(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _bar_predictors(ax, names, values, y0=0.0, width=0.8, annotate="{:.4f}", suspect=()):
    """Horizontal bars in the suite's fixed predictor colours, skipping non-finite values.

    ``suspect`` names predictors whose information set is not the model's on these rows. They are
    hatched rather than dropped: the number is real, it just does not answer the same question,
    and removing it would hide a comparison the report does make.
    """
    drawn = []
    for i, name in enumerate(names):
        v = values.get(name)
        if not _fin(v):
            continue
        ax.barh(y0 + i, float(v), height=width, color=S.color(name),
                edgecolor="white", linewidth=0.5,
                hatch="////" if name in suspect else None)
        ax.text(float(v), y0 + i, "  " + annotate.format(float(v)), va="center",
                fontsize=7, color="#333333")
        drawn.append((i, name))
    return drawn


# =================================================================================================
# ACT I -- orientation: what was asked
# =================================================================================================

def fig01_design(run, out_dir):
    """The withheld spans, the pinned anchors, and how BBS coverage grows toward the present.

    A pooled temporal number is dominated by whichever end of the record carries the most rows, and
    BBS coverage grows ~7x from 1966 to 1995. Drawing the two together is what stops "skill fell
    with extrapolation distance" being read off a figure whose weight moved at the same time.
    """
    rep = run["report"]
    if not rep:
        return None
    bl = rep.get("baseline_ladder") or {}
    first_trained = bl.get("first_trained_year")
    common = bl.get("common_holdout_years") or []

    fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), height_ratios=[1.15, 1],
                           gridspec_kw={"hspace": 0.45})

    # --- lane 1: the three overlay runs, and which years each withheld ---------------------------
    lanes = []
    for cfg in sorted((os.path.join(_REPO, "config", "overlays", f)
                       for f in os.listdir(os.path.join(_REPO, "config", "overlays"))
                       if f.startswith("desk_tempho_"))):
        d = json.load(open(cfg, encoding="utf-8"))
        hy = d["desk"]["trend"]["holdout_years"]
        lanes.append((os.path.basename(cfg).replace(".json", "").replace("desk_tempho_", "run "),
                      min(hy), max(hy), d["desk"]["trend"]))
    a = ax[0]
    for i, (name, lo, hi, tr) in enumerate(lanes):
        a.barh(i, 2025 - 1966 + 1, left=1966, height=0.55, color="#dfe6ec", edgecolor="none")
        a.barh(i, hi - lo + 1, left=lo, height=0.55, color="#c1440e", alpha=0.55,
               edgecolor="none")
        a.text(hi + 1.5, i, f" withheld {lo}–{hi}  ({hi - lo + 1} yr)", va="center", fontsize=7.5,
               color="#8a3208")
    if common:
        a.axvspan(min(common), max(common), color="#1f6fb4", alpha=0.13, zorder=0)
        a.annotate("common window — withheld in every run,\nso the only cross-run comparable set",
                   xy=(float(np.mean([min(common), max(common)])), -0.62), ha="center", va="top",
                   fontsize=7, color="#1f6fb4")
    for yr, lab, col in ((1996, "anchor 1996\n(trained-era control)", "#2e8b74"),
                         (1975, "anchor 1975\n(the measurement)", "#7b5ea7")):
        a.axvline(yr, color=col, lw=1.2, ls="--")
        a.text(yr, -0.95, lab, ha="center", va="top", fontsize=6.5, color=col)
    a.set_yticks(range(len(lanes)))
    a.set_yticklabels([n for n, *_ in lanes], fontsize=8)
    a.set_xlim(1963, 2032)
    a.set_ylim(-1.6, len(lanes) - 0.2)
    a.set_title("The temporal experiment: contiguous spans withheld from the DESK objective, "
                "backward from 1966", fontsize=9.5, loc="left")
    for sp in ("top", "right", "left"):
        a.spines[sp].set_visible(False)

    # --- lane 2: rows per decade, i.e. where the weight of any pooled number sits ----------------
    eras = [k for k in rep if k.endswith("0s") and k[0].isdigit()]
    eras.sort()
    ns = [rep[e]["n"] for e in eras]
    b = ax[1]
    def _era_state(e):
        if not first_trained:
            return "trained"
        if int(e[:4]) + 9 < first_trained:
            return "withheld"
        return "trained" if int(e[:4]) >= first_trained else "straddles"

    states = [_era_state(e) for e in eras]
    cols = {"withheld": "#c1440e", "trained": "#1f6fb4", "straddles": "#c1440e"}
    b.bar(range(len(eras)), ns, color=[cols[st] for st in states], alpha=0.75,
          hatch=["//" if st == "straddles" else "" for st in states], edgecolor="white")
    for i, n in enumerate(ns):
        b.text(i, n, f"{n:,}", ha="center", va="bottom", fontsize=7)
    b.set_xticks(range(len(eras)))
    b.set_xticklabels(eras, fontsize=8)
    b.set_ylabel("supervised cell-years", fontsize=8)
    # max, not the last decade: the 2020s is a partial decade and using it understates the growth
    # by nearly 3x, which is exactly the confound this panel exists to make visible.
    b.set_title(f"Where the rows are — {max(ns) / max(ns[0], 1):.0f}x growth from the 1960s to "
                f"the peak.  Red = withheld here, blue = trained, hatched = straddles "
                f"{first_trained}.", fontsize=9.5, loc="left")
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)

    return S.finish(fig, os.path.join(out_dir, "01_design.png"),
                    "Extrapolation distance and sample size move together, so any pooled temporal "
                    "figure is weighted toward its own shallow, cheap end. That is why the sweep "
                    "pins a common withheld window and reports by-distance bins.")


# =================================================================================================
# ACT I -- which kind of extrapolation is hard
# =================================================================================================

def fig02_axes(run, out_dir):
    """Place, time, and both -- the same z-space error under each kind of extrapolation.

    The first three groups are MARGINALS from `absolute_position.populations` (a held-out cell over
    all years; a withheld year over all cells) and the fourth is the ladder's crossed bucket. They
    are drawn as marginals rather than forced into a 2x2, because the populations genuinely
    overlap and a grid would imply a partition the file does not contain.
    """
    rep = run["report"]
    if not rep:
        return None
    ap = ((rep.get("zspace_reconstruction") or {}).get("absolute_position") or {})
    pops = ap.get("populations") or {}
    tb = ((rep.get("baseline_ladder") or {}).get("temporal_buckets") or {})
    groups = []
    for key, title, sub in (
            ("train", "in-sample", "trained cells, all years"),
            ("heldout", "unseen PLACE", "held-out cell, 162 km block + buffer"),
            ("withheld_years", "unseen TIME", "withheld year, any cell")):
        blk = pops.get(key)
        if blk:
            groups.append((title, sub, blk["n"],
                           {k: v.get("median_err")
                            for k, v in L.canonical_predictors(blk["predictors"]).items()}))
    both = (tb.get("unseen_year_unseen_cell") or {}).get("overall")
    if both:
        groups.append(("unseen PLACE and TIME", "held-out cell in a withheld year", both["n"],
                       {k: v.get("median_err")
                        for k, v in L.canonical_predictors(both["predictors"]).items()
                        if v.get("n")}))
    if not groups:
        return None

    names = S.ordered({n for _, _, _, d in groups for n in d})
    # DETECTED, not assumed. `spatial_idw` interpolates WITHIN a year, so on a population that is
    # entirely withheld years it can only produce a number by reading those years -- information
    # the model never saw anywhere. A finite value there is therefore proof the run predates the
    # `exclude_years` fix; after it the bar has no admissible source and is absent with a stated
    # reason. Keying on the file rather than on a config flag means the same figure reads an old
    # archived run and a fresh one correctly.
    _wy = L.canonical_predictors(
        (pops.get("withheld_years") or {}).get("predictors", {})).get("spatial_idw", {})
    pre_fix = _fin(_wy.get("median_err"))
    suspect = {"spatial_idw"} if pre_fix else set()
    # NOT sharey: a shared y-axis shares one ticker, so blanking the labels on panels 2..n would
    # blank them on panel 1 as well. The rows are aligned by fixed index instead.
    fig, axs = plt.subplots(1, len(groups), figsize=(3.5 * len(groups) + 1.6, 3.9), sharex=True)
    axs = np.atleast_1d(axs)
    hi = max(float(v) for _, _, _, d in groups for v in d.values() if _fin(v))
    for k, (ax, (title, sub, n, vals)) in enumerate(zip(axs, groups)):
        # FIXED row per predictor across every panel. The predictor sets differ (the crossed
        # bucket has no same-year bar at all), and a per-panel y order would silently put two
        # different predictors on the same visual row.
        _bar_predictors(ax, names, vals, suspect=suspect)
        nc = vals.get("no_change")
        if _fin(nc):
            ax.axvline(float(nc), color=S.color("no_change"), ls="--", lw=1, zorder=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([S.label(nm) for nm in names] if k == 0 else [""] * len(names),
                           fontsize=8)
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, hi * 1.34)
        ax.set_title(f"{title}\n{sub}\nn={n:,}", fontsize=9)
        ax.set_xlabel("median ‖z_pred − z_obs‖  (lower is better)", fontsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    cap = ("Read the ORDER, not the level. `spacetime_idw` beats DESK wherever it has the target "
           "cell's own observations to borrow from — trained cells, and withheld years at cells it "
           "has seen — and loses on a held-out block, where it does not. That is a statement about "
           "what the bar is handed, not a spatial advantage for DESK: the in-sample column is very "
           "nearly tautological for an interpolator.")
    if pre_fix:
        cap += (" HATCHED: this run predates the exclude_years fix, so the spatial bar interpolated a "
                "withheld year's truth from training cells surveyed in the SAME year — a different "
                "information set from the model's, not a weaker bar. Its numbers are not "
                "comparable on any population containing withheld years, which here is all of "
                "them. Re-run `validate` to replace them.")
    elif (rep.get("baseline_ladder") or {}).get("common_holdout_years"):
        cap += (" The spatial bar is absent from the withheld-year populations by construction: it "
                "interpolates within a year, and a withheld year has no admissible source. That "
                "absence is the correct result, which is why the spacetime variant exists.")
    return S.finish(fig, os.path.join(out_dir, "02_extrapolation_axes.png"), cap)


# =================================================================================================
# ACT II -- scale: how much room is there
# =================================================================================================

def fig03_room(run, out_dir):
    """Floor to honest ceiling, with the model placed inside it. THE orienting figure.

    Every skill number in this suite is a ratio against a floor, and none of them says how far the
    ceiling is. `resolving_room` computes that distance and refuses when the baseline is not a
    floor -- but on the shipped runs it falls back to `esk_truncation` for every question, because
    the split-half oracle is not among that block's predictors. The honest ceiling is not missing;
    it is in the sibling `ceiling` block, which re-runs each question against a split-half truth.
    Both are drawn, so the overstatement is visible rather than asserted.
    """
    ep, rep = run["epochs"], run["report"]
    if not ep and not rep:
        return None
    rows = []
    for q in (((ep or {}).get("ceiling") or {}).get("types") or {}):
        h = L.honest_room(ep, q, split="heldout")
        shipped = L.shipped_room(ep, q, split="heldout")
        h["bar_r"] = ((((ep or {}).get("ceiling") or {}).get("types") or {})
                      .get(q, {}).get("heldout", {}).get("all_distances", {})
                      .get("dot", {}).get("predictors", {})
                      .get("spacetime_idw", {}).get("pearson_r"))
        rows.append(("route · " + q, h, shipped))
    for p in L.epoch_direction_rooms(rep, "windowed"):
        if _fin(p.get("room")):
            rows.append(("z-space direction · " + p["pair"],
                         {"question": p["key"], "baseline_r": p["null_dir_cos"],
                          "ceiling_r": p["ceiling_dir_cos"], "model_r": p["model_dir_cos"],
                          "room": p["room"], "share_of_room": p.get("share_of_room"),
                          "ceiling": "esk_oracle_independent", "n": p["n"],
                          "narrow": p["room"] < S.NARROW_ROOM,
                          "bar_r": p.get("spacetime_idw_dir_cos"),
                          "why": "the SAME change vector rebuilt from two disjoint halves of each "
                                 "window's years — an independent observation of the same place"},
                         None))
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(11.5, 0.52 * len(rows) + 2.4))
    for i, (name, h, shipped) in enumerate(rows):
        y = len(rows) - 1 - i
        if "refused" in h:
            ax.plot([-0.012], [y], marker="x", ms=6, color="#a05a2c", zorder=4)
            ax.text(0.0, y, f"   REFUSED — {h['refused'][:104]}…", va="center", fontsize=6.8,
                    color="#a05a2c", style="italic")
            continue
        b, c, m = h["baseline_r"], h["ceiling_r"], h["model_r"]
        narrow = h["narrow"]
        ax.plot([b, c], [y, y], lw=7, solid_capstyle="butt",
                color="#e8e8e8" if narrow else "#cfe0ee", zorder=1)
        ax.plot([b], [y], marker="|", ms=13, color=S.color("no_change"), zorder=3)
        ax.plot([c], [y], marker="|", ms=13, color=S.color("esk_oracle_independent"), zorder=3)
        if _fin(h.get("bar_r")):
            ax.plot([h["bar_r"]], [y], marker="D", ms=5, color=S.color("spacetime_idw"), zorder=4)
        if _fin(m):
            ax.plot([m], [y], marker="o", ms=7, color=S.color("desk"), zorder=5)
        if shipped and _fin(shipped.get("ceiling_r")) and shipped.get(
                "ceiling_shares_target_noise"):
            ax.plot([shipped["ceiling_r"]], [y], marker="|", ms=11, ls="none",
                    color=S.color("esk_truncation"), zorder=3)
            ax.plot([c, shipped["ceiling_r"]], [y, y], lw=1.0, ls=":",
                    color=S.color("esk_truncation"), zorder=2)
        sh = h.get("share_of_room")
        txt = "n/a" if not _fin(sh) else f"{100 * sh:.0f}%"
        ax.text(1.035, y, txt, va="center", ha="right", fontsize=8,
                color="#999999" if narrow else "#222222",
                fontweight="normal" if narrow else "bold")
        if narrow:
            ax.text(1.05, y, " NARROW", va="center", fontsize=6.5, color="#b03030")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, *_ in reversed(rows)], fontsize=7.8)
    ax.set_xlim(-0.02, 1.16)
    ax.set_xlabel("pearson r against the observed truth  (the unit resolving_room works in)",
                  fontsize=8)
    ax.set_title("How much room each comparison HAS, and where DESK sits in it", fontsize=11,
                 loc="left", pad=34)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="|", ls="none", ms=11, color=S.color("no_change"),
                   label="floor (no-change null)"),
        plt.Line2D([], [], marker="o", ls="none", ms=7, color=S.color("desk"), label="DESK"),
        plt.Line2D([], [], marker="|", ls="none", ms=11,
                   color=S.color("esk_oracle_independent"),
                   label="ceiling — independent split-half observation"),
        plt.Line2D([], [], marker="|", ls="none", ms=11, color=S.color("esk_truncation"),
                   label="esk_truncation — shares the target's noise, NOT a ceiling"),
    ]
    handles.insert(2, plt.Line2D([], [], marker="D", ls="none", ms=5,
                                 color=S.color("spacetime_idw"),
                                 label="spacetime IDW (the honest bar)"))
    ax.legend(handles=handles, fontsize=7, frameon=False, ncol=3,
              loc="lower left", bbox_to_anchor=(0.0, 1.015))

    return S.finish(fig, os.path.join(out_dir, "03_room.png"),
                    "Percentages at the right are share_of_room = (model − floor) / (ceiling − "
                    "floor). The dotted extension marks where the shipped `room` block put the "
                    "ceiling: esk_truncation is the target passed through a rank-64 filter, noise "
                    "included, so reading it as a bound overstates the room by roughly half. Rows "
                    "with no bar were REFUSED by resolving_room — for a same-era spatial question "
                    "the frozen-modern null carries the whole modern spatial structure and is a "
                    "strong competitor, not a floor, so the comparison cannot rank anything.")


def fig04_noise_ceiling(run, out_dir):
    """Why a perfect prediction still cannot reach dir-cos 1.0, and why the shortfall moves by era.

    The target is raw BBS at ~1.08 routes per cell-year, so a single-year endpoint is one observer
    on one morning. That noise inflates ‖dt‖ and randomises its direction, attenuating every
    dir-cos toward zero -- and unequally by era, since the first-year-observer share is 25.6% in
    1966-1980 against 12.3% in 2001-2025. The attenuation is therefore DIFFERENTIAL along the very
    axis the temporal sweep varies.
    """
    rep = run["report"]
    ed = (rep or {}).get("epoch_directions") or {}
    sweep = ed.get("half_width_sweep") or {}
    # Two shapes for the same thing: the epoch panel writes the era dict directly, while the
    # top-level `per_era_attenuation` wraps it in `by_era`. Reading only one silently produces an
    # empty panel, which is indistinguishable from "there was no attenuation to report".
    atten = ed.get("attenuation_by_era")
    if not isinstance(atten, dict) or "by_era" in atten:
        atten = (atten or (rep or {}).get("per_era_attenuation") or {}).get("by_era") or {}
    atten = {k: v for k, v in atten.items() if isinstance(v, dict)
             and "dir_cos_attenuation" in v}
    if not sweep and not atten:
        return None

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))

    a = ax[0]
    if sweep.get("widths"):
        ws = sorted(sweep["widths"], key=lambda k: int(k))
        dc = [sweep["widths"][w]["median_model_dir_cos"] for w in ws]
        nd = [sweep["widths"][w]["mean_surveys_per_endpoint"] for w in ws]
        a.plot([int(w) for w in ws], dc, "-o", color=S.color("desk"), label="median model dir-cos")
        a.set_xlabel("endpoint half-width (years averaged either side)", fontsize=8)
        a.set_ylabel("median model dir-cos", fontsize=8)
        op = sweep.get("operative")
        if op is not None:
            a.axvline(int(op), color="#888888", ls="--", lw=1)
            a.text(int(op), min(dc), f" operative = {op}", fontsize=7, color="#666666",
                   va="bottom")
        a2 = a.twinx()
        a2.plot([int(w) for w in ws], nd, "-s", ms=4, color="#1f6fb4", alpha=0.6)
        a2.set_ylabel("mean surveys per endpoint", fontsize=8, color="#1f6fb4")
        a2.tick_params(axis="y", colors="#1f6fb4", labelsize=7)
        # The curve is not monotone and should not be titled as if it were: widening the window
        # buys quieter endpoints but eventually averages away real change (+/-2 yr already smooths
        # ~10% of a 30-50 yr interval). The peak is where the two effects cross.
        _best = ws[int(np.argmax(dc))]
        a.set_title(f"Same model, quieter endpoints: dir-cos {dc[0]:.3f} → {max(dc):.3f}\n"
                    f"the gain IS the noise; the fall past hw={_best} is real change averaged away",
                    fontsize=9)
        a.legend(handles=[plt.Line2D([], [], marker="o", color=S.color("desk"),
                                     label="median model dir-cos (left)"),
                          plt.Line2D([], [], marker="s", ms=4, color="#1f6fb4",
                                     label="surveys per endpoint (right)")],
                 fontsize=7, frameon=False, loc="lower right")
    for sp in ("top",):
        a.spines[sp].set_visible(False)

    b = ax[1]
    if atten:
        eras = sorted(atten)
        att = [atten[e]["dir_cos_attenuation"] for e in eras]
        share = [atten[e]["noise_share_of_long_gap"] for e in eras]
        b.plot(eras, att, "-o", color="#7b5ea7", label="dir-cos attenuation  √(1 − adj/long)")
        b.plot(eras, share, "-s", ms=4, color="#b0a08c",
               label="noise share of a 20-yr gap")
        obs = None
        pairs = (ed.get("windowed") or {}).get("pairs") or {}
        if pairs:
            obs = float(np.median([p["model_dir_cos"] for p in pairs.values()]))
            b.axhline(obs, color=S.color("desk"), ls="--", lw=1)
            b.text(0.02, obs, f" observed median dir-cos {obs:.3f}", fontsize=7,
                   color=S.color("desk"), va="bottom", transform=b.get_yaxis_transform())
            m = float(np.mean(att))
            b.axhline(obs / m, color=S.color("desk"), ls=":", lw=1)
            b.text(0.02, obs / m, f" ÷ attenuation → true ≈ {obs / m:.3f}", fontsize=7,
                   color=S.color("desk"), va="bottom", transform=b.get_yaxis_transform())
        b.set_ylim(0, 1)
        _spread = max(att) - min(att)
        b.set_title(f"Survey noise by era — attenuation {min(att):.2f}–{max(att):.2f} "
                    f"(spread {_spread:.2f})", fontsize=9.5)
        b.legend(fontsize=7, frameon=False, loc="upper right")
        b.tick_params(axis="x", labelsize=7.5)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)

    return S.finish(fig, os.path.join(out_dir, "04_noise_ceiling.png"),
                    "Never read a dir-cos against 1.0. Its ceiling is what an INDEPENDENT "
                    "observation of the same place scores, and the target here is a handful of "
                    "surveys. Because the attenuation differs by era, it also moves along the axis "
                    "the temporal sweep varies — so part of any apparent decay with extrapolation "
                    "distance is the record getting noisier, not the model getting worse.")


# =================================================================================================
# ACT III -- where the skill comes from
# =================================================================================================

def fig05_ladder(run, out_dir):
    """Each rung is handed DIFFERENT information, so the row that wins says WHICH claim survives.

    Beating the no-change null is nearly free -- it assumes sixty years of stasis, and is weakest
    exactly where the historical points are densest. Beating `spacetime_idw`, which may borrow from
    anything near in space AND time, is the claim that the covariates say something interpolation
    does not. DESK is a row in this table, not the subject of it.
    """
    rep = run["report"]
    if not rep:
        return None
    cols, rungs, cell = L.ladder_table(rep)
    if not cols:
        return None
    rungs = S.ordered(rungs)
    first_trained = (rep.get("baseline_ladder") or {}).get("first_trained_year")

    fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 3.9, 0.52 * len(rungs) + 3.0))
    grid = np.full((len(rungs), len(cols)), np.nan)
    for i, r in enumerate(rungs):
        for j, c in enumerate(cols):
            v = cell.get((r, c), {})
            # The REFERENCE is 0% by definition -- it cannot beat itself. Colouring it on the same
            # diverging scale paints the definition dark red, which reads as "worst predictor".
            if r == (rep.get("baseline_ladder") or {}).get("overall", {}).get(
                    "reference", "no_change"):
                continue
            if "win_rate" in v and _fin(v["win_rate"]):
                grid[i, j] = v["win_rate"]
    im = ax.imshow(grid, cmap="RdBu", vmin=0.0, vmax=1.0, aspect="auto")
    ref = (rep.get("baseline_ladder") or {}).get("overall", {}).get("reference", "no_change")
    for i, r in enumerate(rungs):
        for j, c in enumerate(cols):
            v = cell.get((r, c))
            if v is None:
                continue
            if r == ref:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f2f2f2",
                                           edgecolor="white", zorder=2))
                ax.text(j, i, "reference", ha="center", va="center", fontsize=6.5,
                        color="#888888", zorder=3)
            elif "unavailable" in v or not _fin(v.get("win_rate")):
                S.hatch_unavailable(ax, j - 0.5, i - 0.5, 1, 1, "n/a")
            else:
                ax.text(j, i, f"{100 * v['win_rate']:.0f}%", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if abs(v["win_rate"] - 0.5) > 0.32 else "#222222")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(rungs)))
    ax.set_yticklabels([S.label(r) for r in rungs], fontsize=8)
    if first_trained:
        edge = sum(1 for c in cols if c.endswith("0s") and int(c[:4]) + 9 < first_trained)
        if 0 < edge < len(cols):
            ax.axvline(edge - 0.5, color="#c1440e", lw=2)
            # The decade straddling first_trained is part withheld and part trained, so it is
            # marked rather than being silently assigned to one side.
            ax.annotate(f"← withheld    trained →   (first trained {first_trained}; "
                        f"{cols[edge]} straddles the line)",
                        xy=(edge - 0.55, 1.012), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=7.5, color="#c1440e")
    ax.set_title("Share of rows where each predictor beats the no-change null, by era",
                 fontsize=10.5, loc="left", pad=22)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label(
        "beats the null on this fraction of rows", fontsize=7.5)

    return S.finish(fig, os.path.join(out_dir, "05_ladder.png"),
                    "Hatched cells are STRUCTURALLY unavailable, not failures: under a spatial "
                    "holdout a held-out cell has no training years of its own, so cell_trend and "
                    "cell_nearest_year cannot run; under a temporal holdout there are no training "
                    "points in the withheld years, so borrowed_delta cannot. Neither holdout alone "
                    "exercises the whole ladder — which is the argument for running both.")


def fig06_distance(run, out_dir):
    """Error against years past the training edge -- the backward-extrapolation claim itself.

    Pooling a whole withheld block hides this axis, and because BBS coverage grows toward the
    present each run's pooled figure is dominated by its own shallow end. That is why three runs
    withholding 10, 20 and 30 years produced pooled numbers that barely moved.
    """
    rep = run["report"]
    if not rep:
        return None
    panels = [(b, L.distance_curves(rep, b)) for b in
              ("unseen_year_seen_cell", "unseen_year_unseen_cell",
               "common_window_seen_cell", "common_window_unseen_cell")]
    # A single run reaches its common window at exactly one distance, so those panels carry one
    # point and are a line plot of nothing. They become a figure only ACROSS runs -- see --arms.
    panels = [(b, c) for b, c in panels if len(c) >= 2]
    if not panels:
        return None

    fig, axs = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 4.1), sharey=True)
    axs = np.atleast_1d(axs)
    names = S.ordered({n for _, cur in panels for r in cur for n in r["predictors"]})
    for ax, (bucket, curves) in zip(axs, panels):
        x = [(r["lo"] + r["hi"]) / 2 for r in curves]
        for nm in names:
            ys = [r["predictors"].get(nm) for r in curves]
            if not any(_fin(y) for y in ys):
                continue
            ax.plot(x, [y if _fin(y) else np.nan for y in ys], "-o", ms=4,
                    color=S.color(nm), label=S.label(nm),
                    lw=2.4 if nm == "desk" else 1.4,
                    zorder=3 if nm == "desk" else 2)
        ax.set_xticks(x)
        ax.set_xticklabels([r["bin"] for r in curves], fontsize=7.5)
        ax.set_xlabel("years past the training edge", fontsize=8)
        title = bucket.replace("_", " ")
        ax.set_title(f"{title}\n" + " · ".join(f"{r['bin']}: n={r['n']:,}" for r in curves),
                     fontsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("median ‖z_pred − z_obs‖", fontsize=8)
    # One figure-level legend built from EVERY panel's predictors. A per-panel legend shows only
    # what that panel happened to have, and the rungs that go structurally n/a differ by panel --
    # so the reader would see a shorter legend and read it as a shorter ladder.
    fig.legend(handles=[plt.Line2D([], [], marker="o", ms=4, color=S.color(nm), label=S.label(nm))
                        for nm in names],
               fontsize=7, frameon=False, ncol=min(len(names), 5),
               loc="lower center", bbox_to_anchor=(0.5, 0.90))
    for ax in axs:
        ax.set_title(ax.get_title(), fontsize=8.5, pad=6)

    return S.finish(fig, os.path.join(out_dir, "06_distance.png"),
                    "cell_trend degrades fastest — a per-cell line fitted to trained years and "
                    "extrapolated 30 years is exactly the thing that should. A single run reaches "
                    "its common window at one distance only, so the cross-run panels are drawn by "
                    "`--arms`; each run's own unseen_year block covers a different year set and "
                    "the three are not comparable directly.",
                    tight_rect=(0, 0.15, 1, 0.90))


def fig07_direction_magnitude(run, out_dir):
    """Direction and magnitude are two halves of one exact identity, and they TRADE OFF.

        ‖a−b‖² = (‖a‖−‖b‖)²  +  2‖a‖‖b‖(1−cos θ)
                 |-magnitude-|    |---angular---|

    Minimising the total over ‖a‖ at a fixed cosine ρ gives ‖a‖ = ρ‖b‖ exactly, so the diagonal
    below is the MSE-optimal locus and shrinking is the CORRECT response to a poor angle. A point
    above the line is moving further than its direction accuracy justifies.
    """
    rep = run["report"]
    pairs = L.epoch_direction_rooms(rep, "windowed")
    if not pairs:
        return None

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), width_ratios=[1.25, 1])
    a = ax[0]
    lim = 1.22 * max([p["change_magnitude_ratio"] for p in pairs
                      if _fin(p["change_magnitude_ratio"])]
                     + [p["ceiling_dir_cos"] for p in pairs if _fin(p.get("ceiling_dir_cos"))]
                     + [0.4])
    xs = np.linspace(0, lim, 50)
    a.plot(xs, xs, color="#888888", lw=1.2, ls="--")
    a.text(lim * 0.72, lim * 0.72, " MSE-optimal: ‖pred‖ = cos·‖truth‖", fontsize=7.5,
           color="#666666", rotation=38, va="bottom")
    a.fill_between(xs, xs, lim * 1.05, color="#c1440e", alpha=0.05)
    a.text(lim * 0.12, lim * 0.93, "over-moving", fontsize=8, color="#c1440e")
    a.text(lim * 0.62, lim * 0.06, "hedging", fontsize=8, color="#1f6fb4")

    for series, colname, marker in (("desk", "model_dir_cos", "o"),
                                    ("spacetime_idw", "spacetime_idw_dir_cos", "s"),
                                    ("no_change", "null_dir_cos", "x")):
        xv = [p.get(colname) for p in pairs]
        yv = [p["change_magnitude_ratio"] if series == "desk" else np.nan for p in pairs]
        if series != "desk":
            # These two have a direction but no magnitude ratio in the panel, so they sit on a
            # labelled baseline strip rather than being given a fabricated y.
            a.plot([v for v in xv if _fin(v)], [-lim * 0.045] * sum(_fin(v) for v in xv), marker,
                   ms=5, color=S.color(series), label=S.label(series), ls="none", alpha=0.9,
                   clip_on=False)
            continue
        a.plot(xv, yv, marker, ms=8, color=S.color(series), ls="none", label="DESK, per epoch pair")
        # Alternate the label side so the two shortest pairs, which land almost on top of each
        # other, stay readable.
        for j, (p, x, y) in enumerate(zip(pairs, xv, yv)):
            if _fin(x) and _fin(y):
                a.annotate(p["pair"], (x, y), fontsize=6.5,
                           xytext=(6, 5) if j % 2 == 0 else (-6, -11),
                           ha="left" if j % 2 == 0 else "right",
                           textcoords="offset points", color="#444444")
    ceil = [p["ceiling_dir_cos"] for p in pairs if _fin(p.get("ceiling_dir_cos"))]
    if ceil:
        a.axvspan(min(ceil), max(ceil), color=S.color("esk_oracle_independent"), alpha=0.10)
        a.text(float(np.mean(ceil)), lim * 0.965, "ceiling\n(independent observation)",
               fontsize=7, ha="center", va="top", color="#1a7f37")
    a.set_xlim(-0.02, lim)
    a.set_ylim(-lim * 0.09, lim)
    a.text(-0.015, -lim * 0.045, "dir-cos only →", fontsize=6.5, color="#666666",
           ha="right", va="center")
    a.axhline(0, color="#eeeeee", lw=0.8, zorder=0)
    a.set_xlabel("direction cosine  (angular half)", fontsize=8.5)
    a.set_ylabel("magnitude ratio  ‖Δpred‖ / ‖Δtruth‖", fontsize=8.5)
    a.set_title("Every epoch pair, plotted as its two error halves", fontsize=10, loc="left")
    a.legend(fontsize=7, frameon=False, loc="lower right", ncol=1)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

    b = ax[1]
    idx = np.arange(len(pairs))
    mag = [p["err_magnitude_share"] for p in pairs]
    ang = [p["err_angular_share"] for p in pairs]
    b.barh(idx, mag, color="#b0a08c", label="magnitude share of ‖Δpred−Δtruth‖²")
    b.barh(idx, ang, left=mag, color="#7b5ea7", label="angular share")
    for i, p in enumerate(pairs):
        b.text(1.01, i, f"cal {p['magnitude_calibration']:.1f}×", va="center", fontsize=7)
    b.set_yticks(idx)
    b.set_yticklabels([p["pair"] for p in pairs], fontsize=8)
    b.invert_yaxis()
    b.set_xlim(0, 1.16)
    b.set_title("…and where that error actually sits", fontsize=10, loc="left", pad=18)
    b.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)

    return S.finish(fig, os.path.join(out_dir, "07_direction_magnitude.png"),
                    "`cal` is magnitude ratio ÷ dir-cos: 1.0 is MSE-calibrated, above 1 means "
                    "moving further than the direction accuracy justifies. Pairs are never pooled "
                    "— they share cells and nest in time (1967→2025 contains 1985→2005), so an "
                    "aggregate would overstate the evidence. A missing IDW bar is 'no admissible "
                    "bar', not a tie.")


def fig08_co_movement(run, out_dir):
    """Do NEARBY places move the same way? Everything else scores each place on its own.

    A model curve sitting above the observed one means neighbouring places are being given too
    similar a change -- regional structure smoothed away. This is currently the sharpest single
    diagnosis in the report and it exists only as a prose `note`.
    """
    rep = run["report"]
    dc = (rep or {}).get("directional_change") or {}
    splits = {k: v for k, v in (dc.get("by_split") or {}).items() if v.get("co_movement")}
    if dc.get("co_movement"):
        splits.setdefault("pooled", dc)
    if not splits:
        return None
    order = [s for s in ("heldout", "train", "pooled") if s in splits]

    fig, axs = plt.subplots(1, len(order), figsize=(4.3 * len(order), 4.0), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, split in zip(axs, order):
        bins = splits[split]["co_movement"]["bins"]
        x = [(b["km_lo"] + b["km_hi"]) / 2 for b in bins]
        m = [b["model_co_movement"] for b in bins]
        o = [b["observed_co_movement"] for b in bins]
        ax.fill_between(x, o, m, color=S.color("desk"), alpha=0.13)
        ax.plot(x, m, "-o", ms=4, color=S.color("desk"), label="model")
        ax.plot(x, o, "-s", ms=4, color="#333333", label="observed (BBS)")
        ax.axhline(0, color="#cccccc", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("separation between two cells (km)", fontsize=8)
        d = splits[split]
        ax.set_title(f"{split}   n_sites={d.get('n_sites', '?')}\n"
                     f"mean dir-cos {d.get('mean_dir_cos', float('nan')):+.3f} "
                     f"vs null {d.get('mean_dir_cos_null', float('nan')):+.3f}", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("agreement between two places' change directions", fontsize=8)
    axs[0].legend(fontsize=7.5, frameon=False)

    return S.finish(fig, os.path.join(out_dir, "08_co_movement.png"),
                    (splits[order[0]]["co_movement"].get("note") or "").strip())


# =================================================================================================
# ACT IV -- why: components
# =================================================================================================

def fig09_spectrum(run, out_dir, comp=None):
    """One shared 1..64 axis: how much each ESK direction carries, how much of it is real, how much
    the covariates reach, and what DESK actually does to it.

    Four independent measurements that today live in four separate tables. Putting them on one axis
    means the reading is a vertical alignment rather than a cross-reference -- which is the whole
    question of whether the encoder or the covariates are the bottleneck.
    """
    rep = run["report"]
    sn = (rep or {}).get("per_dimension_signal_noise") or {}
    zs = (rep or {}).get("zspace_reconstruction") or {}
    if not sn and not zs:
        return None
    lanes = []
    if sn.get("total_var"):
        lanes.append(("variance at a 20-yr gap\nnoise vs real signal", "total_var", None))
    if sn.get("snr"):
        lanes.append(("signal / noise\n(fixed 20±2 yr gap)", "snr", None))
    if comp:
        lanes.append(("held-out R² from covariates\nvs the interpolation bars", "r2", None))
    if zs.get("shrinkage_by_dim"):
        lanes.append(("DESK shrinkage\nvar(z_desk)/var(z_obs)", "shrink", None))
    if not lanes:
        return None

    fig, axs = plt.subplots(len(lanes), 1, figsize=(11, 2.05 * len(lanes)), sharex=True)
    axs = np.atleast_1d(axs)
    L64 = None
    for ax, (title, key, _) in zip(axs, lanes):
        if key == "total_var":
            # STACKED and LINEAR. These span barely 2x, so a log axis renders them as two flat
            # lines and a fill to the axis floor -- which reads as "all noise" regardless of the
            # numbers. Stacking noise under signal draws the decomposition itself.
            tot = np.asarray(sn["total_var"], float)
            noise = np.asarray(sn["noise_var"], float)
            L64 = len(tot)
            xk = np.arange(1, L64 + 1)
            ax.fill_between(xk, 0, noise, color="#b03030", alpha=0.30,
                            label="noise (adjacent-year differences)")
            ax.fill_between(xk, noise, tot, color="#2e8b74", alpha=0.30,
                            label="signal (real 20-yr change)")
            ax.plot(xk, tot, color="#333333", lw=1.0)
            ax.set_ylim(0, float(np.nanmax(tot)) * 1.08)
            ax.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=2)
        elif key == "snr":
            y = np.asarray(sn["snr"], float)
            L64 = len(y)
            ax.plot(np.arange(1, L64 + 1), y, color="#7b5ea7", lw=1.4)
            ax.axhline(1.0, color="#888888", ls="--", lw=0.9)
            ax.text(1, 1.0, " signal = noise", fontsize=7, color="#666666", va="bottom")
            ax.set_ylim(0, max(2.0, float(np.nanmax(y)) * 1.1))
        elif key == "r2":
            d = comp["decompositions"]
            best = comp.get("best_model")
            src = (comp.get("capacity_ladder") or {}).get(best) or \
                  (comp.get("context_ladder") or {}).get(best) or {}
            y = np.asarray(src.get("r2", []), float)
            L64 = len(y) or L64
            ax.plot(np.arange(1, len(y) + 1), y, color="#c1440e", lw=1.6,
                    label=f"covariates ({best})")
            # The same-year bar is only admissible if the run excluded the withheld years from
            # it. `bars_exclude_years` records what was actually done; its ABSENCE means the run
            # predates that provenance key, so the lane cannot be vouched for and says so instead
            # of being drawn as a peer.
            _excl = d.get("bars_exclude_years")
            _sp_ok = _excl is not None
            for k, nm, col in (("r2_spatial_idw", "spatial IDW bar", S.color("spatial_idw")),
                               ("r2_spacetime_idw", "spacetime IDW bar",
                                S.color("spacetime_idw"))):
                if not d.get(k):
                    continue
                v = np.asarray(d[k], float)
                dashed = (k == "r2_spatial_idw" and not _sp_ok)
                ax.plot(np.arange(1, len(v) + 1), v, lw=1.1, color=col,
                        ls=(0, (2, 2)) if dashed else "-",
                        label=nm + (" — provenance unknown" if dashed else ""))
            if d.get("achievable_r2_level"):
                v = np.asarray(d["achievable_r2_level"], float)
                ax.plot(np.arange(1, len(v) + 1), v, lw=1.0, ls=":", color="#1a7f37",
                        label="achievable (noise-rescaled)")
            ax.axhline(0, color="#cccccc", lw=0.8)
            ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right")
        else:
            y = np.asarray([np.nan if v is None else v for v in zs["shrinkage_by_dim"]], float)
            L64 = len(y)
            ax.plot(np.arange(1, L64 + 1), y, color=S.color("desk"), lw=1.4)
            ax.axhline(1.0, color="#888888", ls="--", lw=0.9)
            sl = zs.get("shrinkage_slope")
            if _fin(sl):
                ax.text(0.99, 0.93, f"slope {sl:+.4f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.5, color="#666666")
        ax.set_ylabel(title, fontsize=7.5)
        for b in (8, 16, 32):
            ax.axvline(b + 0.5, color="#dddddd", lw=0.9, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[-1].set_xlabel("ESK component (eigen order — the basis is kernel-PCA)", fontsize=8.5)
    axs[0].set_title("The 64 latent directions, measured four ways on one axis  "
                     "(bands at 8 / 16 / 32)", fontsize=10.5, loc="left")

    # Say which case THIS run is in rather than restating both and leaving the reader to check.
    sl = zs.get("shrinkage_slope")
    verdict = ""
    if _fin(sl):
        verdict = (f"This run's profile FALLS (slope {sl:+.4f}): the trailing directions are being "
                   f"squeezed hardest. "
                   if float(sl) < -0.002 else
                   f"This run's profile is FLAT (slope {sl:+.4f}), i.e. close to a uniform "
                   f"rescale. ")
    cap = (verdict +
           "A FALLING profile means the low-eigenvalue directions are squeezed hardest — and "
           "since MSE shrinks whatever it predicts worst, those are the temporal ones, so the "
           "kernel is tilted toward spatial similarity and the downstream GP prior is distorted. "
           "A FLAT profile is a uniform rescale, which the fitted w_env absorbs at no cost. "
           "Aggregate ‖z‖² cannot tell the two apart.")
    if not comp:
        cap += ("  The covariate-R² lane needs component_predictability.json, which is not in "
                "this run directory.")
    elif comp["decompositions"].get("bars_exclude_years") is None:
        cap += ("  DASHED: this component_predictability.json predates the `bars_exclude_years` "
                "provenance key, so whether its same-year IDW bar excluded the withheld years "
                "cannot be established from the file. `r2_gain_over_best_bar` takes the "
                "elementwise max of the two bars, so a contaminated spatial bar understates the "
                "covariates' gain. Re-run the diagnostic.")
    return S.finish(fig, os.path.join(out_dir, "09_spectrum.png"), cap)


def fig12_maps(run, out_dir):
    """Is DESK's advantage over the null spatially STRUCTURED, or is it noise?

    Everything above is a scalar over held-out rows. A scalar cannot say whether the win is at the
    range edge, on the coast, in the interior, or spread evenly -- and those imply different next
    moves. Per-panel scaling with an annotated median, following the discipline the turnover maps
    learned the hard way: a single shared vmax once rendered a predicted panel at 20% of its range
    and cost an entire investigation.
    """
    npz_path = os.path.join(run["run_dir"], "validate_spacetime.npz")
    if not os.path.exists(npz_path):
        return None
    z = np.load(npz_path, allow_pickle=True)
    if "recon_rows" not in z or np.asarray(z["recon_rows"]).size == 0:
        return None
    rows, cols = np.asarray(z["recon_rows"]).astype(int), np.asarray(z["recon_cols"]).astype(int)
    ed, en = np.asarray(z["recon_err_desk"]), np.asarray(z["recon_err_nochange"])
    H, W = _grid_shape(z, rows, cols)

    def _cell_mean(idx_r, idx_c, vals):
        lin = idx_r * W + idx_c
        ssum = np.bincount(lin, weights=vals, minlength=H * W)
        cnt = np.bincount(lin, minlength=H * W).astype(float)
        g = np.full(H * W, np.nan)
        m = cnt > 0
        g[m] = ssum[m] / cnt[m]
        return g.reshape(H, W)

    panels = [(_cell_mean(rows, cols, ed), "DESK reconstruction error", "magma", None),
              (_cell_mean(rows, cols, en), "no-change error", "magma", None)]
    diff = panels[1][0] - panels[0][0]
    panels.append((diff, "no-change − DESK   (blue = DESK better)", "RdBu", "sym"))
    if "dir_cos" in z and np.asarray(z["dir_cos"]).size:
        dr = np.asarray(z["dirchg_rows"]).astype(int)
        dc = np.asarray(z["dirchg_cols"]).astype(int)
        panels.append((_cell_mean(dr, dc, np.asarray(z["dir_cos"])),
                       "direction of change: per-cell cos", "RdBu", "sym"))

    ho = None
    ho_path = os.path.join(run["run_dir"], "holdout_cells.npy")
    if os.path.exists(ho_path):
        _h = np.load(ho_path)
        if _h.shape == (H, W):
            ho = _h

    fig, axs = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.4))
    axs = np.atleast_1d(axs)
    for ax, (g, title, cmap, mode) in zip(axs, panels):
        fin = np.isfinite(g)
        if mode == "sym":
            v = float(np.nanpercentile(np.abs(g[fin]), 98)) if fin.any() else 1.0
            lo, hi = -v, v
        else:
            lo = 0.0
            hi = float(np.nanpercentile(g[fin], 98)) if fin.any() else 1.0
        im = ax.imshow(g, cmap=cmap, vmin=lo, vmax=hi)
        if ho is not None:
            ax.contour(ho.astype(float), levels=[0.5], colors="#111111", linewidths=0.6)
        med = float(np.nanmedian(g[fin])) if fin.any() else float("nan")
        ax.set_title(f"{title}\nmedian {med:+.3f}   scale "
                     + (f"±{hi:.3f}" if mode == "sym" else f"0–{hi:.3f}"), fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    cap = ("Panels are scaled INDEPENDENTLY and each carries its own median, so a systematic "
           "level gap is a number rather than a colour nobody can decode. ")
    cap += ("Black outlines are the held-out blocks."
            if ho is not None else
            "holdout_cells.npy is not beside this run, so the held-out blocks are not outlined.")
    return S.finish(fig, os.path.join(out_dir, "12_maps.png"), cap)


def _grid_shape(z, rows, cols):
    """Grid shape from the reference raster the run recorded, falling back to the project grid and
    finally to the index extent -- which is a LOWER bound and shifts nothing, since the arrays are
    indexed by (row, col) from the same origin either way."""
    for path in (str(z["ref_raster"]) if "ref_raster" in z else None,):
        if path and os.path.exists(path):
            import rasterio
            with rasterio.open(path) as src:
                return src.height, src.width
    try:
        from src.config_utils import load_data_config
        ref = load_data_config()["grid"]["ref_raster"]
        if os.path.exists(ref):
            import rasterio
            with rasterio.open(ref) as src:
                return src.height, src.width
    except Exception:
        pass
    return int(rows.max()) + 1, int(cols.max()) + 1


# =================================================================================================
# cross-arm
# =================================================================================================

def fig13_arms(runs, out_dir, question="pair_convergence"):
    """One headline per sweep arm x overlay run, with the seed-noise band drawn.

    A difference smaller than the run-to-run spread of a fixed configuration is not a result, and
    that spread is ~6.6% on this trainer. Drawing it is the difference between a scoreboard and an
    invitation to over-read one.
    """
    cells, why = {}, {}
    for r in runs:
        arm = r.get("arm") or "?"
        h = L.honest_room(r["epochs"], question, split="heldout") if r["epochs"] else \
            {"refused": "no bbs_epoch_neighborhood.json"}
        cells[(arm, r["label"])] = h.get("share_of_room") if "share_of_room" in h else None
        # A missing value has a REASON, and the reasons differ in kind: an older run predates the
        # split-half ceiling block entirely, which is not the same as a question the ceiling could
        # not resolve. A bare "n/a" would merge them.
        if "share_of_room" not in h or not _fin(h.get("share_of_room")):
            r_txt = h.get("refused", "")
            why[(arm, r["label"])] = ("no ceiling block" if "absent from the ceiling" in r_txt
                                      else "no file" if "no bbs_epoch" in r_txt
                                      else "refused")
    arms = sorted({a for a, _ in cells})
    labs = sorted({b for _, b in cells})
    if not arms or not labs:
        return None
    grid = np.full((len(arms), len(labs)), np.nan)
    for i, a in enumerate(arms):
        for j, b in enumerate(labs):
            v = cells.get((a, b))
            if _fin(v):
                grid[i, j] = float(v)

    fig, ax = plt.subplots(figsize=(1.3 * len(labs) + 4.5, 0.44 * len(arms) + 3.0))
    # Diverging ONLY when the sign varies. share_of_room can go negative (a model below its own
    # floor), but on an all-positive column a diverging map spends half its range on values that
    # do not occur and renders a real spread as three shades of the same blue.
    fin = grid[np.isfinite(grid)]
    diverging = fin.size and fin.min() < 0 < fin.max()
    lim = float(np.nanmax(np.abs(grid))) if fin.size else 1.0
    im = ax.imshow(grid, cmap="RdBu" if diverging else "Blues", aspect="auto",
                   vmin=-lim if diverging else 0.0, vmax=lim)
    # The seed spread is a property of a DIFFERENCE, so it is applied down each column: anything
    # within it of that column's best is tied with the best, not distinguishable from it.
    best = {j: np.nanmax(grid[:, j]) if np.isfinite(grid[:, j]).any() else np.nan
            for j in range(len(labs))}
    for i in range(len(arms)):
        for j in range(len(labs)):
            v = grid[i, j]
            if np.isfinite(v):
                tied = _fin(best[j]) and (best[j] - v) <= S.SEED_NOISE
                ax.text(j, i, f"{100 * v:.0f}%" + (" ≈" if tied else ""),
                        ha="center", va="center", fontsize=7.5,
                        color="white" if (not diverging and v > 0.66 * lim) else "#222222",
                        fontweight="bold" if tied else "normal")
            else:
                S.hatch_unavailable(ax, j - 0.5, i - 0.5, 1, 1,
                                    why.get((arms[i], labs[j]), "n/a"))
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms, fontsize=8)
    ax.set_title(f"share_of_room on `{question}`\nheld-out cells, independent split-half ceiling",
                 fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    return S.finish(fig, os.path.join(out_dir, "13_arms.png"),
                    f"'≈' marks an arm within {100 * S.SEED_NOISE:.1f}% of its column's best — "
                    f"the seed-to-seed spread of a fixed configuration, so those arms are tied "
                    f"with the best rather than behind it. Every column falls with extrapolation "
                    f"distance, which is the signal; the between-arm differences mostly are not.")



def fig14_cross_run_decay(runs, out_dir):
    """Skill against extrapolation distance, on the window every run withheld. THE sweep result.

    This is the only comparison in which the three overlay runs measure the same thing. Each run
    is graded on `common_holdout_years` (1966-1975), withheld by all three, so the target rows,
    the cells and the truth are identical -- the shipped numbers confirm it: n=9,389 / 1,652 and a
    no-change null of 0.6054 / 0.5876 in every run. The ONLY thing that varies is how far past its
    own training edge each run has to reach: 1-10, 11-20, 21-30 years.

    Each run's own `unseen_year` block cannot do this. It covers a different year set weighted to
    its own shallow end, so three such numbers differ for a reason that is not extrapolation.
    """
    by_arm = {}
    for r in runs:
        rep = r["report"]
        if not rep:
            continue
        bl = rep.get("baseline_ladder") or {}
        com = bl.get("common_holdout_years") or []
        ft = bl.get("first_trained_year")
        if not com or not ft:
            continue
        for bucket in ("common_window_seen_cell", "common_window_unseen_cell"):
            blk = ((bl.get("temporal_buckets") or {}).get(bucket) or {}).get("overall")
            # `baseline_panel` returns {"note": "no target rows"} when a bucket is empty, and the
            # driver skips a bucket below 20 rows entirely -- so `overall` can exist without a
            # predictor table. Older arms predate parts of this and hit both paths.
            if not isinstance(blk, dict) or not isinstance(blk.get("predictors"), dict):
                continue
            by_arm.setdefault((r.get("arm") or "?", bucket), []).append(
                {"dist": ft - int(np.mean([min(com), max(com)])),
                 "lo": ft - max(com), "hi": ft - min(com), "n": blk["n"],
                 "predictors": {k: v.get("median_err")
                                for k, v in blk["predictors"].items() if v.get("n")}})
    if not by_arm:
        return None
    buckets = [b for b in ("common_window_seen_cell", "common_window_unseen_cell")
               if any(k[1] == b for k in by_arm)]
    arms = sorted({k[0] for k in by_arm})
    names = S.ordered({n for v in by_arm.values() for row in v for n in row["predictors"]})

    fig, axs = plt.subplots(len(arms), len(buckets),
                            figsize=(4.6 * len(buckets), 3.4 * len(arms) + 0.6),
                            squeeze=False, sharey=True)
    for i, arm in enumerate(arms):
        for j, bucket in enumerate(buckets):
            ax = axs[i][j]
            rows = sorted(by_arm.get((arm, bucket), []), key=lambda d: d["dist"])
            if len(rows) < 2:
                ax.text(0.5, 0.5, f"{arm}: only {len(rows)} run(s) with this bucket",
                        transform=ax.transAxes, ha="center", fontsize=8, color="#999999")
                ax.set_axis_off()
                continue
            x = [d["dist"] for d in rows]
            for nm in names:
                ys = [d["predictors"].get(nm) for d in rows]
                if not any(_fin(y) for y in ys):
                    continue
                ax.plot(x, [y if _fin(y) else np.nan for y in ys], "-o", ms=4,
                        color=S.color(nm), label=S.label(nm),
                        lw=2.4 if nm == "desk" else 1.4, zorder=3 if nm == "desk" else 2)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{d['lo']}–{d['hi']} yr" for d in rows], fontsize=8)
            ax.set_title(f"{arm} · {bucket.replace('_', ' ')}   (n={rows[0]['n']:,}, "
                         f"identical in every run)", fontsize=8.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if j == 0:
                ax.set_ylabel("median ‖z_pred − z_obs‖", fontsize=8)
            ax.set_xlabel("years past this run's training edge", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ms=4, color=S.color(nm), label=S.label(nm))
               for nm in names]
    fig.legend(handles=handles, fontsize=7, frameon=False, ncol=min(len(names), 5),
               loc="lower center", bbox_to_anchor=(0.5, 0.955))

    return S.finish(fig, os.path.join(out_dir, "14_cross_run_decay.png"),
                    "Same rows, same truth, same null in all three runs — the only variable is "
                    "reach. Read the SPREAD between the predictors, not their levels: where the "
                    "no-change null is flat by construction, a predictor whose line rises is "
                    "losing skill to distance, and one that stays below the null at 21–30 years "
                    "is still extrapolating usefully that far back.",
                    tight_rect=(0, 0.10, 1, 0.94))


# =================================================================================================
# driver
# =================================================================================================

FIGURES = (fig01_design, fig02_axes, fig03_room, fig04_noise_ceiling,
           fig05_ladder, fig06_distance, fig07_direction_magnitude, fig08_co_movement)


#: The narrative the figures are ordered by. Each act answers a question the previous one raises,
#: and the headings are carried into the HTML so the sequence is legible as a sequence rather than
#: as nine unrelated panels.
ACTS = {
    "fig01_design": ("I · Orientation", "What was asked"),
    "fig02_axes": ("I · Orientation", "What was asked"),
    "fig03_room": ("II · Scale", "How much room is there — and what does a number MEAN here"),
    "fig04_noise_ceiling": ("II · Scale",
                            "How much room is there — and what does a number MEAN here"),
    "fig05_ladder": ("III · Attribution", "Where the skill comes from"),
    "fig06_distance": ("III · Attribution", "Where the skill comes from"),
    "fig07_direction_magnitude": ("III · Attribution", "Where the skill comes from"),
    "fig08_co_movement": ("III · Attribution", "Where the skill comes from"),
    "fig09_spectrum": ("IV · Mechanism", "Which latent directions, and why"),
    "fig12_maps": ("IV · Mechanism", "Which latent directions, and why"),
}


def summary(fn):
    """First PARAGRAPH of a figure's docstring, as one line.

    Splitting on the first newline truncates mid-sentence, which is how a caption written to state
    what a figure answers ends up stating half of it.
    """
    doc = (fn.__doc__ or "").strip()
    return " ".join(doc.split("\n\n")[0].split())


def _uri(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def _glossary_html():
    """The vocabulary the captions assume, from the registries that define it.

    Put FIRST, not in an appendix: `spatial_idw` and `spacetime_idw` differ in exactly the way that
    decides how to read half these figures, and a reader who does not have that distinction cannot
    get the second panel right.
    """
    rows = []
    for name, defs, shared in L.glossary():
        tag = ' <em class="shared">two definitions — see both</em>' if shared else ""
        body = "".join(f'<dd><span class="sp">{st}</span> {tx}</dd>' for st, tx in defs)
        rows.append(f'<dt><code>{name}</code>{tag}</dt>{body}')
    qs = "".join(
        f'<dt><code>{q}</code></dt><dd><span class="sp">{pairs}</span> observed = <code>{obs}'
        f'</code><br>{why}</dd>'
        for q, pairs, obs, why in L.question_glossary())
    return (f'<details open><summary>Vocabulary — what each predictor is handed</summary>'
            f'<dl>{"".join(rows)}</dl></details>'
            f'<details><summary>The five named questions the route stream asks</summary>'
            f'<dl>{qs}</dl></details>')


def build_html(out_dir, figs, meta, title="DESK validation"):
    parts, seen_act = [], None
    for p, cap in figs:
        act = cap[1] if isinstance(cap, tuple) else None
        text = cap[0] if isinstance(cap, tuple) else cap
        # Dedup on the act NAME, not on the whole tuple: a subtitle that differs by a word would
        # otherwise reopen the same act as a second heading.
        if act and act[0] != seen_act:
            parts.append(f"<h2>{act[0]}</h2><p class=\"act\">{act[1]}</p>")
            seen_act = act[0]
        parts.append(f'<figure><figcaption>{text}</figcaption>'
                     f'<img src="{_uri(p)}" style="width:100%;height:auto"></figure>')
    body = "\n".join(parts)
    html = f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;
line-height:1.6;color:#222}}figure{{margin:0 0 2.6rem}}
figcaption{{font-size:.86rem;color:#555;margin-bottom:.5rem;border-left:3px solid #ddd;
padding-left:.7rem}}h1{{font-family:Georgia,serif;margin-bottom:.2rem}}
p.meta{{color:#666;font-size:.85rem}}
h2{{font-family:Georgia,serif;margin:2.4rem 0 0;border-top:1px solid #eee;padding-top:1.4rem}}
p.act{{color:#777;font-size:.9rem;margin:.2rem 0 1.4rem}}
details{{background:#fafafa;border:1px solid #eee;border-radius:6px;padding:.8rem 1rem;
margin:0 0 1rem}}summary{{cursor:pointer;font-weight:600}}
dl{{margin:.8rem 0 0}}dt{{margin-top:.7rem}}dd{{margin:.15rem 0 0 1.2rem;font-size:.88rem;
color:#444}}span.sp{{color:#888;font-size:.8rem;display:inline-block;min-width:0}}
em.shared{{color:#b03030;font-style:normal;font-size:.78rem}}
code{{background:#eef;padding:.1em .3em;border-radius:3px}}</style>
<h1>{title}</h1><p class="meta">{meta}</p>{_glossary_html()}{body}"""
    p = os.path.join(out_dir, "index.html")
    open(p, "w", encoding="utf-8").write(html)
    return p


def _load_component(run_dir):
    for cand in (os.path.join(run_dir, "component_predictability.json"),
                 os.path.join(os.path.dirname(run_dir.rstrip("/")),
                              "component_predictability.json")):
        if os.path.exists(cand):
            return json.load(open(cand, encoding="utf-8"))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None,
                    help="directory holding validate_report.json etc. "
                         "(default: paths.desk_output_dir)")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="several run dirs -> the cross-arm scoreboard instead")
    ap.add_argument("--out", default=None, help="output dir (default: <run-dir>/validate_viz)")
    ap.add_argument("--question", default="pair_convergence",
                    help="question for the cross-arm scoreboard")
    args = ap.parse_args()

    if args.arms:
        out = args.out or "validation_arms"
        os.makedirs(out, exist_ok=True)
        runs = []
        for d in args.arms:
            r = L.load_run(d)
            parts = os.path.abspath(d).split(os.sep)
            r["arm"] = parts[parts.index("results") + 1] if "results" in parts else parts[-1]
            runs.append(r)
        figs = []
        for fn, cap in ((fig14_cross_run_decay, "skill against extrapolation distance, on the "
                                                "window every run withheld"),
                        (fig13_arms, "share of resolvable signal, per sweep arm")):
            made = fn(runs, out) if fn is fig14_cross_run_decay else fn(
                runs, out, question=args.question)
            if made:
                figs.append((made, cap))
                print(f"[viz] {os.path.basename(made)}")
            else:
                print(f"[viz] skip {fn.__name__} (inputs absent)")
        if figs:
            print(f"[viz] {build_html(out, figs, f'{len(runs)} runs', 'DESK validation — sweep')}")
        return

    run_dir = args.run_dir
    if run_dir is None:
        from src.config_utils import load_config
        run_dir = load_config()["paths"]["desk_output_dir"]
    run = L.load_run(run_dir)
    if not any(run[k] for k in ("report", "routes", "epochs")):
        raise SystemExit(f"no validation JSON under {run_dir}")
    out = args.out or os.path.join(run_dir, "validate_viz")
    os.makedirs(out, exist_ok=True)
    comp = _load_component(run_dir)

    figs = []
    for fn in FIGURES:
        made = fn(run, out)
        if made:
            figs.append((made, (summary(fn), ACTS.get(fn.__name__))))
            print(f"[viz] {os.path.basename(made)}")
        else:
            print(f"[viz] skip {fn.__name__} (inputs absent)")
    made = fig09_spectrum(run, out, comp=comp)
    if made:
        figs.append((made, (summary(fig09_spectrum), ACTS.get("fig09_spectrum"))))
        print(f"[viz] {os.path.basename(made)}"
              + ("" if comp else "  (covariate-R² lane needs component_predictability.json)"))
    made = fig12_maps(run, out)
    if made:
        figs.append((made, (summary(fig12_maps), ACTS.get("fig12_maps"))))
        print(f"[viz] {os.path.basename(made)}")
    else:
        print("[viz] skip fig12_maps (validate_spacetime.npz not beside this run)")

    meta = (f"{run['label']} — graded on {(run['report'] or {}).get('graded_on', '?')}; "
            f"{(run['report'] or {}).get('point_coverage', {}).get('n_encoded', '?'):,} encoded "
            f"cell-years" if run["report"] else run["label"])
    title = "DESK validation — " + run["label"]
    print(f"[viz] {build_html(out, figs, meta, title=title)}")


if __name__ == "__main__":
    main()

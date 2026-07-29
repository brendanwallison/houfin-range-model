#!/usr/bin/env python3
"""Compare MAP runs across a juvenile-dispersal-distance sweep.

Reads the artifacts each sweep point already produced -- never re-runs a model, so
this is CPU-only and takes seconds. Input is the manifest written by
``scripts/tacc/submit_juv_mdd_sweep.sh``; points whose fit is missing or
incomparable are marked excluded rather than silently plotted.

The primary comparison is the **viability raster** (``source_mask`` in each run's
``map_diagnostics/07_source_sink_fields.npz``): small multiples across mdd, signed
lambda differences against a reference point, maps of cells that flip
viable<->non-viable, and a pairwise classification-agreement matrix. That last
number is the compact answer to "how much does the conclusion actually move with
dispersal distance."

``source_mask`` is the FOLD criterion -- a positive equilibrium exists, i.e.
``max_N F(N) > (1-Sa)/(Sa*Sj)`` -- not ``lambda > 1``; see
``src/vis/age_model_math.allee_viability``. The ``lam_realized_modern`` field the
difference panels use is lambda AT N = K, which ``c`` pins at 1 wherever the Allee
factor saturates, so read those panels as differences in Allee cost, not in growth
rate. Runs whose npz predates that change carry a ``source_mask`` thresholded on
``lambda > 1``, whose boundary was set by a 1e-6 regularizer and is not comparable
to a current one -- the git-sha check in ``cross_check`` is what stops the two from
being mixed.

Fit quality is reported alongside to spot outliers, NOT to rank points:

* ``log1p_rmse`` / ``log1p_correlation`` on the BBS observation scale, plus
  ``n_observations`` as a hard identity check -- if that differs across points the
  comparison is void and every point is excluded.
* Final MAP loss, with the last-100-step trend. MAP loss is a legitimate relative
  signal for a hyperparameter that enters only through the design matrix (same
  observations, same likelihood, same parameter dimensionality, same priors), but
  it is an optimum, not a marginal likelihood -- it ignores posterior volume, so a
  sharper mode can win on density while being worse on evidence. Under ``quick90``
  with no repeat fits there is also no measured optimizer-noise floor, so the
  residual last-100 trend is printed as a lower bound on what counts as noise.
  Differences smaller than that are not attributable to dispersal distance.
* Fitted ``allee_n50`` and niche summaries. Dispersal distance trades off against
  fecundity, K, and the Allee threshold in producing an observed spread rate, so a
  compensating drift here is the tell that a flat comparison reflects
  non-identifiability rather than genuine insensitivity.
* ``realized_discrete_mdd_km`` vs target from ``path_feature_meta.json`` (finite-
  lattice truncation biases it low), the per-band kernel mass fractions, and an
  independent re-verification that the recorded dispersal spec still matches a
  freshly resolved one.

Usage:
    python scripts/viz/juv_mdd_sweep_summary.py --manifest <SWEEP_ROOT>/sweep_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as _gamma, gammainc

from src.config_utils import load_age_model_config


def band_mass_fractions(splits, mdd, shape):
    """Continuous kernel mass in each radial band -- the property splits control."""
    g2, g3 = _gamma(2.0 / shape), _gamma(3.0 / shape)
    scale = mdd / (g3 / g2)
    cdf = [0.0 if r <= 0 else (1.0 if r >= 1e8
           else float(gammainc(2.0 / shape, (r / scale) ** shape))) for r in splits]
    return [b - a for a, b in zip(cdf[:-1], cdf[1:])]


def load_point(entry, results_dir, cell_km=27.0):
    """Collect every comparable quantity for one sweep point."""
    rec = {
        "point": entry["point"],
        "juvenile_mdd_km": float(entry["juvenile_mdd_km"]),
        "splits_km": entry["resolved_splits_km"],
        "run_dir": str(Path(results_dir) / entry["run_dir"]),
        "git_sha": entry.get("git_sha"),
        "excluded_reason": None,
    }
    splits = entry["resolved_splits_km"]
    rec["band_mass"] = band_mass_fractions(splits, rec["juvenile_mdd_km"], 0.468)
    rec["inner_split_km"] = float(splits[1]) if len(splits) > 2 else float("nan")
    # The radial boundaries are soft sigmoids of width ~2*cell_size, so an inner
    # split below that is only nominally a distinct cohort.
    rec["resolution_suspect"] = bool(0 < rec["inner_split_km"] < 2.0 * cell_km)

    meta_path = Path(entry["path_features_dir"]) / "path_feature_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        rec["realized_discrete_mdd_km"] = meta.get("realized_discrete_mdd_km")
        rec["kernel_mass"] = meta.get("kernel_mass")
        rec["kernel_count"] = meta.get("kernel_count")
        recorded = (meta.get("dispersal") or {}).get("juvenile_radial_splits_km")
        # Independent re-check of the ingest guard: if these ever disagree, the
        # fit was run against path features built under a different spec.
        rec["spec_matches_recorded"] = recorded == splits
    else:
        rec["spec_matches_recorded"] = None

    run = Path(rec["run_dir"])
    ckpt = run / "map_checkpoint.pkl"
    if not ckpt.exists():
        rec["excluded_reason"] = "no map_checkpoint.pkl (fit incomplete)"
        return rec
    with open(ckpt, "rb") as fh:
        chk = pickle.load(fh)
    losses = np.asarray(chk.get("losses", []), dtype=float)
    rec["step"] = int(chk.get("step", len(losses)))
    if losses.size:
        rec["final_loss"] = float(losses[-1])
        tail = losses[-min(100, losses.size):]
        rec["loss_last100_mean"] = float(tail.mean())
        rec["loss_last100_range"] = float(tail.max() - tail.min())
        if tail.size > 2:  # residual descent = lower bound on "what is noise"
            rec["loss_last100_slope_per_step"] = float(
                np.polyfit(np.arange(tail.size), tail, 1)[0])

    fields = run / "map_diagnostics" / "07_source_sink_fields.npz"
    if fields.exists():
        with np.load(fields) as f:
            rec["_lam_realized"] = f["lam_realized_modern"]
            rec["_lam_fundamental"] = f["lam_fundamental_modern"]
            rec["_source_mask"] = f["source_mask"]
            rec["window_years"] = [int(x) for x in f["window_years"]]
    else:
        rec["excluded_reason"] = "no 07_source_sink_fields.npz (rerun map viz)"

    metrics_path = run / "map_diagnostics" / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        fit = m.get("fit") or {}
        rec.update({
            "n_observations": fit.get("n_observations"),
            "log1p_rmse": fit.get("log1p_rmse"),
            "log1p_correlation": fit.get("log1p_correlation"),
            "allee_n50_bbs_count": m.get("allee_n50_bbs_count"),
            "modern_mean_lambda": m.get("modern_mean_lambda"),
            "modern_suitable_fraction": m.get("modern_suitable_fraction"),
            "realized_modern_mean_lambda": (m.get("realized_source_sink") or {})
                .get("realized_modern_mean_lambda"),
            "realized_modern_source_fraction": (m.get("realized_source_sink") or {})
                .get("realized_modern_source_fraction"),
            "disease_severity_median": (m.get("disease") or {})
                .get("disease_severity_median"),
        })
        # Great Plains crossing. `barrier_crossing` is null when the zone raster was
        # unavailable at diagnostics time, so every key here can legitimately be None
        # and the plotting side must tolerate an all-None column rather than assume
        # the sweep was run with it enabled.
        bc = m.get("barrier_crossing") or {}
        rec.update({
            "G_horizon_east_to_west": bc.get("G_horizon_east_to_west"),
            "G_horizon_west_to_east": bc.get("G_horizon_west_to_east"),
            "crossing_asymmetry_ratio": bc.get("asymmetry_ratio_ew_over_we"),
            "rho_barrier": bc.get("rho_barrier"),
            "barrier_self_sustaining": bc.get("barrier_self_sustaining"),
            "median_propagule_pressure_east_to_west":
                bc.get("median_propagule_pressure_east_to_west"),
            "mean_juvenile_edge_correction_in_barrier":
                bc.get("mean_juvenile_edge_correction_in_barrier"),
            # Always populated, even when G/P are suppressed at rho >= 1 -- these are
            # the raw fitted Q contrast and the corridor-shape check, neither of which
            # depends on the diverging sum.
            "mean_q_to_east_in_barrier": bc.get("mean_q_to_east_in_barrier"),
            "mean_q_to_west_in_barrier": bc.get("mean_q_to_west_in_barrier"),
            "corridor_shape_asymmetry_max_log2":
                bc.get("corridor_shape_asymmetry_max_log2"),
            "barrier_crossing_verdict": bc.get("barrier_crossing_verdict"),
        })
    elif rec["excluded_reason"] is None:
        rec["excluded_reason"] = "no metrics.json"
    return rec


def cross_check(points):
    """Mark points that cannot be compared to the others."""
    obs = {p.get("n_observations") for p in points if p.get("n_observations")}
    if len(obs) > 1:
        for p in points:
            p["excluded_reason"] = (p["excluded_reason"]
                                    or f"n_observations differs across points {sorted(obs)}")
    steps = {p.get("step") for p in points if p.get("step")}
    if len(steps) > 1:
        for p in points:
            if p.get("step") and p["step"] != max(steps):
                p["excluded_reason"] = (p["excluded_reason"]
                                        or f"step {p['step']} != {max(steps)} (unequal budget)")
    shas = {p.get("git_sha") for p in points if p.get("git_sha")}
    if len(shas) > 1:
        for p in points:
            p["excluded_reason"] = p["excluded_reason"] or "git sha differs across points"
    return points


#: Row keys ``plot_source_sink`` can draw, in canonical order.
SOURCE_SINK_ROWS = ("class", "lambda", "dlambda", "flip")


def plot_source_sink(usable, ref_point, out, rows=SOURCE_SINK_ROWS):
    """Small multiples + differences against the reference point.

    ``rows`` selects which of :data:`SOURCE_SINK_ROWS` to draw, so the same
    computation can produce both the full four-row diagnostic and a compact
    decision-only version (``("class", "flip")``): the classification itself and
    which cells the dispersal hyperparameter flips. The continuous rows are the
    ones that invite over-reading -- a large Δλ far from λ=1 changes no
    conclusion -- so the compact pair is often the honest summary.
    """
    ref = next((p for p in usable if p["point"] == ref_point), usable[-1])
    n = len(usable)
    rows = tuple(r for r in SOURCE_SINK_ROWS if r in rows)
    if not rows:
        raise ValueError(f"rows must name at least one of {SOURCE_SINK_ROWS}")
    r_of = {name: i for i, name in enumerate(rows)}
    # Size the figure to the RASTER's aspect rather than a fixed per-row height.
    # imshow preserves aspect, so a row taller than the map just pads it with dead
    # space above and below -- which is what left large gaps between rows.
    ny, nx = usable[0]["_lam_realized"].shape
    panel_w = 3.1
    panel_h = panel_w * (ny / nx)
    # Header room for the per-column titles and the suptitle, plus a strip for the
    # flip row's per-panel xlabel when it is drawn.
    extra = 0.85 + (0.30 if "flip" in r_of else 0.0)
    fig, ax = plt.subplots(len(rows), n,
                           figsize=(panel_w * n, panel_h * len(rows) + extra),
                           squeeze=False)
    lam_all = np.concatenate([p["_lam_realized"][np.isfinite(p["_lam_realized"])]
                              for p in usable])
    lo, hi = np.nanpercentile(lam_all, [2, 98])
    lo, hi = min(lo, 1.0), max(hi, 1.0)
    # One Δλ scale for the whole row. Per-panel scaling would make each map look
    # equally dramatic and would contradict the single shared colorbar.
    diffs = [p["_lam_realized"] - ref["_lam_realized"] for p in usable]
    finite = np.concatenate([d[np.isfinite(d)] for d in diffs])
    dlim = max(float(np.nanpercentile(np.abs(finite), 98)), 0.02) if finite.size else 0.02
    ss_cmap = mcolors.ListedColormap(["#d73027", "#4575b4"])
    flip_cmap = mcolors.ListedColormap(["#d73027", "#f7f7f7", "#4575b4"])
    flip_norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], flip_cmap.N)
    im1 = im2 = None

    for j, p in enumerate(usable):
        lam, mask = p["_lam_realized"], p["_source_mask"]
        binary = np.where(np.isfinite(lam), mask.astype(float), np.nan)
        if "class" in r_of:
            a = ax[r_of["class"], j]
            a.imshow(binary, cmap=ss_cmap, vmin=0, vmax=1)
            a.set_title(f"{p['juvenile_mdd_km']:.0f} km\nsource {np.nanmean(binary):.1%}",
                        fontsize=9)
        if "lambda" in r_of:
            a = ax[r_of["lambda"], j]
            im1 = a.imshow(lam, cmap="RdYlBu_r", vmin=lo, vmax=hi)
            a.contour(np.nan_to_num(lam, nan=0.0), [1.0], colors="black", linewidths=0.6)
        if "dlambda" in r_of:
            im2 = ax[r_of["dlambda"], j].imshow(diffs[j], cmap="PuOr_r", vmin=-dlim, vmax=dlim)

        # Magnitude (Δλ) and decision change (flip) answer different questions: a
        # large Δλ well away from λ=1 changes no conclusion, while a tiny one at the
        # boundary flips a cell from source to sink. The flip counts are recorded
        # even when the flip row is not drawn, so the CSV never depends on layout.
        flip = np.where(np.isfinite(lam),
                        mask.astype(int) - ref["_source_mask"].astype(int), np.nan)
        gained = int(np.nansum(flip > 0)); lost = int(np.nansum(flip < 0))
        p["cells_gained_source_vs_ref"], p["cells_lost_source_vs_ref"] = gained, lost
        p["cells_flipped_vs_ref"] = gained + lost
        if "flip" in r_of:
            a = ax[r_of["flip"], j]
            a.imshow(flip, cmap=flip_cmap, norm=flip_norm)
            a.set_xlabel(f"+{gained} / −{lost} source cells", fontsize=8)
            if "class" not in r_of:
                a.set_title(f"{p['juvenile_mdd_km']:.0f} km", fontsize=9)
        if j == n - 1:
            if im1 is not None:
                fig.colorbar(im1, ax=ax[r_of["lambda"], :].tolist(), fraction=.02,
                             label="λ_realized")
            if im2 is not None:
                fig.colorbar(im2, ax=ax[r_of["dlambda"], :].tolist(), fraction=.02,
                             label="Δλ_realized")

    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    ref_km = f"{ref['juvenile_mdd_km']:.0f} km"
    labels = {"class": "Source / sink", "lambda": "Realized λ",
              "dlambda": f"Δλ vs {ref_km}", "flip": f"Class flips vs {ref_km}"}
    for name, i in r_of.items():
        ax[i, 0].set_ylabel(labels[name], fontsize=9)
    # source_mask is the fold criterion (a positive equilibrium exists), not λ>1 --
    # see map_diagnostics._write_source_sink_fields and age_model_math.allee_viability.
    if "class" in r_of:
        ax[r_of["class"], 0].legend(handles=[
            plt.Rectangle((0, 0), 1, 1, color="#4575b4", label="Viable (max$_N$ λ > 1)"),
            plt.Rectangle((0, 0), 1, 1, color="#d73027", label="Non-viable"),
        ], loc="lower left", fontsize=6, frameon=True)
    if "flip" in r_of:
        ax[r_of["flip"], 0].legend(handles=[
            plt.Rectangle((0, 0), 1, 1, color="#4575b4", label="became source"),
            plt.Rectangle((0, 0), 1, 1, color="#d73027", label="became sink"),
        ], loc="lower left", fontsize=6, frameon=True)
    fig.suptitle("Realized source/sink structure across juvenile mean dispersal distance", y=.995)
    # Tight inter-panel spacing: these are maps on a shared frame, so the gaps carry
    # no information and only cost legibility when projected. Set explicitly rather
    # than via tight_layout, which re-pads around titles and colorbars.
    fig.subplots_adjust(left=.035, right=.985 if im1 is None and im2 is None else .93,
                        top=1.0 - (0.62 / fig.get_figheight()),
                        bottom=0.10 if "flip" in r_of else 0.02,
                        wspace=.02, hspace=.06)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return ref


def agreement_matrix(usable):
    """Fraction of land cells whose source/sink class agrees, for every pair."""
    n = len(usable)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(usable):
        for j, b in enumerate(usable):
            valid = np.isfinite(a["_lam_realized"]) & np.isfinite(b["_lam_realized"])
            if valid.any():
                mat[i, j] = float((a["_source_mask"][valid] == b["_source_mask"][valid]).mean())
    return mat


def _plot_barrier_crossing_vs_mdd(ax_g, ax_r, usable, mdd):
    """Great Plains crossing gain and its asymmetry against juvenile MDD.

    THE POINT OF THE WHOLE MEASURE. Kernel mass in the long-distance band is a
    different object at every sweep point and so cannot be compared across them;
    ``G`` is in units of expected descendants and internally re-optimizes between
    "one long jump" and "many short hops that reproduce inside the Plains", so it
    can. If ``G`` moves with MDD while the ASYMMETRY RATIO stays flat, the
    directional conclusion is robust to the dispersal hyperparameter even though the
    absolute crossing rate is not -- which is the result the sweep exists to
    establish.

    TWO WAYS ``G`` CAN BE ABSENT, and they must not be reported the same way:

    * the diagnostic never ran (no zone raster) -- nothing at all is available;
    * it ran and found ``rho >= 1``, so the Neumann series diverges and
      ``map_diagnostics`` deliberately reports ``G`` as null. Here ``rho`` IS
      available and is the headline result, and the directional signal survives in
      the raw ``Q`` contrast. Falling back to "zone raster absent" in this case
      would both misstate the cause and discard the sweep's actual finding.
    """
    gew = np.array([p.get("G_horizon_east_to_west") or np.nan for p in usable], dtype=float)
    gwe = np.array([p.get("G_horizon_west_to_east") or np.nan for p in usable], dtype=float)
    ratio = np.array([p.get("crossing_asymmetry_ratio") or np.nan for p in usable], dtype=float)
    rho = np.array([p.get("rho_barrier") or np.nan for p in usable], dtype=float)
    dq = np.array([(p.get("mean_q_to_west_in_barrier") or np.nan)
                   - (p.get("mean_q_to_east_in_barrier") or np.nan) for p in usable],
                  dtype=float)
    if not np.isfinite(gew).any() and not np.isfinite(rho).any():
        for a, msg in ((ax_g, "no barrier_crossing metrics\n(zone raster absent)"),
                       (ax_r, "")):
            a.text(.5, .5, msg, ha="center", va="center", fontsize=9, transform=a.transAxes)
            a.set_xticks([]); a.set_yticks([])
        return

    if not np.isfinite(gew).any():
        # Supercritical at every point: plot what is defined (rho) and what still
        # carries direction (the raw Q contrast), and say why G is missing.
        ax_g.plot(mdd, rho, marker="^", color="#2166ac")
        ax_g.axhline(1.0, color="#d73027", ls="--", lw=1.0)
        ax_g.set(xlabel="Juvenile mean dispersal distance (km)", ylabel="ρ(barrier)",
                 title="Barrier spectral radius\nG undefined: ρ ≥ 1 at every point")
        ax_g.text(.5, .06, "ρ ≥ 1 ⇒ the linearized barrier self-sustains,\n"
                           "so the crossing gain diverges and is not reported",
                  ha="center", fontsize=7.5, color="0.35", transform=ax_g.transAxes)
        ax_r.plot(mdd, dq, marker="o", color="#4d9221")
        ax_r.axhline(0.0, color="0.6", ls="--", lw=.8)
        ax_r.set(xlabel="Juvenile mean dispersal distance (km)",
                 ylabel="mean Q(→W) − Q(→E) in barrier",
                 title="Journey-survival asymmetry\n(the directional signal that survives ρ ≥ 1)")
        return

    ax_g.plot(mdd, gew, marker="o", color="#54278f", label="east → west")
    ax_g.plot(mdd, gwe, marker="s", color="#e08214", label="west → east")
    ax_g.set(xlabel="Juvenile mean dispersal distance (km)",
             ylabel="G (descendants per founder, 124 yr)",
             title="Great Plains crossing gain", yscale="log")
    ax_g.legend(fontsize=8)

    ax_r.plot(mdd, ratio, marker="o", color="#238443")
    ax_r.axhline(1.0, color="0.6", ls="--", lw=.8)
    ax_r.set(xlabel="Juvenile mean dispersal distance (km)",
             ylabel="G(E→W) / G(W→E)", title="Crossing asymmetry\n(flat = robust to MDD)")
    twin = ax_r.twinx()
    # rho on the same panel because it is the validity condition for G(inf): at rho>=1
    # the barrier self-sustains and the infinite-horizon gain diverges.
    twin.plot(mdd, rho, marker="^", ls=":", color="#2166ac", label="ρ(barrier)")
    twin.axhline(1.0, color="#2166ac", ls=":", lw=.6)
    twin.set_ylabel("ρ(barrier)", color="#2166ac")
    twin.tick_params(axis="y", labelcolor="#2166ac")
    if np.nanmax(ratio) - np.nanmin(ratio) < 0.1 * np.nanmean(ratio):
        # Placed high in the axes: the rho series occupies the lower band.
        ax_r.text(.5, .93, "ratio varies <10% across MDD", ha="center", fontsize=7.5,
                  color="#238443", transform=ax_r.transAxes)


def plot_fit_metrics(usable, mat, out):
    labels = [f"{p['juvenile_mdd_km']:.0f}" for p in usable]
    mdd = [p["juvenile_mdd_km"] for p in usable]
    fig, ax = plt.subplots(2, 3, figsize=(18.0, 8.5))

    loss = [p.get("final_loss", np.nan) for p in usable]
    # Error bars are the last-100-step spread: a lower bound on optimizer noise,
    # since no repeat fits were run. Differences inside it are not attributable.
    err = [p.get("loss_last100_range", 0.0) / 2 for p in usable]
    ax[0, 0].errorbar(mdd, loss, yerr=err, marker="o", color="#54278f", capsize=3)
    ax[0, 0].set(xlabel="Juvenile mean dispersal distance (km)", ylabel="Final MAP loss",
                 title="MAP objective\n(bars = last-100-step spread ≈ noise floor)")

    ax[0, 1].plot(mdd, [p.get("log1p_rmse", np.nan) for p in usable],
                  marker="o", color="#d94801", label="log1p RMSE")
    ax[0, 1].set(xlabel="mdd (km)", ylabel="log1p RMSE")
    twin = ax[0, 1].twinx()
    twin.plot(mdd, [p.get("log1p_correlation", np.nan) for p in usable],
              marker="s", color="#238443", label="log1p correlation")
    twin.set_ylabel("log1p correlation")
    ax[0, 1].set_title("BBS fit quality (observation scale)")
    lines = ax[0, 1].get_lines() + twin.get_lines()
    ax[0, 1].legend(lines, [l.get_label() for l in lines], fontsize=8, loc="best")

    real = [p.get("realized_discrete_mdd_km", np.nan) for p in usable]
    ax[1, 0].plot(mdd, mdd, ls="--", color="0.6", label="1:1")
    ax[1, 0].plot(mdd, real, marker="o", color="#08519c", label="realized (discrete)")
    ax[1, 0].set(xlabel="Target mdd (km)", ylabel="Realized mdd (km)",
                 title="Lattice truncation: realized < target")
    ax[1, 0].legend(fontsize=8)

    im = ax[1, 1].imshow(mat, cmap="viridis", vmin=np.nanmin(mat), vmax=1.0)
    ax[1, 1].set(xticks=range(len(labels)), yticks=range(len(labels)),
                 xticklabels=labels, yticklabels=labels,
                 title="Source/sink classification agreement")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(mat[i, j]):
                ax[1, 1].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                              fontsize=7, color="white" if mat[i, j] < 0.9 else "black")
    fig.colorbar(im, ax=ax[1, 1], fraction=.046, label="Fraction of land cells agreeing")

    _plot_barrier_crossing_vs_mdd(ax[0, 2], ax[1, 2], usable, mdd)

    fig.suptitle("Juvenile dispersal sensitivity: fit quality, kernel realization, "
                 "and Great Plains crossing", y=.99)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--reference-point", default=None,
                    help="Point to difference against (default: the 330 km point if present).")
    ap.add_argument("--out", default=None, help="Output dir (default: beside the manifest).")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    out = Path(args.out) if args.out else Path(args.manifest).parent
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_age_model_config()

    points = cross_check([load_point(e, cfg["results_dir"]) for e in manifest["points"]])
    usable = [p for p in points if p["excluded_reason"] is None and "_lam_realized" in p]
    usable.sort(key=lambda p: p["juvenile_mdd_km"])

    ref_point = args.reference_point
    if ref_point is None:
        ref_point = next((p["point"] for p in usable
                          if abs(p["juvenile_mdd_km"] - 330.0) < 1e-6),
                         usable[-1]["point"] if usable else None)

    print(f"\n=== {manifest.get('sweep')} : {len(usable)}/{len(points)} points usable ===")
    hdr = f"{'point':>8} {'mdd':>6} {'inner':>7} {'realized':>9} {'loss':>12} {'rmse':>7} {'corr':>6} {'source%':>8}"
    print(hdr); print("-" * len(hdr))
    for p in points:
        if p["excluded_reason"]:
            print(f"{p['point']:>8} {p['juvenile_mdd_km']:>6.0f}   EXCLUDED: {p['excluded_reason']}")
            continue
        print(f"{p['point']:>8} {p['juvenile_mdd_km']:>6.0f} {p['inner_split_km']:>7.1f} "
              f"{p.get('realized_discrete_mdd_km') or float('nan'):>9.1f} "
              f"{p.get('final_loss', float('nan')):>12.1f} "
              f"{p.get('log1p_rmse', float('nan')):>7.4f} "
              f"{p.get('log1p_correlation', float('nan')):>6.3f} "
              f"{100 * (p.get('realized_modern_source_fraction') or float('nan')):>7.1f}%"
              + ("  [inner cohort under-resolved]" if p["resolution_suspect"] else "")
              + ("  [SPEC MISMATCH]" if p.get("spec_matches_recorded") is False else ""))

    if usable:
        ref = plot_source_sink(usable, ref_point, out / "juv_mdd_source_sink.png")
        # Decision-only companion: the classification and what the dispersal
        # hyperparameter actually changes about it, without the continuous rows.
        plot_source_sink(usable, ref_point, out / "juv_mdd_source_sink_flips.png",
                         rows=("class", "flip"))
        mat = agreement_matrix(usable)
        plot_fit_metrics(usable, mat, out / "juv_mdd_fit_metrics.png")
        off = mat[~np.eye(len(usable), dtype=bool)]
        print(f"\nsource/sink agreement vs {ref['juvenile_mdd_km']:.0f} km reference: "
              f"pairwise min {np.nanmin(off):.3f}, median {np.nanmedian(off):.3f}")
        spread = np.nanmax([p.get("final_loss", np.nan) for p in usable]) - \
                 np.nanmin([p.get("final_loss", np.nan) for p in usable])
        floor = np.nanmax([p.get("loss_last100_range", np.nan) for p in usable])
        print(f"MAP loss spread across points {spread:.1f} vs within-run last-100 spread "
              f"{floor:.1f}"
              + ("  -> spread is WITHIN the noise floor; do not rank points on loss"
                 if np.isfinite(floor) and spread <= floor else ""))
        print("Caveat: MAP loss is an optimum, not a marginal likelihood, and no repeat "
              "fits were run, so treat loss ordering as directional only.")
    else:
        print("\nNo usable points yet -- nothing plotted.")

    for p in points:
        p.pop("_lam_realized", None); p.pop("_lam_fundamental", None); p.pop("_source_mask", None)
    with open(out / "sweep_summary.json", "w") as fh:
        json.dump({"manifest": manifest, "reference_point": ref_point, "points": points},
                  fh, indent=2, default=str)
    cols = sorted({k for p in points for k in p})
    with open(out / "sweep_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for p in points:
            w.writerow({k: p.get(k) for k in cols})
    figs = " and 3 figures" if usable else " (no figures -- no usable points)"
    print(f"\nwrote {out}/sweep_summary.csv, sweep_summary.json{figs}")


if __name__ == "__main__":
    main()

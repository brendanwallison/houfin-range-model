#!/usr/bin/env python3
"""Compare MAP runs across a juvenile-dispersal-distance sweep.

Reads the artifacts each sweep point already produced -- never re-runs a model, so
this is CPU-only and takes seconds. Input is the manifest written by
``scripts/tacc/submit_juv_mdd_sweep.sh``; points whose fit is missing or
incomparable are marked excluded rather than silently plotted.

The primary comparison is the **realized source/sink raster** (from each run's
``map_diagnostics/07_source_sink_fields.npz``): small multiples across mdd, signed
lambda differences against a reference point, maps of cells that flip
source<->sink, and a pairwise classification-agreement matrix. That last number is
the compact answer to "how much does the source/sink conclusion actually move with
dispersal distance."

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
            "disease_depression_median": (m.get("spatiotemporal_diagnostics") or {})
                .get("disease_depression_median"),
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


def plot_source_sink(usable, ref_point, out):
    """Small multiples + differences against the reference point."""
    ref = next((p for p in usable if p["point"] == ref_point), usable[-1])
    n = len(usable)
    fig, ax = plt.subplots(4, n, figsize=(3.1 * n, 10.8), squeeze=False)
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

    for j, p in enumerate(usable):
        lam, mask = p["_lam_realized"], p["_source_mask"]
        binary = np.where(np.isfinite(lam), mask.astype(float), np.nan)
        ax[0, j].imshow(binary, cmap=ss_cmap, vmin=0, vmax=1)
        ax[0, j].set_title(f"{p['juvenile_mdd_km']:.0f} km\nsource {np.nanmean(binary):.1%}",
                           fontsize=9)
        im1 = ax[1, j].imshow(lam, cmap="RdYlBu_r", vmin=lo, vmax=hi)
        ax[1, j].contour(np.nan_to_num(lam, nan=0.0), [1.0], colors="black", linewidths=0.6)

        im2 = ax[2, j].imshow(diffs[j], cmap="PuOr_r", vmin=-dlim, vmax=dlim)

        # Magnitude (Δλ) and decision change (flip) answer different questions: a
        # large Δλ well away from λ=1 changes no conclusion, while a tiny one at the
        # boundary flips a cell from source to sink. Show both.
        flip = np.where(np.isfinite(lam),
                        mask.astype(int) - ref["_source_mask"].astype(int), np.nan)
        gained = int(np.nansum(flip > 0)); lost = int(np.nansum(flip < 0))
        p["cells_gained_source_vs_ref"], p["cells_lost_source_vs_ref"] = gained, lost
        p["cells_flipped_vs_ref"] = gained + lost
        ax[3, j].imshow(flip, cmap=flip_cmap, norm=flip_norm)
        ax[3, j].set_xlabel(f"+{gained} / −{lost} source cells", fontsize=8)
        if j == n - 1:
            fig.colorbar(im1, ax=ax[1, :].tolist(), fraction=.02, label="λ_realized")
            fig.colorbar(im2, ax=ax[2, :].tolist(), fraction=.02, label="Δλ_realized")

    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    ref_km = f"{ref['juvenile_mdd_km']:.0f} km"
    ax[0, 0].set_ylabel("Source / sink", fontsize=9)
    ax[1, 0].set_ylabel("Realized λ", fontsize=9)
    ax[2, 0].set_ylabel(f"Δλ vs {ref_km}", fontsize=9)
    ax[3, 0].set_ylabel(f"Class flips vs {ref_km}", fontsize=9)
    ax[0, 0].legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#4575b4", label="Source (λ>1)"),
        plt.Rectangle((0, 0), 1, 1, color="#d73027", label="Sink (λ≤1)"),
    ], loc="lower left", fontsize=6, frameon=True)
    ax[3, 0].legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#4575b4", label="became source"),
        plt.Rectangle((0, 0), 1, 1, color="#d73027", label="became sink"),
    ], loc="lower left", fontsize=6, frameon=True)
    fig.suptitle("Realized source/sink structure across juvenile mean dispersal distance", y=.99)
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
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


def plot_fit_metrics(usable, mat, out):
    labels = [f"{p['juvenile_mdd_km']:.0f}" for p in usable]
    mdd = [p["juvenile_mdd_km"] for p in usable]
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.5))

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

    fig.suptitle("Juvenile dispersal sensitivity: fit quality and kernel realization", y=.99)
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
    figs = " and 2 figures" if usable else " (no figures -- no usable points)"
    print(f"\nwrote {out}/sweep_summary.csv, sweep_summary.json{figs}")


if __name__ == "__main__":
    main()

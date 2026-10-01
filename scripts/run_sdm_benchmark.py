#!/usr/bin/env python3
"""Correlative-SDM benchmark against the dynamic range model.

The benchmark compares DESIGNATIONS, not map appearance: the dynamic model's
``lam_fundamental >= 1`` (can a population sustain itself here WITHOUT
immigration) against a correlative SDM's thresholded probability surface (a
suitability call inferred FROM occurrence). A correlative model cannot populate
the "occupied sink" cell of the niche x occupancy 2x2 other than by threshold
noise, because it infers the niche axis from the occupancy axis (Pulliam 2000).

Subcommands
-----------
  preflight    Check every input exists and is on the model grid. Fits nothing,
               exits nonzero if not. Cheap enough for a TACC LOGIN NODE -- it
               samples years and verifies geometry once per source, so it costs
               a few hundred stats, not thousands of raster header reads.
               Run this BEFORE submitting anything.

EVERYTHING ELSE BELONGS IN A BATCH JOB. `brt` fits 5-fold LightGBM over ~64k
route-years and `biolith` runs NUTS; neither should touch a login node. Use
scripts/tacc/submit_sdm_benchmark.sh.
  premise      Observed BBS abundance/occupancy by Great Plains zone and latitude
               band. Run this first: it is the data check the whole design rests
               on, and it needs no model and no covariates.
  designation  The dynamic model's own niche x occupancy 2x2, per zone.
  brt          Fit the boosted-regression-tree baselines (needs covariates).
  biolith      Fit occu() and nmixture() (needs covariates).

NOT YET IMPLEMENTED: the collation/figure stage (Test D, the longitude transect
faceted by latitude band) and the environment-vs-space contrast (Test B) beyond
the ``--spatial`` flag on ``biolith``, which is wired but has not been run.

COVARIATES ARE HPC-ONLY. The climate / LUH-3 / encoder-state / latent-Z grids are
built on TACC; a laptop checkout carries only stale 25 km ESRI:102039 leftovers,
which the covariate loader refuses by geometry rather than sampling silently. So
``premise`` and ``designation`` run anywhere; ``brt`` and ``biolith`` need the
processed tree.

    python scripts/run_sdm_benchmark.py premise
    python scripts/run_sdm_benchmark.py designation --run-dir <model_results/RUN>
    python scripts/run_sdm_benchmark.py brt --tier standard --out results/sdm
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.sdm_benchmark import baselines, covariates, data, designation  # noqa: E402

ZONE_LABELS = {1: "west", 2: "great_plains", 3: "east"}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _write(out_dir, name, payload):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    with open(p, "w") as fh:
        json.dump(payload, fh, indent=2, default=_jsonable)
    print(f"  wrote {p}")
    return p


def cmd_preflight(args):
    """Can this machine actually run the benchmark? Checks, never fits.

    Exits nonzero when something needed is absent, so a four-hour job is not how
    you discover an unbuilt grid directory.
    """
    ok = True
    print("\n-- BBS response --")
    try:
        df = data.route_years(args.start_year, args.end_year)
        routes = df[["CountryNum", "StateNum", "Route"]].drop_duplicates()
        print(f"   OK   {len(df):,} route-years, {len(routes):,} routes, "
              f"{args.start_year}-{args.end_year}, "
              f"{100*(df['count'] > 0).mean():.1f}% with detections")
        _s, counts, years = data.site_replicates(df, args.rep_start, args.rep_end)
        print(f"   OK   replicate matrix {counts.shape}, years {years}")
    except Exception as e:
        ok = False
        print(f"   FAIL {type(e).__name__}: {e}")
        df = None

    print("\n-- covariate tiers --")
    if df is not None:
        yrs = sorted(df["Year"].unique().tolist())
        # Only the tiers the job will actually run. An unbuilt tier must not
        # block a submission that never asks for it.
        for tier in args.tiers:
            try:
                r = covariates.probe_tier(yrs, tier)
            except Exception as e:
                ok = False
                print(f"   FAIL {tier:9s} {type(e).__name__}: {e}")
                continue
            if r["available"]:
                print(f"   OK   {tier:9s} {r['n_features']:4d} features")
                continue
            # Covariate streams end before BBS does, so distinguish "this tier
            # was never built" from "it just stops earlier than the window".
            last = covariates.last_complete_year(yrs, tier)
            if last is not None and last < max(yrs):
                print(f"   WARN {tier:9s} complete only through {last} "
                      f"(window asks for {max(yrs)}); will clamp to {last}")
            else:
                ok = False
                print(f"   FAIL {tier:9s} unavailable")
                for m in (r.get("first_missing") or [])[:3]:
                    print(f"          {m}")

    print("\n-- dynamic model (needed only for `designation`) --")
    if args.run_dir:
        try:
            f = designation.load_dynamic_fields(args.run_dir)
            lam = f["lam_fundamental_modern"]
            fin = np.isfinite(lam)
            print(f"   OK   lam_fundamental: {int(fin.sum())} finite cells, "
                  f"{100*(lam[fin] >= 1).mean():.1f}% >= 1, "
                  f"window={f.get('window_years')}")
        except Exception as e:
            ok = False
            print(f"   FAIL {type(e).__name__}: {e}")
    else:
        print("   skip (pass --run-dir to check one)")

    print(f"\n{'READY' if ok else 'NOT READY'}")
    if not ok:
        sys.exit(1)
    return {"ok": ok}


def cmd_premise(args):
    """Observed structure, before any model touches it."""
    df = data.attach_zones(data.route_years(args.start_year, args.end_year))
    band = data.plains_band(df)
    out = {"window": [args.start_year, args.end_year],
           "lat_band": [data.PLAINS_LAT_MIN, data.PLAINS_LAT_MAX],
           "n_route_years": int(len(df)),
           "closure_check": data.closure_check(df, args.rep_start, args.rep_end),
           "by_zone": {}, "by_lat_band": {}}

    print(f"\nObserved BBS {args.start_year}-{args.end_year}, "
          f"{data.PLAINS_LAT_MIN:.0f}-{data.PLAINS_LAT_MAX:.0f}N")
    print(f"{'zone':14s} {'n':>7s} {'mean':>7s} {'occ%':>6s}")
    for z, lab in ZONE_LABELS.items():
        s = band[band["zone"] == z]["count"]
        if not len(s):
            continue
        rec = {"n": int(len(s)), "mean_count": float(s.mean()),
               "occupancy": float((s > 0).mean())}
        out["by_zone"][lab] = rec
        print(f"{lab:14s} {rec['n']:7,d} {rec['mean_count']:7.2f} "
              f"{100*rec['occupancy']:6.1f}")

    # The depression is a mid-latitude phenomenon; show the bands so the
    # restriction to PLAINS_LAT_MIN..MAX is visible rather than asserted.
    d2 = df.copy()
    d2["latband"] = (d2["Latitude"] // 4 * 4).astype(int)
    for lb, g in d2.groupby("latband"):
        if lb < 24 or lb > 52:
            continue
        rec = {}
        for z, lab in ZONE_LABELS.items():
            s = g[g["zone"] == z]["count"]
            if len(s) >= 30:
                rec[lab] = {"n": int(len(s)), "mean_count": round(float(s.mean()), 3),
                            "occupancy": round(float((s > 0).mean()), 3)}
        if rec:
            out["by_lat_band"][int(lb)] = rec

    cc = out["closure_check"]
    print(f"\nclosure {args.rep_start}-{args.rep_end}: "
          f"{cc['pct_per_year']:+.2f}%/yr  (years-as-replicates assumes ~flat)")
    _write(args.out, "premise.json", out)
    return out


def cmd_designation(args):
    """The dynamic model's niche x occupancy 2x2, per zone."""
    import rasterio
    from src.config_utils import load_data_config

    fields = designation.load_dynamic_fields(args.run_dir)
    lam = fields["lam_fundamental_modern"]
    niche, finite = designation.niche_from_lambda(lam, args.lam_threshold)
    ny, nx = lam.shape

    df = data.plains_band(data.attach_zones(
        data.route_years(args.rep_start, args.rep_end)))
    occ, surv, mean_count = designation.cell_occupancy(df, ny, nx,
                                                       min_routes=args.min_routes)
    with rasterio.open(load_data_config()["regions"]["great_plains_zones"]) as src:
        zones = src.read(1)

    out = {"run_dir": args.run_dir, "lam_threshold": args.lam_threshold,
           "window": [args.rep_start, args.rep_end],
           "window_years_in_fields": fields.get("window_years", None),
           "by_zone": {}}
    print(f"\nDynamic-model designation, run={os.path.basename(args.run_dir)}")
    print(f"{'zone':14s} {'cells':>6s} {'occ%':>6s} {'niche%':>7s} {'OCC_SINK':>9s} {'kappa':>7s}")
    for z, lab in list(ZONE_LABELS.items()) + [(None, "all")]:
        m = finite & surv & ((zones == z) if z else (zones > 0))
        if not m.sum():
            continue
        c = designation.contingency(niche, occ, m)
        c["occupancy"] = float(occ[m].mean())
        c["niche_fraction"] = float(niche[m].mean())
        c["kappa"] = designation.cohens_kappa(niche, occ, m)
        out["by_zone"][lab] = c
        print(f"{lab:14s} {c['n_cells']:6d} {100*c['occupancy']:6.1f} "
              f"{100*c['niche_fraction']:7.1f} {c['occupied_sink_fraction']:9.3f} "
              f"{c['kappa']:+7.3f}")

    gp = out["by_zone"].get("great_plains", {}).get("occupied_sink_fraction")
    w = out["by_zone"].get("west", {}).get("occupied_sink_fraction")
    if gp and w:
        out["plains_vs_west_occupied_sink_ratio"] = round(gp / max(w, 1e-9), 2)
        print(f"\noccupied-sink enrichment, Great Plains vs West: "
              f"{out['plains_vs_west_occupied_sink_ratio']}x")
    _write(args.out, "designation_dynamic.json", out)
    return out


def _clamp_to_covariates(df, tier):
    """Trim the window to the last year the tier actually covers, loudly.

    A batch job should not die because LUH-3 stops in 2024 while BBS reaches
    2025. Clamping is announced, never silent, and refuses outright if no year
    is complete -- that means the tier was never built, which clamping cannot fix.
    """
    yrs = sorted(df["Year"].unique().tolist())
    last = covariates.last_complete_year(yrs, tier)
    if last is None:
        raise FileNotFoundError(
            f"tier {tier!r} has no complete year in {yrs[0]}-{yrs[-1]}; its "
            f"grids have not been built. Run `preflight` for the exact paths.")
    if last < max(yrs):
        print(f"  clamping {tier} window to {yrs[0]}-{last} "
              f"(covariates stop before {max(yrs)})")
        df = df[df["Year"] <= last]
    return df


def _design_for_tier(df, tier):
    if tier == "standard":
        return covariates.build_design(df)
    if tier == "full":
        return covariates.full_states_design(df)
    if tier == "latent":
        return covariates.latent_z_design(df)
    raise ValueError(f"unknown tier {tier!r}")


def _warn_saturation(label, diag):
    """Say plainly when a fit was pinned at the tree-depth limit.

    Saturation means slow AND poorly mixed sampling, so the uncertainty is not
    to be believed even when the point estimates look reasonable.
    """
    if not diag:
        print(f"  note: {label} sampler diagnostics unavailable")
        return
    frac = diag.get("frac_at_max_treedepth")
    if frac is not None and frac > 0.5:
        print(f"  WARNING: {label}: {100*frac:.0f}% of iterations hit max tree "
              f"depth ({diag.get('max_treedepth_steps')} steps) -- badly "
              f"conditioned posterior, treat as unconverged")
    if diag.get("divergences"):
        print(f"  WARNING: {label}: {diag['divergences']} divergent transitions")


def _already_done(args, name):
    """True when this stage's report exists and --skip-existing was passed.

    A resubmit after a wall-clock kill should not redo the stages that already
    succeeded -- the BRT tiers alone are ~25 minutes of the budget.
    """
    p = os.path.join(args.out, name)
    if getattr(args, "skip_existing", False) and os.path.exists(p):
        print(f"  skipping: {p} already exists (--skip-existing)")
        return True
    return False


def cmd_brt(args):
    """Boosted-regression-tree baselines under spatial block CV."""
    if _already_done(args, f"brt_{args.tier}.json"):
        return None
    df = data.attach_zones(data.route_years(args.start_year, args.end_year))
    df = _clamp_to_covariates(df, args.tier).reset_index(drop=True)
    X, names = _design_for_tier(df, args.tier)
    y = df["count"].to_numpy(dtype=float)
    rows, cols = df["row"].to_numpy(), df["col"].to_numpy()

    land, _t, _c, nx, ny = data.load_grid()
    valid = np.zeros((ny, nx), dtype=bool)
    valid[rows, cols] = True

    occ_pred = np.full(len(df), np.nan)
    cnt_pred = np.full(len(df), np.nan)
    folds = list(baselines.spatial_block_folds(
        valid, args.block_cells, args.n_folds, args.buffer_cells, args.seed))
    for k, (tr_grid, va_grid) in enumerate(folds):
        tr = baselines.rows_in(tr_grid, rows, cols)
        va = baselines.rows_in(va_grid, rows, cols)
        if tr.sum() < 100 or va.sum() < 50:
            print(f"  fold {k}: too small (train {tr.sum()}, val {va.sum()}), skipped")
            continue
        Xtr, mu, sd = covariates.standardize(X[tr])
        Xva, _, _ = covariates.standardize(X[va], mu, sd)
        mo, it_o = baselines.fit_occurrence(Xtr, y[tr], Xva, y[va], seed=args.seed)
        mc, it_c = baselines.fit_count(Xtr, y[tr], Xva, y[va], seed=args.seed)
        occ_pred[va] = mo.predict_proba(Xva)[:, 1]
        cnt_pred[va] = mc.predict(Xva)
        print(f"  fold {k}: train={tr.sum():6d} val={va.sum():6d} "
              f"trees(occ/cnt)={it_o}/{it_c}")

    # A final model on ALL rows for MAPPING. The fold models above give honest
    # out-of-fold scores; a map needs one model that has seen everything, which
    # is the standard SDM workflow (CV to evaluate, full fit to predict).
    grid_occ = grid_cnt = None
    if not args.no_grid:
        land, _t2, _c2, _nx2, _ny2 = data.load_grid()
        gdf = covariates.grid_frame(land, int(df["Year"].max()))
        try:
            Xg, _ = _design_for_tier(gdf, args.tier)
        except Exception as e:
            print(f"  grid prediction skipped: {e}")
            Xg = None
        if Xg is not None:
            Xa, mu_a, sd_a = covariates.standardize(X)
            Xga, _, _ = covariates.standardize(Xg, mu_a, sd_a)
            mo_f, _ = baselines.fit_occurrence(Xa, y, Xa, y, seed=args.seed)
            mc_f, _ = baselines.fit_count(Xa, y, Xa, y, seed=args.seed)
            good = ~np.isnan(Xga).any(axis=1)
            po = np.full(len(gdf), np.nan); pc = np.full(len(gdf), np.nan)
            po[good] = mo_f.predict_proba(Xga[good])[:, 1]
            pc[good] = mc_f.predict(Xga[good])
            grid_occ = covariates.design_to_grid(po, gdf.row, gdf.col, (ny, nx))
            grid_cnt = covariates.design_to_grid(pc, gdf.row, gdf.col, (ny, nx))
            print(f"  grid surface: {int(good.sum()):,} of {len(gdf):,} land cells")

    ok = np.isfinite(occ_pred)
    lab = y > 0
    phi = baselines.nb2_concentration_mle(y[ok], np.maximum(cnt_pred[ok], 1e-6))
    thr_sss = designation.threshold_max_sss(occ_pred[ok], lab[ok])
    thr_p10 = designation.threshold_p10(occ_pred[ok], lab[ok])
    out = {"tier": args.tier, "n_features": len(names), "n_scored": int(ok.sum()),
           "auc": designation.auc(occ_pred[ok], lab[ok]),
           "tss_maxsss": designation.tss(occ_pred[ok], lab[ok], thr_sss),
           "threshold_maxsss": thr_sss, "threshold_p10": thr_p10,
           "nb2_concentration": phi,
           "nb2_deviance_explained": baselines.nb2_deviance_explained(
               y[ok], np.maximum(cnt_pred[ok], 1e-6), phi),
           "cv": {"block_cells": args.block_cells, "n_folds": args.n_folds,
                  "buffer_cells": args.buffer_cells, "seed": args.seed}}
    print(f"\nBRT[{args.tier}] AUC={out['auc']:.4f} TSS={out['tss_maxsss']:.4f} "
          f"NB2 phi={phi:.3f} dev.expl={out['nb2_deviance_explained']:.4f}")
    _write(args.out, f"brt_{args.tier}.json", out)
    payload = dict(occ_pred=occ_pred, cnt_pred=cnt_pred, y=y,
                   row=rows, col=cols, year=df["Year"].to_numpy(),
                   zone=df["zone"].to_numpy())
    if grid_occ is not None:
        payload["grid_occ"] = grid_occ.astype(np.float32)
        payload["grid_cnt"] = grid_cnt.astype(np.float32)
    np.savez_compressed(os.path.join(args.out, f"brt_{args.tier}_pred.npz"), **payload)
    return out


def _biolith_grid_psi(args, occupancy, covariates, data, X_sites, pca_info,
                      cov_year, mu=None, sd=None):
    """Posterior-mean psi on EVERY land cell, from the fitted occupancy model.

    Uses biolith's own ``predict`` with the fitted MCMC rather than
    reconstructing the linear predictor by hand, so the coefficient layout and
    any link/offset stay biolith's business.

    The PCA rotation AND the standardization are the ones FITTED ON THE SITES,
    reapplied to the grid. Refitting either on grid cells expresses the model's
    coefficients in a different basis and silently produces nonsense: with the
    grid standardized to its own mean/sd, psi at SURVEYED cells came back 0.91-
    0.94 where the site predictions for those same cells were 0.82-0.83, and the
    zone ordering inverted. Same model, same cells, different answers -- the
    signature of a transform mismatch, not of a modelling choice.
    """
    import numpy as _np
    from biolith.models import occu
    from biolith.utils import predict as bio_predict

    land, _t, _c, nx, ny = data.load_grid()
    gdf = covariates.grid_frame(land, cov_year)
    try:
        Xg, _names = _design_for_tier(gdf, args.tier)
    except Exception as e:
        print(f"  grid prediction skipped: {e}")
        return None

    if pca_info is not None:
        if args.pca_per_stream:
            Xg, _, _ = covariates.pca_reduce_by_stream(
                Xg, _names, fitted=pca_info["fitted"])
        else:
            Xg, _, _ = covariates.pca_reduce(
                Xg, basis=pca_info["basis"], center=pca_info["center"],
                mu=pca_info["mu"], sd=pca_info["sd"])
    Xg, _, _ = covariates.standardize(Xg, mu, sd)

    good = ~_np.isnan(Xg).any(axis=1)
    if not good.any():
        print("  grid prediction skipped: no land cell has complete covariates")
        return None
    Xg = _np.nan_to_num(Xg)

    # One dummy visit: psi does not depend on the observation layer, but the
    # model signature requires obs_covs shaped like the fit's.
    n = Xg.shape[0]
    oc = _np.zeros((n, 1, 1, 1))
    obs = _np.full((1, n, 1, 1), _np.nan)
    try:
        post = bio_predict(occu, args._mcmc, site_covs=Xg, obs_covs=oc, obs=obs,
                           num_samples=min(200, args.samples))
    except Exception as e:
        print(f"  grid prediction failed: {type(e).__name__}: {e}")
        return None
    psi = _np.asarray(post["psi"])
    psi = psi[..., 0] if psi.ndim == 3 and psi.shape[-1] == 1 else psi
    vals = _np.where(good, psi.mean(axis=0), _np.nan)
    print(f"  grid surface: {int(good.sum()):,} of {n:,} land cells")
    return covariates.design_to_grid(vals, gdf.row, gdf.col, (ny, nx))


def cmd_biolith(args):
    """Detection-corrected occu() and nmixture(), reduced to occupancy."""
    if _already_done(args, f"biolith_{args.tier}.json"):
        return None
    from src.analysis.sdm_benchmark import occupancy

    df = data.attach_zones(data.route_years(args.rep_start, args.rep_end))
    sites, counts, years = data.site_replicates(df, args.rep_start, args.rep_end)

    site_df = df.drop_duplicates(subset=["CountryNum", "StateNum", "Route"])
    site_df = site_df.set_index(["CountryNum", "StateNum", "Route"]).loc[
        list(zip(sites.CountryNum, sites.StateNum, sites.Route))].reset_index()
    # Site covariates are read at ONE year (the window is treated as closed).
    # It must be a year the tier actually covers: LUH-3 and climate stop before
    # BBS does, so rep_end would ask for a raster that was never built.
    cov_year = covariates.last_complete_year(
        list(range(args.rep_start, args.rep_end + 1)), args.tier)
    if cov_year is None:
        raise FileNotFoundError(
            f"tier {args.tier!r} has no complete year in "
            f"{args.rep_start}-{args.rep_end}; run `preflight` for the paths.")
    if cov_year < args.rep_end:
        print(f"  site covariates read at {cov_year} "
              f"(tier {args.tier} does not reach {args.rep_end})")
    site_df["Year"] = cov_year
    out_cov_year = cov_year
    X, names = _design_for_tier(site_df, args.tier)
    cond_raw = covariates.condition_number(X)
    if args.pca:
        # biolith fits a LINEAR model, and the encoder's channels are near
        # duplicates of one another (12 bases x 12 months, plus quantile
        # levels). Rotating to principal components keeps the same information
        # while removing the collinearity that makes the coefficient posterior a
        # ridge -- which is what saturates NUTS's tree depth.
        if args.pca_per_stream:
            # Only sensible alongside a variance budget: it protects the small
            # streams from being truncated away, but leaves the streams
            # collinear WITH EACH OTHER (elevation with temperature, built-up
            # with population), so the design stays ill-conditioned.
            X, names, pca_info = covariates.pca_reduce_by_stream(
                X, names, var_target=args.pca_var)
            print(f"  PCA (per stream): {X.shape[1]} components")
            for st, r in sorted(pca_info["per_stream"].items()):
                print(f"      {st:10s} {r['n_in']:4d} -> {r['n_out']:3d}")
            if not args.pca_var:
                print("      NOTE per-stream without a variance budget leaves "
                      "cross-stream collinearity; the global rotation is better")
        else:
            n_pc = None if args.pca < 0 else int(args.pca)
            X, names, pca_info = covariates.pca_reduce(
                X, n_components=n_pc, var_target=args.pca_var)
            print(f"  PCA (global): {X.shape[1]} components"
                  + ("" if (args.pca_var or n_pc)
                     else "  [rotate only, nothing truncated]"))
            if args.pca_var:
                print("      WARNING a global VARIANCE budget can bury the "
                      "small human-footprint streams; --pca-per-stream protects "
                      "them, or use --pca-select-cv")
        print(f"      condition number {cond_raw:.3g} -> "
              f"{covariates.condition_number(X):.3g}")
        mu = sd = None
        if args.pca_select_cv:
            # Variance is not predictive relevance, so let held-out AUC pick.
            blocks = (sites.row.to_numpy() // 6) * 1000 + (sites.col.to_numpy() // 6)
            y_bin = (np.nanmax(counts, axis=1) > 0).astype(int)
            k, sel = covariates.select_n_components_cv(X, y_bin, groups=blocks)
            print(f"  PCA CV selection: keeping {k} of {X.shape[1]} "
                  f"({sel.get('rule', '')})")
            X, names = X[:, :k], names[:k]
            pca_info["cv_selection"] = sel
        out_pca = {k: v for k, v in pca_info.items() if k != "fitted"}
        print(f"      condition number {cond_raw:.3g} -> "
              f"{covariates.condition_number(X):.3g}")
        mu = sd = None
    else:
        out_pca = None
        print(f"  design condition number {cond_raw:.3g} "
              f"(>1e4 makes a linear fit badly conditioned; try --pca)")
    X, mu, sd = covariates.standardize(X)
    keep = ~np.isnan(X).any(axis=1)
    if not keep.all():
        print(f"  dropping {int((~keep).sum())} sites with NaN covariates")
        sites, counts, X = sites[keep].reset_index(drop=True), counts[keep], X[keep]

    if args.dump_design:
        # The assembled site x covariate matrix is TINY (3853 x ~40 floats,
        # under a megabyte) even though the grids behind it are not. Dumping it
        # lets the sampler be diagnosed off-HPC without moving any raster.
        dp = os.path.join(args.out, f"design_{args.tier}.npz")
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(dp, X=X, names=np.array(names, dtype=object),
                            counts=counts, years=np.array(years),
                            row=sites.row.to_numpy(), col=sites.col.to_numpy(),
                            lon=sites.Longitude.to_numpy(),
                            lat=sites.Latitude.to_numpy(),
                            covariate_year=out_cov_year)
        print(f"  wrote {dp}  ({os.path.getsize(dp)/1e6:.2f} MB)")

    coords = None
    if args.spatial:
        coords = np.column_stack([sites.Longitude, sites.Latitude]).astype(float)
        coords = (coords - coords.mean(0)) / coords.std(0)

    mx, need = occupancy.suggest_max_abundance(counts)
    if args.max_abundance:
        mx = int(args.max_abundance)
    if not args.no_nmixture:
        from src.analysis.sdm_benchmark import nmixture_marginal as _nm
        n_rep = counts.shape[1]
        gib = (occupancy.enumeration_bytes(len(sites), mx) if args.enumerate_nmixture
               else _nm.enumeration_free_bytes(len(sites), mx, n_rep)) / 2 ** 30
        how = "enumerated" if args.enumerate_nmixture else "direct marginal sum"
        print(f"  max_abundance={mx} (data need {need}); {how} ~{gib:.2f} GiB")

    out = {"tier": args.tier, "window": [args.rep_start, args.rep_end],
           "covariate_year": out_cov_year, "years": years,
           "n_covariates": int(X.shape[1]),
           "condition_number_raw": cond_raw,
           "pca": bool(args.pca),
           "pca_info": out_pca,
           "n_sites": int(len(sites)), "spatial": bool(args.spatial),
           "max_abundance": mx}

    # occu() first, and PERSIST IT IMMEDIATELY. It is the robust member of the
    # pair and the one the designation comparison needs; losing a finished fit
    # because the optional nmixture blew up afterwards is not acceptable.
    ib = occupancy.build_inputs(sites, counts, years, X, binary=True)
    r_occu = occupancy.fit_occu(ib, coords=coords, num_samples=args.samples,
                                num_warmup=args.warmup, num_chains=args.chains,
                                seed=args.seed)
    args._mcmc = getattr(r_occu, "mcmc", None)
    psi = occupancy.psi_from_occu(r_occu)
    occu_diag = occupancy.diagnostics_of(r_occu)
    grid_psi = None
    if not args.no_grid:
        grid_psi = _biolith_grid_psi(args, occupancy, covariates, data, X,
                                     pca_info if args.pca else None, cov_year,
                                     mu=mu, sd=sd)
    out["occu"] = {"psi_mean": float(psi.mean()),
                   "convergence": occupancy.convergence(r_occu),
                   "sampler": occu_diag,
                   "detection": occupancy.detection_summary(r_occu)}
    from src.analysis.sdm_benchmark.occupancy import worst_r_hat
    _cv = out["occu"]["convergence"]
    print(f"\nbiolith[{args.tier}] occu psi={psi.mean():.3f} "
          f"r_hat_max={worst_r_hat(_cv):.3f} ({_cv.get('worst', {}).get('param', '?')})")
    if _cv.get("error"):
        print(f"  WARNING: convergence not assessed -- {_cv['error']}")
    _warn_saturation("occu", occu_diag)
    _write(args.out, f"biolith_{args.tier}.json", out)
    pay = dict(psi=psi, row=sites.row.to_numpy(), col=sites.col.to_numpy(),
               lon=sites.Longitude.to_numpy(), lat=sites.Latitude.to_numpy())
    if grid_psi is not None:
        pay["grid_psi"] = grid_psi.astype(np.float32)
    np.savez_compressed(os.path.join(args.out, f"biolith_{args.tier}_pred.npz"), **pay)

    if args.no_nmixture:
        print("  nmixture skipped (--no-nmixture)")
        return out

    # nmixture is opt-in: enumeration is QUADRATIC in max_abundance, and House
    # Finch route counts reach 490 in the western native range, which puts the
    # honest max_abundance far past what a GPU can hold. A failure here must not
    # discard the occu fit above.
    pn = None
    out["nmixture_method"] = ("enumerated" if args.enumerate_nmixture
                              else f"marginal/{args.mixture}")
    try:
        ic = occupancy.build_inputs(sites, counts, years, X, binary=False)
        if args.enumerate_nmixture:
            r_nm = occupancy.fit_nmixture(ic, max_abundance=mx, coords=coords,
                                          site_random_effects=args.site_random_effects,
                                          num_samples=args.samples,
                                          num_warmup=args.warmup,
                                          num_chains=args.chains, seed=args.seed,
                                          budget_gib=args.budget_gib)
        else:
            r_nm = occupancy.fit_nmixture_marginal(
                ic, max_abundance=mx, mixture=args.mixture,
                site_random_effects=args.site_random_effects,
                num_samples=args.samples, num_warmup=args.warmup,
                num_chains=args.chains, seed=args.seed,
                budget_gib=args.budget_gib)
        pn = occupancy.psi_from_nmixture(r_nm)
        chk = occupancy.analytic_poisson_check(r_nm)
        diag = getattr(r_nm, "diagnostics", {}) or {}
        ident = occupancy.identifiability(r_nm)
        det = occupancy.detection_summary(r_nm)
        out["nmixture"] = {"p_occ_mean": float(pn.mean()),
                           "jensen_gap_max": chk["max_abs_gap"],
                           "convergence": occupancy.convergence(r_nm),
                           "sampler": diag,
                           "identifiability": ident,
                           "detection": det}
        for k, v in ident.items():
            if abs(v) > 0.9:
                print(f"  WARNING: {k}={v:+.3f} -- lambda and p are nearly "
                      f"unidentified (the intrinsic N-mixture ridge, NOT a "
                      f"covariate problem; rotation cannot fix it)")
        if det.get("degenerate"):
            print(f"  WARNING: detection estimated at {det['mean']:.3f} "
                  f"-- degenerate, the abundance scale is not identified")
        _warn_saturation("nmixture", diag)
        out["occu_vs_nmixture_corr"] = float(np.corrcoef(psi, pn)[0, 1])
        cv = out["nmixture"]["convergence"]
        print(f"  nmixture P(N>0)={pn.mean():.3f} "
              f"corr={out['occu_vs_nmixture_corr']:.3f} "
              f"jensen_gap={chk['max_abs_gap']:.4f} "
              f"r_hat_max={worst_r_hat(cv):.3f} "
              f"({cv.get('worst', {}).get('param', '?')})")
        if cv.get("error"):
            print(f"  WARNING: convergence not assessed -- {cv['error']}")
    except (MemoryError, RuntimeError) as e:
        out["nmixture"] = {"failed": f"{type(e).__name__}: {e}"}
        print(f"  nmixture FAILED (occu result above is kept): {e}")

    _write(args.out, f"biolith_{args.tier}.json", out)
    if pn is not None:
        # EXTEND the payload written after occu -- do not rebuild it. Rebuilding
        # here silently dropped grid_psi: the occu save wrote the continental
        # surface and this one overwrote the file without it, so a run whose log
        # said "grid surface: 17,209 of 17,209 land cells" still produced a file
        # with no grid in it.
        pay["p_occ_nmix"] = pn
        np.savez_compressed(
            os.path.join(args.out, f"biolith_{args.tier}_pred.npz"), **pay)
    return out


def cmd_compare(args):
    """THE BENCHMARK: correlative designation vs the dynamic model's lam >= 1.

    Both sides are binarized and overlaid on the same surveyed cells. The number
    to read is ``occupied_sink`` -- occupied ground the niche call says is not
    self-sustaining. A correlative SDM infers its niche axis FROM occurrence, so
    it can only populate that cell by threshold noise; the dynamic model can,
    because lambda and N are separate quantities.
    """
    import rasterio
    from src.config_utils import load_data_config

    fields = designation.load_dynamic_fields(args.run_dir)
    lam = fields["lam_fundamental_modern"]
    niche_dyn, finite = designation.niche_from_lambda(lam, args.lam_threshold)
    ny, nx = lam.shape

    df = data.plains_band(data.attach_zones(
        data.route_years(args.rep_start, args.rep_end)))
    occ, surv, _mean = designation.cell_occupancy(df, ny, nx)
    with rasterio.open(load_data_config()["regions"]["great_plains_zones"]) as src:
        zones = src.read(1)

    models = {"dynamic (lambda>=1)": niche_dyn}
    continuous = {}                     # un-thresholded surfaces, by model
    for path in sorted(glob.glob(os.path.join(args.out, "*_pred.npz"))):
        tag = os.path.basename(path)[: -len("_pred.npz")]
        with np.load(path, allow_pickle=True) as z:
            keys = set(z.files)
            if "psi" in keys:                       # biolith: one row per site
                score, r, c = z["psi"], z["row"], z["col"]
            elif "occ_pred" in keys:                # BRT: one row per route-year
                score, r, c = z["occ_pred"], z["row"], z["col"]
            else:
                continue
            r, c = np.asarray(r), np.asarray(c)
        ok = np.isfinite(score)
        score, r, c = np.asarray(score)[ok], r[ok], c[ok]
        # Aggregate to cells (BRT rows are route-years), then threshold against
        # OBSERVED occupancy -- which is precisely why the resulting "niche"
        # axis cannot disagree with occupancy except by threshold noise.
        # Mean over the rows landing in a cell. BRT rows are route-years and
        # biolith rows are sites, so several can share a 27 km cell.
        tot = np.zeros((ny, nx), dtype=float)
        cnt = np.zeros((ny, nx), dtype=float)
        np.add.at(tot, (r, c), score.astype(float))
        np.add.at(cnt, (r, c), 1.0)
        has = cnt > 0
        grid = np.where(has, tot / np.maximum(cnt, 1.0), np.nan)
        m = has & surv
        if not m.any():
            print(f"  {tag}: no predictions on surveyed cells, skipped")
            continue
        continuous[tag] = grid
        thr = designation.threshold_max_sss(grid[m], occ[m])
        models[tag] = has & (np.where(has, grid, -np.inf) >= thr)

    out = {"run_dir": args.run_dir, "occupancy_vs_demography": {}, "models": {}}

    # THRESHOLD-FREE FIRST. psi is already the quantity of interest and needs no
    # cut; thresholding it destroys the finding (a model reporting psi=0.83 in
    # the Great Plains reads as "35% niche" under maxSSS). Mean lambda needs no
    # cut either. Both sides compare directly, so this is the headline and the
    # contingency table below is the supporting detail.
    print(f"\nOCCUPANCY vs SELF-SUSTAINING -- no threshold on either side, "
          f"{data.PLAINS_LAT_MIN:.0f}-{data.PLAINS_LAT_MAX:.0f}N")
    psi_names = [n for n in continuous if n.startswith("biolith")]
    head = f"    {'zone':14s} {'cells':>5s}" + "".join(
        f"{n.replace('biolith_', 'psi '):>13s}" for n in psi_names)
    print(head + f"{'BBS any':>9s}{'lam>=1':>9s}{'mean lam':>9s}")
    for zone, lab in ZONE_LABELS.items():
        base = finite & surv & (zones == zone)
        if not base.sum():
            continue
        rec = {"n_cells": int(base.sum()),
               "bbs_any_detection": float(occ[base].mean()),
               "lam_ge_1_fraction": float(niche_dyn[base].mean()),
               "mean_lambda": float(np.nanmean(lam[base]))}
        row = f"    {lab:14s} {int(base.sum()):5d}"
        for n in psi_names:
            v = float(np.nanmean(continuous[n][base]))
            rec[f"psi_{n.replace('biolith_', '')}"] = v
            row += f"{v:13.3f}"
        out["occupancy_vs_demography"][lab] = rec
        print(row + f"{rec['bbs_any_detection']:9.3f}"
                    f"{100*rec['lam_ge_1_fraction']:8.1f}%{rec['mean_lambda']:9.3f}")

    print(f"\nDesignation comparison (maxSSS-thresholded -- supporting detail; "
          f"the cut is a convention, not a finding)")
    for zone, lab in list(ZONE_LABELS.items()) + [(None, "all")]:
        base = finite & surv & ((zones == zone) if zone else (zones > 0))
        if not base.sum():
            continue
        print(f"\n  {lab.upper()}  ({int(base.sum())} cells, "
              f"{100*occ[base].mean():.1f}% occupied)")
        print(f"    {'model':26s} {'niche%':>7s} {'OCC_SINK':>9s} {'UNOCC_SRC':>10s} {'kappa':>7s}")
        for name, niche in models.items():
            r = designation.contingency(niche, occ, base)
            rec = {"niche_fraction": float(niche[base].mean()),
                   "occupied_sink": r["fractions"]["occupied_sink"],
                   "unoccupied_source": r["fractions"]["unoccupied_source"],
                   "kappa": designation.cohens_kappa(niche, occ, base)}
            out["models"].setdefault(name, {})[lab] = rec
            print(f"    {name:26s} {100*rec['niche_fraction']:7.1f} "
                  f"{rec['occupied_sink']:9.3f} {rec['unoccupied_source']:10.3f} "
                  f"{rec['kappa']:+7.3f}")
    _write(args.out, "comparison.json", out)
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, rep=True):
        p.add_argument("--out", default=str(_REPO / "results" / "sdm_benchmark"))
        p.add_argument("--skip-existing", action="store_true",
                       help="skip a stage whose report already exists (resume "
                            "after a wall-clock kill without redoing work)")
        p.add_argument("--start-year", type=int, default=data.DEFAULT_START_YEAR)
        p.add_argument("--end-year", type=int, default=data.DEFAULT_END_YEAR)
        if rep:
            p.add_argument("--rep-start", type=int, default=data.DEFAULT_REPLICATE_START)
            p.add_argument("--rep-end", type=int, default=data.DEFAULT_REPLICATE_END)

    p = sub.add_parser("preflight", help="check inputs exist; fits nothing")
    common(p)
    p.add_argument("--run-dir", default=None, help="also check a MAP run's fields")
    p.add_argument("--tiers", nargs="+", default=["standard", "full", "latent"],
                   choices=["standard", "full", "latent"],
                   help="only check these tiers (default: all three)")
    p.set_defaults(fn=cmd_preflight)

    p = sub.add_parser("compare", help="THE BENCHMARK: SDM vs dynamic designation")
    common(p)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--lam-threshold", type=float, default=1.0)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("premise", help="observed BBS structure, no model")
    common(p); p.set_defaults(fn=cmd_premise)

    p = sub.add_parser("designation", help="dynamic model's niche x occupancy 2x2")
    common(p)
    p.add_argument("--run-dir", required=True, help="a completed MAP run directory")
    p.add_argument("--lam-threshold", type=float, default=1.0)
    p.add_argument("--min-routes", type=int, default=1)
    p.set_defaults(fn=cmd_designation)

    p = sub.add_parser("brt", help="boosted-regression-tree baselines")
    common(p, rep=False)
    p.add_argument("--tier", choices=["standard", "full", "latent"], default="standard")
    p.add_argument("--block-cells", type=int, default=6)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--buffer-cells", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_brt)

    p = sub.add_parser("biolith", help="occu() and nmixture() baselines")
    common(p)
    p.add_argument("--tier", choices=["standard", "full", "latent"], default="standard")
    p.add_argument("--spatial", action="store_true",
                   help="add a spatial GP (Test B: environment vs space)")
    p.add_argument("--mixture", choices=["NB", "P"], default="NB",
                   help="latent abundance distribution (unmarked::pcount's "
                        "mixture). NB gives overdispersion from ONE parameter "
                        "and matches the dynamic model's NB2")
    p.add_argument("--site-random-effects", action="store_true",
                   help="add per-site random effects. OFF by default: 3853 free "
                        "per-site parameters wreck the sampler geometry AND "
                        "absorb variation the benchmark wants attributed to "
                        "environment")
    p.add_argument("--no-nmixture", action="store_true",
                   help="skip the abundance fit (occu only)")
    p.add_argument("--enumerate-nmixture", action="store_true",
                   help="use biolith's ENUMERATED nmixture instead of the "
                        "direct marginal sum. Quadratic in max_abundance (55 GiB "
                        "at this data's ceiling); for cross-checking only")
    p.add_argument("--max-abundance", type=int, default=None,
                   help="override the data-derived ceiling (TRUNCATES abundance "
                        "above it, biasing low -- state it if you use it)")
    p.add_argument("--budget-gib", type=float, default=8.0,
                   help="refuse an nmixture enumeration larger than this")
    p.add_argument("--pca", type=int, nargs="?", const=-1, default=None,
                   help="rotate covariates to principal components before "
                        "fitting. Bare --pca keeps enough components for "
                        "--pca-var; an integer fixes the count. Recommended for "
                        "the full tier, whose 302 channels are near duplicates")
    p.add_argument("--pca-per-stream", action="store_true",
                   help="rotate within each covariate stream instead of "
                        "globally. Only useful WITH --pca-var: it protects the "
                        "small human-footprint streams from being truncated "
                        "away, but leaves the streams collinear with each other "
                        "(measured condition number 31.6 vs 1 for global)")
    p.add_argument("--pca-var", type=float, default=None,
                   help="OPTIONAL variance budget. Off by default: rotation "
                        "alone fixes the conditioning (the components are "
                        "standardized before fitting, so the design is "
                        "orthonormal however many are kept), while truncating "
                        "on variance can discard real signal -- variance is not "
                        "predictive relevance")
    p.add_argument("--pca-select-cv", action="store_true",
                   help="choose the component count by held-out AUC under "
                        "spatial-block CV (one-SE rule) instead of keeping all")
    p.add_argument("--no-grid", action="store_true",
                   help="skip the continental prediction surface. It is ON by "
                        "default: an SDM's product is a map, and restricting it "
                        "to surveyed cells shows the survey design instead")
    p.add_argument("--dump-design", action="store_true",
                   help="save the assembled site x covariate matrix (<1 MB) so "
                        "the sampler can be diagnosed off-HPC without moving "
                        "any raster")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_biolith)
    return ap


def main():
    args = build_parser().parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

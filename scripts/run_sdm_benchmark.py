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


def cmd_brt(args):
    """Boosted-regression-tree baselines under spatial block CV."""
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
    np.savez_compressed(os.path.join(args.out, f"brt_{args.tier}_pred.npz"),
                        occ_pred=occ_pred, cnt_pred=cnt_pred, y=y,
                        row=rows, col=cols, year=df["Year"].to_numpy(),
                        zone=df["zone"].to_numpy())
    return out


def cmd_biolith(args):
    """Detection-corrected occu() and nmixture(), reduced to occupancy."""
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
    X, mu, sd = covariates.standardize(X)
    keep = ~np.isnan(X).any(axis=1)
    if not keep.all():
        print(f"  dropping {int((~keep).sum())} sites with NaN covariates")
        sites, counts, X = sites[keep].reset_index(drop=True), counts[keep], X[keep]

    coords = None
    if args.spatial:
        coords = np.column_stack([sites.Longitude, sites.Latitude]).astype(float)
        coords = (coords - coords.mean(0)) / coords.std(0)

    mx, need = occupancy.suggest_max_abundance(counts)
    print(f"  max_abundance={mx} (need {need}; biolith default 100 would truncate)")

    out = {"tier": args.tier, "window": [args.rep_start, args.rep_end],
           "covariate_year": out_cov_year, "years": years,
           "n_sites": int(len(sites)), "spatial": bool(args.spatial),
           "max_abundance": mx}

    ib = occupancy.build_inputs(sites, counts, years, X, binary=True)
    r_occu = occupancy.fit_occu(ib, coords=coords, num_samples=args.samples,
                                num_warmup=args.warmup, num_chains=args.chains,
                                seed=args.seed)
    psi = occupancy.psi_from_occu(r_occu)
    out["occu"] = {"psi_mean": float(psi.mean()),
                   "convergence": occupancy.convergence(r_occu)}

    ic = occupancy.build_inputs(sites, counts, years, X, binary=False)
    r_nm = occupancy.fit_nmixture(ic, max_abundance=mx, coords=coords,
                                  site_random_effects=not args.no_site_re,
                                  num_samples=args.samples, num_warmup=args.warmup,
                                  num_chains=args.chains, seed=args.seed)
    pn = occupancy.psi_from_nmixture(r_nm)
    chk = occupancy.analytic_poisson_check(r_nm)
    out["nmixture"] = {"p_occ_mean": float(pn.mean()),
                       "jensen_gap_max": chk["max_abs_gap"],
                       "convergence": occupancy.convergence(r_nm)}
    out["occu_vs_nmixture_corr"] = float(np.corrcoef(psi, pn)[0, 1])
    print(f"\nbiolith[{args.tier}] psi={psi.mean():.3f} P(N>0)={pn.mean():.3f} "
          f"corr={out['occu_vs_nmixture_corr']:.3f} "
          f"jensen_gap={chk['max_abs_gap']:.4f}")
    _write(args.out, f"biolith_{args.tier}.json", out)
    np.savez_compressed(os.path.join(args.out, f"biolith_{args.tier}_pred.npz"),
                        psi=psi, p_occ_nmix=pn, row=sites.row.to_numpy(),
                        col=sites.col.to_numpy(),
                        lon=sites.Longitude.to_numpy(), lat=sites.Latitude.to_numpy())
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, rep=True):
        p.add_argument("--out", default=str(_REPO / "results" / "sdm_benchmark"))
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
    p.add_argument("--no-site-re", action="store_true",
                   help="disable site random effects (Poisson, not Poisson-lognormal)")
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

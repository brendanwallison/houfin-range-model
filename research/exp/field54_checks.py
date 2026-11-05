"""FIELD54 Stage A checks: native 54 km covariates against the 27 km products.

Writes one JSON (``--out``) with, per family:
  * same-rule aggregation identity: 54 km value vs the 2x2 block aggregate of the 27 km
    children (mean for area averages, sum for HYDE counts), on 54 km land cells;
  * quantile channels: 54 km value inside [min, max] of the children's same quantile;
  * fine-lattice identity for elevation / BUI (the 54 km sub-cell lattice IS the union
    of the 27 km children's, so the fine grids should be identical);
  * LUH-3 parent-weighted vs plain-average rule at 54 km;
  * states: schema identity, NaN on land cells, per-channel correlation with the 2x2
    block mean of the 27 km states.

    python research/exp/field54_checks.py --d27 $SCRATCH/houfin/data --d54 $HOUFIN_DATA54 \
        --s27 $WORK/houfin/processed/encoder/states --s54 $HOUFIN_PROCESSED54/states \
        --out $HOUFIN_PROCESSED54/field54_checks.json
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_YEAR_TIF = re.compile(r"^(?P<var>.+)_(?P<year>\d{4})_grid\.tif$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    with rasterio.open(path) as src:
        return src.read(1, masked=True).astype("float64").filled(np.nan)


def pad_to(a, H2, W2):
    """Pad a 27 km array (NaN) so it tiles into 2x2 blocks of the 54 km grid."""
    H, W = a.shape[-2:]
    out = np.full(a.shape[:-2] + (2 * H2, 2 * W2), np.nan)
    out[..., :H, :W] = a
    return out


def blocks(a, H2, W2):
    """(..., H27, W27) -> (..., H2, W2, 4) children."""
    p = pad_to(a, H2, W2)
    lead = p.shape[:-2]
    return p.reshape(lead + (H2, 2, W2, 2)).swapaxes(-3, -2).reshape(lead + (H2, W2, 4))


def child_mean(a, H2, W2):
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(blocks(a, H2, W2), axis=-1)


def child_sum(a, H2, W2):
    b = blocks(a, H2, W2)
    s = np.nansum(b, axis=-1)
    s[~np.isfinite(b).any(axis=-1)] = np.nan
    return s


def child_minmax(a, H2, W2):
    import warnings
    b = blocks(a, H2, W2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(b, axis=-1), np.nanmax(b, axis=-1)


def all_children_finite(a, H2, W2):
    return np.isfinite(blocks(a, H2, W2)).all(axis=-1)


def stats_diff(x54, ref, sel):
    """Difference metrics of x54 vs ref over boolean sel (both finite)."""
    m = sel & np.isfinite(x54) & np.isfinite(ref)
    if not m.any():
        return {"n": 0}
    x, r = x54[m], ref[m]
    d = x - r
    sd = float(np.std(r))
    out = {"n": int(m.sum()), "max_abs": float(np.max(np.abs(d))),
           "p99_abs": float(np.quantile(np.abs(d), 0.99)),
           "rms": float(np.sqrt(np.mean(d * d))), "sd_ref": sd,
           "max_abs_over_sd": float(np.max(np.abs(d)) / sd) if sd > 0 else None,
           "rms_over_sd": float(np.sqrt(np.mean(d * d)) / sd) if sd > 0 else None}
    if np.std(x) > 0 and sd > 0:
        out["corr"] = float(np.corrcoef(x, r)[0, 1])
    return out


def within_range(x54, lo, hi, sel, rel_tol=1e-5):
    m = sel & np.isfinite(x54) & np.isfinite(lo) & np.isfinite(hi)
    x, a, b = x54[m], lo[m], hi[m]
    tol = rel_tol * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    below = np.clip(a - tol - x, 0, None)
    above = np.clip(x - b - tol, 0, None)
    exc = np.maximum(below, above)
    spread = np.maximum(b - a, 0)
    bad = exc > 0
    return {"n": int(m.sum()), "n_outside": int(bad.sum()),
            "frac_outside": float(bad.mean()) if m.any() else None,
            "max_excess": float(exc.max()) if m.any() else None,
            "max_excess_over_child_spread": float(np.max(exc[bad] / np.maximum(spread[bad], 1e-12)))
            if bad.any() else 0.0}


def var_years(grid_dir):
    out = {}
    for f in os.listdir(grid_dir):
        m = _YEAR_TIF.match(f)
        if m:
            out.setdefault(m.group("var"), []).append(int(m.group("year")))
    return {k: sorted(v) for k, v in out.items()}


def stack(grid_dir, var, years, pool):
    return np.stack(list(pool.map(lambda y: read(os.path.join(grid_dir, f"{var}_{y}_grid.tif")),
                                  years)))


def summarize(per_var, keys=("max_abs_over_sd", "rms_over_sd", "corr", "frac_outside",
                             "max_excess")):
    agg = {}
    for k in keys:
        vals = [v[k] for v in per_var.values() if isinstance(v, dict) and v.get(k) is not None]
        if vals:
            agg[k] = {"min": float(np.min(vals)), "median": float(np.median(vals)),
                      "max": float(np.max(vals))}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d27", required=True)
    ap.add_argument("--d54", required=True)
    ap.add_argument("--s27", required=True)
    ap.add_argument("--s54", required=True)
    ap.add_argument("--grid54-mask", default=None,
                    help="processed/grid54/land_mask/ocean_mask_54km.tif (age-model inputs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", default="mask,luh,hyde,soil,elev,bui,climate,states")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--clim-dir", default="climate", help="54 km regrouped climate subdir of --d54")
    ap.add_argument("--clim-grid", default="climate_grid_monthly", help="54 km climate grid subdir of --d54")
    args = ap.parse_args()
    sections = set(args.sections.split(","))
    from src.data.masks import read_land_mask

    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    pool = ThreadPoolExecutor(args.threads)
    t0 = time.time()

    def save():
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)

    with rasterio.open(os.path.join(args.d54, "ref_grid_54km.tif")) as r54, \
            rasterio.open(os.path.join(args.d27, "ref_grid_27km.tif")) as r27:
        H2, W2 = r54.height, r54.width
        H, W = r27.height, r27.width
        t54, t27 = r54.transform, r27.transform
    assert (t54.c, t54.f) == (t27.c, t27.f) and abs(t54.a - 2 * t27.a) < 1e-6, (t54, t27)
    assert H2 == (H + 1) // 2 and W2 == W // 2
    # read_land_mask validates 0 = land / 1 = ocean and returns True = LAND
    land54 = np.asarray(read_land_mask(os.path.join(args.d54, "land_mask", "ocean_mask_54km.tif")),
                        dtype=bool)
    land27 = np.asarray(read_land_mask(os.path.join(args.d27, "land_mask", "ocean_mask_27km.tif")),
                        dtype=bool)
    # 27 km land child count per 54 km cell (pad row = ocean)
    nland_child = blocks(land27.astype(float), H2, W2)
    nland_child = np.nan_to_num(nland_child).sum(-1)

    if "mask" in sections:
        m = {"sha256_data54": sha256(os.path.join(args.d54, "land_mask", "ocean_mask_54km.tif")),
             "n_land54": int(land54.sum()), "n_land27": int(land27.sum()),
             "land54_with_0_land_children": int((land54 & (nland_child == 0)).sum()),
             "land54_min_land_children": int(nland_child[land54].min())}
        if args.grid54_mask:
            m["sha256_grid54_age_inputs"] = sha256(args.grid54_mask)
            m["identical"] = m["sha256_data54"] == m["sha256_grid54_age_inputs"]
        res["mask"] = m
        save()
        print("[mask]", m, flush=True)

    # ---------------- LUH-3 ----------------
    if "luh" in sections:
        d27, d54 = os.path.join(args.d27, "luh3_grid"), os.path.join(args.d54, "luh3_grid")
        diag = os.path.join(args.d54, "luh3_grid_plainavg_diag")
        pw = json.load(open(os.path.join(d54, "parent_weighting.json")))["parent_weighted"] \
            if os.path.exists(os.path.join(d54, "parent_weighting.json")) else {}
        vy27, vy54 = var_years(d27), var_years(d54)
        out = {"vars27": sorted(vy27), "vars54": sorted(vy54),
               "same_varset": sorted(vy27) == sorted(vy54), "same_rule": {}, "weighted": {}}
        for var in sorted(vy54):
            yrs = sorted(set(vy27.get(var, [])) & set(vy54[var]))
            a27 = stack(d27, var, yrs, pool)
            cm = child_mean(a27, H2, W2)
            full = all_children_finite(a27, H2, W2)
            sel = np.broadcast_to(land54, cm.shape)
            if var in pw:
                w54 = stack(d54, var, yrs, pool)
                p54 = stack(diag, var, yrs, pool)
                out["same_rule"][var] = stats_diff(p54, cm, sel)
                out["same_rule"][var]["all_children_finite"] = stats_diff(p54, cm, sel & full)
                dd = stats_diff(w54, p54, sel)
                fin = sel & np.isfinite(w54) & np.isfinite(p54)
                rel = np.abs(w54 - p54)[fin] / max(np.std(p54[fin]), 1e-12)
                dd.update({"frac_cells_diff_gt_1pct_sd": float((rel > 0.01).mean()),
                           "frac_cells_diff_gt_10pct_sd": float((rel > 0.10).mean()),
                           "median_abs": float(np.median(np.abs(w54 - p54)[fin])),
                           "p95_abs": float(np.quantile(np.abs(w54 - p54)[fin], 0.95)),
                           "mean_w": float(w54[fin].mean()), "mean_plain": float(p54[fin].mean()),
                           "parents": pw[var]})
                # last-year snapshot (2024) for the record
                out["weighted"][var] = dd
                out["weighted"][var]["vs_27km_childmean"] = stats_diff(w54, cm, sel)
            else:
                x54 = stack(d54, var, yrs, pool)
                out["same_rule"][var] = stats_diff(x54, cm, sel)
                out["same_rule"][var]["all_children_finite"] = stats_diff(x54, cm, sel & full)
            print(f"[luh] {var} {time.time() - t0:.0f}s", flush=True)
        out["same_rule_summary"] = summarize(out["same_rule"])
        out["weighted_summary"] = summarize(out["weighted"], keys=(
            "max_abs_over_sd", "rms_over_sd", "corr", "frac_cells_diff_gt_1pct_sd",
            "frac_cells_diff_gt_10pct_sd"))
        res["luh3"] = out
        save()

    # ---------------- HYDE ----------------
    if "hyde" in sections:
        d27, d54 = os.path.join(args.d27, "hyde35_grid"), os.path.join(args.d54, "hyde35_grid")
        vy27, vy54 = var_years(d27), var_years(d54)
        out = {"same_varset": sorted(vy27) == sorted(vy54), "vars": {}}
        for var in sorted(vy54):
            yrs = sorted(set(vy27.get(var, [])) & set(vy54[var]))
            a27 = stack(d27, var, yrs, pool)
            x54 = stack(d54, var, yrs, pool)
            ref = child_mean(a27, H2, W2) if "density" in var else child_sum(a27, H2, W2)
            sel = np.broadcast_to(land54, ref.shape)
            full = all_children_finite(a27, H2, W2)
            s = stats_diff(x54, ref, sel)
            s["rule"] = "mean" if "density" in var else "sum"
            s["all_children_finite"] = stats_diff(x54, ref, sel & full)
            fin = sel & np.isfinite(x54) & np.isfinite(ref)
            if "density" not in var:
                s["total54"] = float(np.nansum(x54[fin]))
                s["total27_children"] = float(np.nansum(ref[fin]))
            s["years"] = [yrs[0], yrs[-1], len(yrs)]
            out["vars"][var] = s
        res["hyde"] = out
        save()
        print("[hyde] done", flush=True)

    # ---------------- Soil ----------------
    if "soil" in sections:
        d27, d54 = os.path.join(args.d27, "soilgrids_grid"), os.path.join(args.d54, "soilgrids_grid")
        f27 = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d27, "*_grid.tif")))
        f54 = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d54, "*_grid.tif")))
        out = {"same_files": f27 == f54, "vars": {}}
        for f in f54:
            a27 = read(os.path.join(d27, f))
            a27l = np.where(land27, a27, np.nan)  # 27 km values are hole-filled on 27 km land
            x54 = read(os.path.join(d54, f))
            cm = child_mean(a27l, H2, W2)
            s = stats_diff(x54, cm, land54)
            # interior: every child is 27 km land (no coastal / fill effect)
            s["all_children_land"] = stats_diff(x54, cm, land54 & (nland_child == 4))
            s["nan_on_land54"] = int((land54 & ~np.isfinite(x54)).sum())
            out["vars"][f] = s
        out["summary"] = summarize(out["vars"])
        res["soil"] = out
        save()
        print("[soil] done", flush=True)

    # ---------------- Elevation ----------------
    if "elev" in sections:
        d27, d54 = os.path.join(args.d27, "elevation"), os.path.join(args.d54, "elevation")
        out = {"vars": {}}
        for q in ("q10", "q50", "q90"):
            a27 = read(os.path.join(d27, f"elev_{q}.tif"))
            x54 = read(os.path.join(d54, f"elev_{q}.tif"))
            lo, hi = child_minmax(a27, H2, W2)
            out["vars"][q] = {"within_children_range_land54": within_range(x54, lo, hi, land54),
                              "within_children_range_all": within_range(
                                  x54, lo, hi, np.ones_like(land54)),
                              "vs_child_mean": stats_diff(x54, child_mean(a27, H2, W2), land54),
                              "nan_on_land54": int((land54 & ~np.isfinite(x54)).sum())}
        # fine-lattice identity: DEM on the 27 km /15 lattice vs the 54 km /30 lattice
        try:
            from src.data.preprocess.elevation import dem_to_fine_grid
            dem = sorted(glob.glob(os.path.join(args.d27, "dem", "*.tif")))[0]
            with rasterio.open(os.path.join(args.d27, "ref_grid_27km.tif")) as r:
                f27, _ = dem_to_fine_grid(dem, r.transform, r.crs, r.height, r.width, 15)
            with rasterio.open(os.path.join(args.d54, "ref_grid_54km.tif")) as r:
                f54, _ = dem_to_fine_grid(dem, r.transform, r.crs, r.height, r.width, 30)
            sub = f54[:f27.shape[0], :f27.shape[1]]
            both = np.isfinite(sub) & np.isfinite(f27)
            out["fine_lattice"] = {"shape27": list(f27.shape), "shape54": list(f54.shape),
                                   "n_both": int(both.sum()),
                                   "n_nan_mismatch": int((np.isfinite(sub) != np.isfinite(f27)).sum()),
                                   "max_abs_diff": float(np.max(np.abs(sub[both] - f27[both]))),
                                   "frac_identical": float((sub[both] == f27[both]).mean())}
            # union-quantile identity, exact: recompute 54 km quantiles from the 27 km lattice
            from src.processing import regrid
            pad = np.full((2 * H2 * 15, 2 * W2 * 15), np.nan)
            pad[:f27.shape[0], :f27.shape[1]] = f27
            q54 = regrid.block_quantiles(pad, 30, (0.1, 0.5, 0.9))
            for k, q in enumerate(("q10", "q50", "q90")):
                x54 = read(os.path.join(d54, f"elev_{q}.tif"))
                out["vars"][q]["vs_union_of_27km_lattice"] = stats_diff(
                    x54, q54[k].astype("float32").astype("float64"), np.ones_like(land54))
            del f27, f54, sub, pad
        except Exception as exc:  # report, never fail the whole check
            out["fine_lattice_error"] = repr(exc)
        res["elevation"] = out
        save()
        print("[elev] done", flush=True)

    # ---------------- BUI ----------------
    if "bui" in sections:
        d27, d54 = os.path.join(args.d27, "bui_grid"), os.path.join(args.d54, "bui_grid")
        man27 = json.load(open(os.path.join(d27, "manifest.json")))
        man54 = json.load(open(os.path.join(d54, "manifest.json")))
        out = {"same_variables": man27["variables"] == man54["variables"],
               "fine_factor": [man27["fine_factor"], man54["fine_factor"]],
               "raw_fill_value27": man27["raw_fill_value"], "raw_fill_value54": man54["raw_fill_value"],
               "vars": {}}
        vy27, vy54 = var_years(d27), var_years(d54)
        av27 = stack(d27, "bui_avail", vy54["bui_avail"], pool)
        av54 = stack(d54, "bui_avail", vy54["bui_avail"], pool)
        av27p = np.where(np.isfinite(pad_to(av27, H2, W2)), pad_to(av27, H2, W2), 0.0)
        cm_av = blocks(av27p, H2, W2).mean(-1)
        out["vars"]["bui_avail"] = {"vs_child_mean_all_cells": stats_diff(
            av54, cm_av, np.ones(av54.shape, bool))}
        cov27 = av27 > 0
        for var in sorted(v for v in vy54 if v != "bui_avail"):
            yrs = vy54[var]
            a27 = stack(d27, var, yrs, pool)
            x54 = stack(d54, var, yrs, pool)
            a27c = np.where(cov27, a27, np.nan)  # only children with real coverage
            lo, hi = child_minmax(a27c, H2, W2)
            cov54 = av54 > 0
            out["vars"][var] = {"within_children_range_covered": within_range(x54, lo, hi, cov54),
                                "n_years": len(yrs)}
        res["bui"] = out
        save()
        print("[bui] done", flush=True)

    # ---------------- Climate ----------------
    if "climate" in sections:
        d27 = os.path.join(args.d27, "climate_grid_monthly")
        d54 = os.path.join(args.d54, args.clim_grid)
        m27 = json.load(open(os.path.join(d27, "manifest.json")))
        m54 = json.load(open(os.path.join(d54, "manifest.json")))
        out = {"same_variables": m27["variables"] == m54["variables"],
               "n_variables": [m27["n_variables"], m54["n_variables"]],
               "years": [m27["years"], m54["years"]], "vars": {}}
        vy27, vy54 = var_years(d27), var_years(d54)
        out["years_compared"] = {}
        for var in m54["variables"]:
            # compare over the bio-years BOTH grids hold (the manifests say 1902-2025; the grids stop at 2024)
            yrs = sorted(set(vy27.get(var, [])) & set(vy54.get(var, [])))
            yrs = yrs[::int(os.environ.get("F54_CLIM_YEAR_STRIDE", "1"))]     # verification: a year sample suffices
            if not yrs:
                raise SystemExit(f"climate {var}: no year present in both grids")
            out["years_compared"][var] = [yrs[0], yrs[-1], len(yrs)]
            print(f"[climate] {var} ({len(yrs)} years)", flush=True)
            a27 = stack(d27, var, yrs, pool)
            x54 = stack(d54, var, yrs, pool)
            lo, hi = child_minmax(a27, H2, W2)
            sel = np.broadcast_to(land54, x54.shape)
            out["vars"][var] = {"within": within_range(x54, lo, hi, sel),
                                "vs_child_mean": stats_diff(x54, child_mean(a27, H2, W2), sel),
                                "nan_on_land54": int((sel & ~np.isfinite(x54)).sum())}
        out["summary"] = {
            "within": summarize({k: v["within"] for k, v in out["vars"].items()},
                                keys=("frac_outside", "max_excess", "max_excess_over_child_spread")),
            "vs_child_mean": summarize({k: v["vs_child_mean"] for k, v in out["vars"].items()}),
            "nan_on_land54_total": int(sum(v["nan_on_land54"] for v in out["vars"].values()))}
        rs = os.path.join(args.d54, args.clim_dir, "regroup_summary.json")
        if os.path.exists(rs):
            out["regroup_summary"] = {k: v for k, v in json.load(open(rs)).items() if k != "chunks"}
            # coverage: 54 km land cells without any sub-point
            import pandas as pd
            mp = pd.read_parquet(os.path.join(args.d54, args.clim_dir, "subpoint_parent_map.parquet"),
                                 columns=["parent_id"])
            cnt = np.zeros(H2 * W2, int)
            np.add.at(cnt, mp["parent_id"].to_numpy(), 1)
            cnt = cnt.reshape(H2, W2)
            out["coverage"] = {
                "land54_without_subpoints": int((land54 & (cnt == 0)).sum()),
                "subpoints_per_land54_cell": {
                    "min": int(cnt[land54].min()), "p05": float(np.quantile(cnt[land54], .05)),
                    "median": float(np.median(cnt[land54])), "max": int(cnt[land54].max())},
                "nonland54_parents_with_subpoints": int((~land54 & (cnt > 0)).sum()),
                "subpoints_in_nonland54_parents": int(cnt[~land54].sum())}
        res["climate"] = out
        save()
        print("[climate] done", flush=True)

    # ---------------- States ----------------
    if "states" in sections:
        sc27 = json.load(open(os.path.join(args.s27, "state_schema.json")))
        sc54 = json.load(open(os.path.join(args.s54, "state_schema.json")))

        def strip(sc):
            return [{k: s.get(k) for k in ("name", "start", "end", "dim", "type", "variables")}
                    for s in sc["streams"]]
        out = {"schema_identical_streams": strip(sc27) == strip(sc54),
               "total_dim": [sc27["total_dim"], sc54["total_dim"]],
               "years27": [sc27["start_year"], sc27["end_year"], sc27.get("sample_start")],
               "years54": [sc54["start_year"], sc54["end_year"], sc54.get("sample_start")],
               "schema_other_keys_differ": sorted(
                   k for k in set(sc27) | set(sc54) if k != "streams" and sc27.get(k) != sc54.get(k))}
        f54 = sorted(glob.glob(os.path.join(args.s54, "yearly_states", "state_*.npz")))
        out["n_year_files"] = len(f54)
        names = [s["name"] for s in sc54["streams"]]
        nan_land = {n: 0 for n in names}
        # per-channel correlation over all years x land cells (streaming sums)
        acc = {}
        for f in f54:
            y = os.path.basename(f)
            z54 = np.load(f)
            p27 = os.path.join(args.s27, "yearly_states", y)
            z27 = np.load(p27) if os.path.exists(p27) else None
            for n in names:
                a = z54[n].astype("float64")                       # (H2, W2, C)
                nan_land[n] += int(np.isnan(a[land54]).any(axis=-1).sum())
                if z27 is None:
                    continue
                b = np.moveaxis(z27[n].astype("float64"), -1, 0)   # (C, H, W)
                b = np.where(land27[None], b, np.nan)
                cm = np.moveaxis(child_mean(b, H2, W2), 0, -1)     # (H2, W2, C)
                x, r = a[land54], cm[land54]
                ok = np.isfinite(x) & np.isfinite(r)
                x, r = np.where(ok, x, 0), np.where(ok, r, 0)
                s = acc.setdefault(n, np.zeros((6, a.shape[-1])))
                s += np.stack([ok.sum(0), x.sum(0), r.sum(0), (x * x).sum(0), (r * r).sum(0),
                               (x * r).sum(0)])
        out["nan_cells_on_land54_any_channel"] = nan_land
        corr = {}
        for n, s in acc.items():
            nn, sx, sr, sxx, srr, sxr = s
            cov = sxr / nn - sx / nn * sr / nn
            vx = sxx / nn - (sx / nn) ** 2
            vr = srr / nn - (sr / nn) ** 2
            with np.errstate(invalid="ignore", divide="ignore"):
                c = cov / np.sqrt(vx * vr)
            vars_ = next(s_["variables"] for s_ in sc54["streams"] if s_["name"] == n)
            corr[n] = {"min": float(np.nanmin(c)), "median": float(np.nanmedian(c)),
                       "n_below_0.99": int((c < 0.99).sum()), "n_nan": int(np.isnan(c).sum()),
                       "lowest": sorted(((float(ci), v) for ci, v in zip(c, vars_)
                                         if np.isfinite(ci)))[:5]}
        out["corr_54_vs_27_blockmean"] = corr
        res["states"] = out
        save()
        print("[states] done", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

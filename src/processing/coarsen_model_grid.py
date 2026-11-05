"""Coarsen the 27 km age-model inputs to an integer-factor coarser grid (``--factor 2`` -> 54 km).

WHY. A 54 km fit has 1/4 the cells, so it is far cheaper in VRAM and time; it is a resolution
DIAGNOSTIC for the age model, built from exactly the inputs of a named 27 km base run so the two
fits differ in grid alone. The grid-dependent pieces split into two kinds, and each is handled
the way its semantics demand:

AVERAGED here (linear quantities whose coarse value is the area mean of the fine ones):
  * the Z cube -- mean over the LAND children of each block. Features are uncentered Ruzicka
    embeddings with z_i.z_j ~ similarity, so z_bar_B.z_bar_B' is EXACTLY the mean pairwise child
    similarity between blocks B and B' (the kernel mean embedding). That is not what DESK trained
    natively at 54 km would produce; cube_meta.json records the note and the source identity.
  * the native-range seed shape (initpop_shape) -- mean over land children.
REBUILT natively (non-linear or geometric; never averaged):
  * the land mask -- land fraction from the same Natural Earth polygons on a fine grid whose
    subpixels coincide with the 27 km build's (27 km x fine_factor 25 = 54 km x 50 = 1.08 km),
    thresholded at tau, AND "has >= 1 27 km land child" so every coarse land cell owns z values.
  * Z_disp / path features, the disease arrival map and the forward-model kernels are rebuilt
    downstream by their own scripts on the coarse mask (generate_all_path_features.py,
    scripts/build_disease_arrival_map.py, model_inputs.py) -- never averaged.
REMAPPED: BBS observations (each route-year to its parent cell; counts unchanged; a parent that is
  not coarse land is snapped to the nearest coarse land cell, or dropped with --coastal drop), pre-invasion pseudo-zeros (a coarse cell is a pseudo-zero only
  if ALL its land children are), and the Great Plains zone raster (majority of children).

The grid keeps the 27 km origin (snapped to the BBS lattice); an odd fine dimension is padded at the
SOUTH (bottom) / EAST with virtual all-ocean rows/columns so blocks nest exactly.

    python src/processing/coarsen_model_grid.py --factor 2 --source-input-dir <27 km numpyro input>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import rasterio
from rasterio.transform import Affine
import scipy.sparse as sp

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.masks import read_land_mask  # noqa: E402

DEFAULT_SOURCE_INPUT = ("/work/07980/bwa386/ls6/houfin/processed/model_inputs/"
                        "numpyro_input_fd_clean_v2_m2_shared_s64")
DATA = os.environ.get("HOUFIN_DATA", "/home/breallis/houfin/scratch/houfin/data")
PROCESSED = os.environ.get("HOUFIN_PROCESSED", "/work/07980/bwa386/ls6/houfin/processed")

CUBE_NOTE = ("Block-averaged z is the kernel MEAN EMBEDDING of the 27 km cells in each block: "
             "z_bar_B . z_bar_B' = mean over child pairs (i in B, j in B') of z_i . z_j, i.e. the mean "
             "pairwise Ruzicka similarity. It is NOT equivalent to DESK trained natively at the coarse "
             "resolution (that would embed the coarse cell's own pooled community), and |z_bar|^2 < "
             "mean |z_i|^2 wherever the children differ.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def coarse_shape(ny, nx, f):
    return -(-ny // f), -(-nx // f)


def pad_to_blocks(a, f, fill):
    """Pad (ny, nx, ...) at the south/east so both dims are multiples of f."""
    ny, nx = a.shape[:2]
    cy, cx = coarse_shape(ny, nx, f)
    pad = [(0, cy * f - ny), (0, cx * f - nx)] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, pad, constant_values=fill)


def children_blocks(a, f):
    """(ny, nx, ...) padded -> (cy, cx, f*f, ...) children of each block."""
    cy, cx = a.shape[0] // f, a.shape[1] // f
    b = a.reshape(cy, f, cx, f, *a.shape[2:]).swapaxes(1, 2)
    return b.reshape(cy, cx, f * f, *a.shape[2:])


def land_fraction_native(src_dir, transform_c, shape_c, shape_f, f, fine_factor_f, exclude_iso_a3):
    """Coarse-grid land fraction from Natural Earth, on subpixels identical to the fine build's.

    The fine (27 km) mask used fine_factor_f subpixels per cell; the coarse grid uses f*fine_factor_f,
    so every subpixel coincides and the coarse fraction is exactly the mean of the four fine fractions
    (the fine build being reproducible -- checked in main). Subpixels in the virtual padding (outside
    the fine grid's extent) are forced to water so padding never imports land from beyond the box.
    """
    import geopandas as gpd
    import rasterio.features
    from src.data.preprocess.land_mask import (compute_land_fraction, rasterize_country_exclusions,
                                               subtract_lakes)
    ffc = f * fine_factor_f
    H, W = shape_c
    fine_t = transform_c * Affine.scale(1.0 / ffc, 1.0 / ffc)
    shape = (H * ffc, W * ffc)
    crs = "ESRI:102003"

    def rast(path):
        g = gpd.read_file(path).to_crs(crs)
        return rasterio.features.rasterize(((geom, 1) for geom in g.geometry), out_shape=shape,
                                           transform=fine_t, fill=0, dtype="uint8")
    fine = rast(os.path.join(src_dir, "ne_10m_land.shp"))
    fine = subtract_lakes(fine, rast(os.path.join(src_dir, "ne_10m_lakes.shp")))
    excl = rasterize_country_exclusions(os.path.join(src_dir, "ne_10m_admin_0_countries.shp"),
                                        exclude_iso_a3, crs, shape, fine_t)
    fine[excl > 0] = 0
    fine[shape_f[0] * fine_factor_f:, :] = 0      # virtual south padding
    fine[:, shape_f[1] * fine_factor_f:] = 0      # virtual east padding
    return compute_land_fraction(fine, ffc)


def write_mask(path, ocean, transform, crs):
    profile = dict(driver="GTiff", height=ocean.shape[0], width=ocean.shape[1], count=1,
                   dtype="uint8", crs=crs, transform=transform, nodata=255)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(ocean.astype("uint8"), 1)


def averaging_operator(land_f, land_c, f):
    """Sparse (N_land_c x N_land_f) mean over LAND children; fine land cells whose parent is not
    coarse land get no row (their values are dropped). Row order = np.where(land_c)."""
    fr, fc = np.where(land_f)
    cr, cc = np.where(land_c)
    cidx = np.full(land_c.shape, -1, dtype=np.int64)
    cidx[cr, cc] = np.arange(cr.size)
    parent = cidx[fr // f, fc // f]
    keep = parent >= 0
    n_child = np.bincount(parent[keep], minlength=cr.size)
    if (n_child == 0).any():
        raise ValueError(f"{int((n_child == 0).sum())} coarse land cells have no fine land child")
    A = sp.csr_matrix((1.0 / n_child[parent[keep]], (parent[keep], np.nonzero(keep)[0])),
                      shape=(cr.size, fr.size))
    return A, parent, n_child


def parent_cells(rows, cols, land_c, f, coastal="snap"):
    """Fine (row, col) -> coarse parent (row // f, col // f), with the coastal rule (pure).

    A parent that is not coarse land is SNAPPED to the nearest coarse land cell by cell-centre
    distance (``coastal="snap"``) or flagged for dropping (``"drop"``). Returns
    ``(rr, rc, off, keep, dist_cells)``: the coarse cells, the "parent was not coarse land" flag,
    the rows to keep, and the snap distance in coarse cells (0 where not snapped). The one rule
    shared by ``coarsen_bbs`` and the native 54 km observation build (research/exp/field54_obs.py).
    """
    rr, rc = np.asarray(rows) // f, np.asarray(cols) // f
    off = ~land_c[rr, rc]
    dist_cells = np.zeros(len(rr), dtype=float)
    if coastal == "snap":
        keep = np.ones_like(off)
        if off.any():
            from scipy.ndimage import distance_transform_edt
            dist, (iy, ix) = distance_transform_edt(~land_c, return_indices=True)
            dist_cells[off] = dist[rr[off], rc[off]]
            rr = rr.copy(); rc = rc.copy()
            rr[off], rc[off] = iy[rr[off], rc[off]], ix[rr[off], rc[off]]
    elif coastal == "drop":
        keep = ~off
    else:
        raise ValueError(f"coastal must be 'snap' or 'drop', got {coastal!r}")
    assert land_c[rr[keep], rc[keep]].all()
    return rr, rc, off, keep, dist_cells


def coarsen_bbs(npz_path, land_f, land_c, f, coastal="snap", res_c_km=54.0):
    """Remap the BBS npz to the coarse grid. Returns (arrays dict, report dict).

    Any extra per-observation array in the npz (leading length == len(observed_results), e.g. route /
    observer indices) is carried through unchanged for real observations, dropping rows exactly when the
    observation is dropped; on the regenerated pseudo-zero rows it takes the (required-constant) value
    those arrays hold on the fine pseudo-zero rows.
    """
    d = np.load(npz_path, allow_pickle=True)
    n_obs_all = d["observed_results"].shape[0]
    n_pseudo = int(d["N_pseudo"])
    core = {"obs_rows", "obs_cols", "obs_year", "observed_results", "obs_quality"}
    per_obs = [k for k in d.files if k not in core and np.ndim(d[k]) >= 1
               and d[k].shape[0] == n_obs_all]
    if not np.array_equal(np.asarray(d["land"]).astype(bool), land_f):
        raise ValueError("BBS npz land mask differs from the 27 km ocean mask")
    rows, cols = d["obs_rows"], d["obs_cols"]
    years, counts, qual = d["obs_year"], d["observed_results"], d["obs_quality"]

    # Real observations: parent cell. A parent that is not coarse land (a mostly-water coastal /
    # lakeshore block) is either SNAPPED to the nearest coarse land cell by cell-centre distance
    # (default: these are high-count urban coasts -- Long Beach, Santa Barbara, Rochester, Racine --
    # whose loss would bias the fit) or dropped (``coastal="drop"``).
    real = slice(n_pseudo, None)
    rr, rc, off, keep, d_all = parent_cells(rows[n_pseudo:], cols[n_pseudo:], land_c, f, coastal)
    moved = {"n": int(off.sum()), "sum_counts": int(counts[real][off].sum()),
             "n_positive": int((counts[real][off] > 0).sum()),
             "fine_cells": int(np.unique(np.stack([rows[real][off], cols[real][off]], 1), axis=0).shape[0])
             if off.any() else 0, "mode": coastal}
    if coastal == "snap":
        d_cells = d_all[off]
        moved.update(coarse_cells=int(np.unique(np.stack([rr[off], rc[off]], 1), axis=0).shape[0]) if off.any() else 0,
                     max_snap_cells=float(d_cells.max()) if off.any() else 0.0,
                     max_snap_km=float(d_cells.max() * res_c_km) if off.any() else 0.0,
                     mean_snap_km=float(d_cells.mean() * res_c_km) if off.any() else 0.0)
    dropped = moved

    # Pseudo-zeros: coarse cell iff ALL its fine land children are uninvaded.
    p_years = years[:n_pseudo]
    pyrs = np.unique(p_years)
    U = np.zeros_like(land_f)
    U[rows[:n_pseudo], cols[:n_pseudo]] = True
    # the npz pseudo set is a full (cell x year) cross product; assert, since the rebuild relies on it
    if n_pseudo != int(U.sum()) * pyrs.size:
        raise ValueError("pseudo-zeros are not a full uninvaded-cell x year cross product")
    lf_b = children_blocks(pad_to_blocks(land_f, f, False), f)
    u_b = children_blocks(pad_to_blocks(U, f, False), f)
    all_uninv = (~lf_b | u_b).all(axis=-1) & lf_b.any(axis=-1)
    any_uninv = (lf_b & u_b).any(axis=-1)
    pseudo_c = all_uninv & land_c
    ui_r, ui_c = np.where(pseudo_c)
    pr = np.concatenate([ui_r for _ in pyrs]); pc = np.concatenate([ui_c for _ in pyrs])
    py = np.concatenate([np.full(ui_r.size, y) for y in pyrs])
    out = {
        "obs_rows": np.concatenate([pr, rr[keep]]).astype(int),
        "obs_cols": np.concatenate([pc, rc[keep]]).astype(int),
        "obs_year": np.concatenate([py, years[real][keep]]).astype(int),
        "observed_results": np.concatenate([np.zeros(py.size, int), counts[real][keep]]).astype(int),
        "obs_quality": np.concatenate([np.zeros(py.size, int), qual[real][keep]]).astype(int),
    }
    for k in per_obs:
        a = d[k]
        pv = np.unique(a[:n_pseudo])
        if pv.size > 1:
            raise ValueError(f"per-observation array {k} is not constant on pseudo-zero rows")
        fill = pv[0] if pv.size else np.zeros((), a.dtype)
        out[k] = np.concatenate([np.full((py.size,) + a.shape[1:], fill, dtype=a.dtype),
                                 a[real][keep]])
    # Seed shape: mean over land children (0 off coarse land).
    init_b = children_blocks(pad_to_blocks(np.asarray(d["initpop_shape"], np.float64), f, 0.0), f)
    n_land_child = lf_b.sum(-1)
    init_c = np.where(land_c, (init_b * lf_b).sum(-1) / np.maximum(n_land_child, 1), 0.0).astype(np.float32)
    ny_c, nx_c = land_c.shape
    out.update({
        "Nx": nx_c, "Ny": ny_c, "land": land_c.astype(int), "ocean": (~land_c).astype(int),
        "initpop_shape": init_c, "initpop_rows": np.where(init_c > 0)[0], "initpop_cols": np.where(init_c > 0)[1],
        "N_obs": int(keep.sum()), "N_pseudo": int(py.size),
        "unit_distance": d["unit_distance"], "time": d["time"],
    })
    report = {
        "fine": {"n_real": int(n_obs_all - n_pseudo), "sum_counts": int(counts[real].sum()),
                 "n_pseudo": n_pseudo, "pseudo_cells": int(U.sum()), "pseudo_years": [int(pyrs[0]), int(pyrs[-1])],
                 "initpop_cells": int((np.asarray(d["initpop_shape"]) > 0).sum())},
        "coarse": {"n_real": int(keep.sum()), "sum_counts": int(counts[real][keep].sum()),
                   "n_pseudo": int(py.size), "pseudo_cells": int(pseudo_c.sum()),
                   "cells_with_any_uninvaded_child": int((any_uninv & land_c).sum()),
                   "initpop_cells": int((init_c > 0).sum())},
        "coastal_real_obs": dropped,
        "carried_per_obs_arrays": per_obs,
    }
    return out, report


def coarsen_cube(src_input_dir, land_f, land_c, A, f, out_dir, transform_c):
    """Block-average the base run's Z (gathered at 27 km land cells) to Z_latent_{year}.npy."""
    meta_path = os.path.join(src_input_dir, "metadata.pkl")
    with open(meta_path, "rb") as fh:
        m = pickle.load(fh)
    if not np.array_equal(np.asarray(m["land_mask"]).astype(bool), land_f):
        raise ValueError("source input land mask differs from the 27 km ocean mask")
    T, N, M = int(m["time"]), int(m["N_land"]), int(m["M"])
    zpath = os.path.join(src_input_dir, m["z_gathered_path"])
    z = np.memmap(zpath, dtype="float32", mode="r", shape=(T, N, M))
    cr, cc = np.where(land_c)
    os.makedirs(out_dir, exist_ok=True)
    years = [int(y) for y in np.asarray(m["years"])]
    for t, y in enumerate(years):
        zc = A @ np.asarray(z[t], dtype=np.float64)
        grid = np.full(land_c.shape + (M,), np.nan, dtype=np.float32)
        grid[cr, cc] = zc.astype(np.float32)
        np.save(os.path.join(out_dir, f"Z_latent_{y}.npy"), grid)
    src_contract = dict(m["z_kernel_contract"])
    contract = {
        "schema_version": 2,
        "kernel": src_contract["kernel"], "centered": src_contract["centered"],
        "latent_dim": M, "feature_blocks": src_contract["feature_blocks"],
        "temporal_output": m["path_feature_meta"]["kernel_contract"]["temporal_output"],
        "esk_basis_dir": src_contract["esk_basis_dir"],
        "kernel_contract": "Z(x) dot Z(x') ~= uncentered Ruzicka(x,x') [block mean embedding; see note]",
        "years": years,
        "ensemble_method": src_contract.get("ensemble_method"),
        "ensemble_members": src_contract.get("ensemble_members"),
        "desk_checkpoint": src_contract.get("desk_checkpoint"),
        "coarsened_from": {
            "method": f"mean over 27 km LAND children of each {f}x{f} block (sparse averaging operator)",
            "factor": f,
            "source_input_dir": os.path.realpath(src_input_dir),
            "source_z": {"name": os.path.basename(zpath), "size": os.path.getsize(zpath),
                         "content_sha256": sha256(zpath)},
            "source_metadata": {"name": "metadata.pkl", "content_sha256": sha256(meta_path)},
            "source_ingest_id": m.get("ingest_id"),
            "source_seasonal_derivation": {k: v for k, v in (src_contract.get("seasonal_derivation") or {}).items()
                                           if k != "source_contract"},
            "source_grid_shape": list(land_f.shape),
            "grid_shape": list(land_c.shape),
            "transform": list(transform_c)[:6],
        },
        "coarsening_note": CUBE_NOTE,
    }
    from src.community_encoder.cube_contract import validate_cube_contract
    validate_cube_contract(contract, M)
    with open(os.path.join(out_dir, "cube_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(contract, fh, indent=2)
    return {"years": [years[0], years[-1]], "M": M, "n_years": len(years),
            "source_z_sha256": contract["coarsened_from"]["source_z"]["content_sha256"]}


def coarsen_zones(zones_path, land_f, land_c, f):
    """Majority zone code over land children (all children off land). Ties -> barrier code 2 if
    tied, else the lower code."""
    with rasterio.open(zones_path) as src:
        z = src.read(1)
        nodata = src.nodata
    zb = children_blocks(pad_to_blocks(z.astype(np.int64), f, 0), f)
    lb = children_blocks(pad_to_blocks(land_f, f, False), f)
    use = np.where(land_c[..., None], lb, True) & (zb > 0)
    codes = np.arange(1, int(z.max()) + 1)
    cnt = np.stack([((zb == k) & use).sum(-1) for k in codes], -1)
    best = cnt.max(-1)
    tied = (cnt == best[..., None]).sum(-1) > 1
    out = codes[np.argmax(cnt, axis=-1)]
    if 2 in codes:
        out = np.where(tied & (cnt[..., 1] == best), 2, out)
    out = np.where(best > 0, out, 0).astype(np.uint8)
    return out, {"ties": int((tied & (best > 0)).sum()), "nodata": nodata,
                 "counts": {int(k): int((out == k).sum()) for k in codes}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--source-input-dir", default=DEFAULT_SOURCE_INPUT)
    ap.add_argument("--mask", default=f"{DATA}/land_mask/ocean_mask_27km.tif")
    ap.add_argument("--bbs-npz", default=f"{DATA}/bbs_2026_release/bbs_data_for_python.npz")
    ap.add_argument("--zones", default=f"{PROCESSED}/regions/great_plains_zones_27km.tif")
    ap.add_argument("--land-source-dir", default=f"{DATA}/land_source")
    ap.add_argument("--exclude-iso-a3", default="CUB,BHS")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--fine-factor", type=int, default=25, help="the FINE build's subpixels per cell")
    ap.add_argument("--cube-name", default="cube_s64")
    ap.add_argument("--coastal", choices=["snap", "drop"], default="snap",
                    help="real obs whose parent is not coarse land: snap to nearest coarse land (default) or drop")
    ap.add_argument("--out-root", default=None, help="default: $HOUFIN_PROCESSED/grid{res}")
    ap.add_argument("--bbs-only", action="store_true",
                    help="only coarsen --bbs-npz onto the existing coarse mask; output keeps its file name")
    a = ap.parse_args()
    f = int(a.factor)

    land_f, mmeta = read_land_mask(a.mask, return_meta=True)
    with rasterio.open(a.mask) as src:
        t_f, crs = src.transform, src.crs
    res_c = int(round(t_f.a * f))
    res_km = res_c // 1000
    out = Path(a.out_root or f"{PROCESSED}/grid{res_km}")
    t_c = Affine(t_f.a * f, t_f.b, t_f.c, t_f.d, t_f.e * f, t_f.f)   # same origin
    shape_c = coarse_shape(*land_f.shape, f)
    for sub in ("land_mask", "bbs", "regions", "encoder"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    report = {"factor": f, "fine_grid": list(land_f.shape), "coarse_grid": list(shape_c),
              "cell_km": [t_f.a / 1000, res_c / 1000], "origin": [t_f.c, t_f.f], "crs": str(crs)}

    if a.bbs_only:
        # Coarsen another BBS npz onto the EXISTING coarse mask (e.g. the observer-route variant):
        # nothing else is rebuilt, and the output keeps the source npz's file name.
        mask_path = out / "land_mask" / f"ocean_mask_{res_km}km.tif"
        land_c = read_land_mask(mask_path)
        arrays, rep = coarsen_bbs(a.bbs_npz, land_f, land_c, f, a.coastal, res_c / 1000.0)
        arrays["coarsening_manifest"] = np.array(json.dumps({
            "source_npz": os.path.realpath(a.bbs_npz), "source_sha256": sha256(a.bbs_npz), "factor": f,
            "mask": str(mask_path), "mask_sha256": sha256(mask_path)}))
        bbs_path = out / "bbs" / os.path.basename(a.bbs_npz)
        np.savez(bbs_path, **arrays)
        report["bbs"] = rep
        report["outputs"] = {"bbs_npz": str(bbs_path), "mask": str(mask_path)}
        with open(out / f"coarsen_manifest_{Path(a.bbs_npz).stem}.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(json.dumps(report, indent=2, default=str))
        return

    # Ref grid (geometry only).
    ref = out / f"ref_grid_{res_km}km.tif"
    with rasterio.open(ref, "w", driver="GTiff", height=shape_c[0], width=shape_c[1], count=1,
                       dtype="uint8", crs=crs, transform=t_c, nodata=0) as dst:
        dst.write(np.zeros(shape_c, np.uint8), 1)

    # Land mask: reproducibility check of the fine build, then the native coarse build.
    iso = [x for x in a.exclude_iso_a3.split(",") if x]
    frac_f = land_fraction_native(a.land_source_dir, t_f, land_f.shape, land_f.shape, 1, a.fine_factor, iso)
    repro_diff = int(((frac_f >= a.tau) != land_f).sum())
    frac_c = land_fraction_native(a.land_source_dir, t_c, shape_c, land_f.shape, f, a.fine_factor, iso)
    lf_b = children_blocks(pad_to_blocks(land_f, f, False), f)
    has_child = lf_b.any(-1)
    native = frac_c >= a.tau
    land_c = native & has_child
    frac_mean_children = children_blocks(pad_to_blocks(frac_f, f, 0.0), f).mean(-1)
    report["land_mask"] = {
        "fine_rebuild_cells_differing_from_existing_27km_mask": repro_diff,
        "coarse_fraction_vs_mean_of_fine_fractions_max_abs": float(np.abs(frac_c - frac_mean_children).max()),
        "native_land": int(native.sum()), "has_land_child": int(has_child.sum()),
        "native_without_land_child": int((native & ~has_child).sum()),
        "land_child_but_not_native": int((has_child & ~native).sum()),
        "fine_land_cells_in_non_land_parents": int((lf_b.sum(-1) * (~land_c)).sum()),
        "final_land": int(land_c.sum()), "tau": a.tau, "fine_factor_coarse": f * a.fine_factor,
    }
    mask_path = out / "land_mask" / f"ocean_mask_{res_km}km.tif"
    write_mask(mask_path, ~land_c, t_c, crs)
    np.save(out / "land_mask" / f"land_fraction_{res_km}km.npy", frac_c.astype(np.float32))

    # BBS.
    arrays, rep = coarsen_bbs(a.bbs_npz, land_f, land_c, f, a.coastal, res_c / 1000.0)
    arrays["coarsening_manifest"] = np.array(json.dumps({
        "source_npz": os.path.realpath(a.bbs_npz), "source_sha256": sha256(a.bbs_npz), "factor": f,
        "mask": str(mask_path)}))
    bbs_path = out / "bbs" / "bbs_data_for_python.npz"
    np.savez(bbs_path, **arrays)
    report["bbs"] = rep

    # Z cube.
    A, parent, n_child = averaging_operator(land_f, land_c, f)
    sp.save_npz(out / "land_mask" / "averaging_operator.npz", A)
    report["children_per_coarse_land_cell"] = {str(k): int((n_child == k).sum()) for k in range(1, f * f + 1)}
    report["cube"] = coarsen_cube(a.source_input_dir, land_f, land_c, A, f,
                                  str(out / "encoder" / f"{a.cube_name}_grid{res_km}"), t_c)

    # Great Plains zones.
    if os.path.exists(a.zones):
        zc, zrep = coarsen_zones(a.zones, land_f, land_c, f)
        zpath = out / "regions" / f"great_plains_zones_{res_km}km.tif"
        with rasterio.open(zpath, "w", driver="GTiff", height=shape_c[0], width=shape_c[1], count=1,
                           dtype="uint8", crs=crs, transform=t_c, nodata=0) as dst:
            dst.write(zc, 1)
        report["zones"] = zrep

    report["outputs"] = {"ref_grid": str(ref), "mask": str(mask_path), "mask_sha256": sha256(mask_path),
                         "bbs_npz": str(bbs_path)}
    with open(out / "coarsen_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

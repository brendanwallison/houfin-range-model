"""Verify the 54 km age-model inputs against the 27 km base run they were coarsened from.

Built by src/processing/coarsen_model_grid.py --factor 2 (+ the native rebuilds: Z_disp, disease map,
forward kernels via model_inputs). Record: research/experiments/GRID54_INPUTS_2026-10-29.md.

  (1) Z: block mean embedding identity z_bar_B.z_bar_B' == mean child-pair z_i.z_j (sampled pairs);
      truncate-then-average == average-then-truncate; |z_bar|^2 / mean|z_i|^2 distribution.
  (2) observations: real count / count sum preserved minus documented drops; pseudo-zero cells are a
      subset (every land child uninvaded); final ingested obs and pre-invasion absence weight.
  (3) every coarse land cell has >= 1 fine land child; averaging operator rows sum to 1.
  (4) rebuilt 54 km Z_disp vs block mean of the 27 km Z_disp, per-feature correlation.
  (5) kernels: mass, self-cell weight, realized MDD at 27/54 km, point vs cell-integrated, and the
      54 km kernels vs the 27 km kernel AGGREGATED to 54 km blocks; stored kernels == recomputed.
  (6) disease arrival: 54 km map vs block mean of the 27 km map (|diff| < 0.1 yr).
  (7) tensor sizes and device-memory estimate; holdout-block geometry; invasion sites.

    research/bin/rx py research/exp/grid54_verify.py [--z-disp-every 4]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import rasterio
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

P = "/work/07980/bwa386/ls6/houfin/processed"
SRC = f"{P}/model_inputs/numpyro_input_fd_clean_v2_m2_shared_s64"
G = f"{P}/grid54"
DST = f"{G}/model_inputs/numpyro_input_s64_grid54"
FINE_NPZ = "/home/breallis/houfin/scratch/houfin/data/bbs_2026_release/bbs_data_for_python.npz"
F = 2


def load_meta(d):
    with open(os.path.join(d, "metadata.pkl"), "rb") as fh:
        return pickle.load(fh)


def zmem(d, m, disp=False):
    if disp:
        return np.memmap(os.path.join(d, m["z_disp_gathered_path"]), dtype="float32", mode="r",
                         shape=(m["time"], m["N_land"], m["K"], m["M"]))
    return np.memmap(os.path.join(d, m["z_gathered_path"]), dtype="float32", mode="r",
                     shape=(m["time"], m["N_land"], m["M"]))


def q(a, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    a = np.asarray(a, dtype=np.float64)
    return {f"q{x}": float(np.percentile(a, x)) for x in qs}


def operator(mf, mc):
    """Fine-land -> coarse-land mean operator, recomputed independently of the saved one."""
    lf, lc = mf["land_mask"].astype(bool), mc["land_mask"].astype(bool)
    cidx = np.full(lc.shape, -1, np.int64)
    cidx[mc["land_rows"], mc["land_cols"]] = np.arange(mc["N_land"])
    parent = cidx[mf["land_rows"] // F, mf["land_cols"] // F]
    keep = parent >= 0
    n = np.bincount(parent[keep], minlength=mc["N_land"])
    A = sp.csr_matrix((1.0 / n[parent[keep]], (parent[keep], np.nonzero(keep)[0])),
                      shape=(mc["N_land"], mf["N_land"]))
    return A, parent, n, lf, lc


def test_z(mf, mc, A, parent, rng, out):
    zf, zc = zmem(SRC, mf), zmem(DST, mc)
    res = {"pairs": {}, "trunc": {}, "norm_ratio": {}}
    children = [np.nonzero(parent == b)[0] for b in range(mc["N_land"])]
    errs, ratios, trunc = [], [], []
    for t in rng.choice(mf["time"], 6, replace=False):
        Zf = np.asarray(zf[t], np.float64)
        Zc = np.asarray(zc[t], np.float64)
        Zbar = A @ Zf
        assert np.allclose(Zbar, Zc, atol=1e-5), "stored coarse Z != block mean of fine Z"
        for _ in range(500):
            b1, b2 = rng.integers(mc["N_land"], size=2)
            lhs = Zc[b1] @ Zc[b2]
            rhs = (Zf[children[b1]] @ Zf[children[b2]].T).mean()
            errs.append(abs(lhs - rhs) / max(abs(rhs), 1e-12))
        trunc.append(float(np.abs(A @ Zf[:, :24] - (A @ Zf)[:, :24]).max()))
        sq = (Zf ** 2).sum(1)
        mean_sq = np.bincount(parent[parent >= 0], weights=sq[parent >= 0],
                              minlength=mc["N_land"]) / np.bincount(parent[parent >= 0], minlength=mc["N_land"])
        ratios.append((Zbar ** 2).sum(1) / mean_sq)
    out["z"] = {"mean_embedding_identity_max_rel_err": float(np.max(errs)),
                "mean_embedding_identity_n_pairs": len(errs),
                "truncate_then_average_vs_average_then_truncate_max_abs": float(np.max(trunc)),
                "norm_ratio_zbar2_over_mean_zi2": q(np.concatenate(ratios)),
                "stored_coarse_Z_equals_block_mean": True}


def test_obs(mf, mc, out):
    fn = np.load(FINE_NPZ)
    cn = np.load(f"{G}/bbs/bbs_data_for_python.npz", allow_pickle=True)
    man = json.load(open(f"{G}/coarsen_manifest.json"))["bbs"]
    npf, npc = int(fn["N_pseudo"]), int(cn["N_pseudo"])
    yf, yc = fn["observed_results"][npf:], cn["observed_results"][npc:]
    d = man["coastal_real_obs"]
    lost_n, lost_s = (0, 0) if d["mode"] == "snap" else (d["n"], d["sum_counts"])
    ok_n = yf.size - lost_n == yc.size
    ok_s = int(yf.sum()) - lost_s == int(yc.sum())
    # pseudo-zero subset: every land child of a coarse pseudo cell is a fine pseudo cell
    U = np.zeros(fn["land"].shape, bool); U[fn["obs_rows"][:npf], fn["obs_cols"][:npf]] = True
    lf = fn["land"].astype(bool)
    pc = np.unique(np.stack([cn["obs_rows"][:npc], cn["obs_cols"][:npc]], 1), axis=0)
    viol = 0
    for r, c in pc:
        blk_l = lf[2 * r:2 * r + 2, 2 * c:2 * c + 2]; blk_u = U[2 * r:2 * r + 2, 2 * c:2 * c + 2]
        viol += int((blk_l & ~blk_u).any() or not blk_l.any())
    # dropped observations: where (lat band) and how big
    rows, cols = fn["obs_rows"][npf:], fn["obs_cols"][npf:]
    lc = cn["land"].astype(bool)
    drop = ~lc[rows // 2, cols // 2]
    top = {}
    if drop.any():
        cells, inv = np.unique(np.stack([rows[drop], cols[drop]], 1), axis=0, return_inverse=True)
        s = np.bincount(inv.ravel(), weights=yf[drop])
        o = np.argsort(-s)[:8]
        top = {f"{int(cells[i, 0])},{int(cells[i, 1])}": float(s[i]) for i in o}
    # ingested: pseudo-zeros = obs in pre-invasion years (t < inv_timestep)
    def ingested(m):
        pre = m["obs_time_indices"] < m["inv_timestep"]
        cells = np.unique(np.stack([m["obs_rows"][pre], m["obs_cols"][pre]], 1), axis=0).shape[0]
        return {"n_obs": int(m["observed_results"].size), "n_real": int((~pre).sum()),
                "sum_counts": int(m["observed_results"].sum()),
                "n_pseudo_obs": int(pre.sum()), "pseudo_sites": cells}
    i27, i54 = ingested(mf), ingested(mc)
    for i, km in ((i27, 27), (i54, 54)):
        i["pseudo_area_km2"] = i["pseudo_sites"] * km * km
    # snapped observations land on coarse land, and every real obs sits within max_snap of its parent
    dist = np.hypot(cn["obs_rows"][npc:] - rows // 2, cn["obs_cols"][npc:] - cols // 2) if d["mode"] == "snap" else None
    out["obs"] = {"npz_real_count_preserved_minus_drops": bool(ok_n),
                  "npz_count_sum_preserved_minus_drops": bool(ok_s),
                  "npz_fine": man["fine"], "npz_coarse": man["coarse"], "coastal": d,
                  "coastal_top_fine_cells_by_count_sum": top,
                  "snap_check": None if dist is None else {
                      "all_real_obs_on_coarse_land": bool(lc[cn["obs_rows"][npc:], cn["obs_cols"][npc:]].all()),
                      "n_moved": int((dist > 0).sum()), "max_cells": float(dist.max()),
                      "n_moved_2_cells": int((dist >= 1.5).sum()),
                      "same_order_years_counts": bool(np.array_equal(cn["obs_year"][npc:], fn["obs_year"][npf:])
                                                      and np.array_equal(yc, yf))},
                  "pseudo_subset_violations": viol, "pseudo_coarse_cells": int(pc.shape[0]),
                  "ingested_27km": i27, "ingested_54km": i54,
                  "pseudo_obs_ratio_54_over_27": i54["n_pseudo_obs"] / i27["n_pseudo_obs"],
                  "pseudo_area_ratio_54_over_27": i54["pseudo_area_km2"] / i27["pseudo_area_km2"]}


def test_land(n, A, lc, out):
    rs = np.asarray(A.sum(1)).ravel()
    out["land"] = {"coarse_land": int(lc.sum()), "min_land_children": int(n.min()),
                   "children_hist": {str(k): int((n == k).sum()) for k in range(1, 5)},
                   "operator_row_sum_max_dev": float(np.abs(rs - 1).max()),
                   "saved_operator_equal": bool(abs(sp.load_npz(f"{G}/land_mask/averaging_operator.npz") - A).max() < 1e-12)}


def test_zdisp(mf, mc, A, every, out):
    df, dc = zmem(SRC, mf, True), zmem(DST, mc, True)
    K, M = mc["K"], mc["M"]
    sxy = np.zeros((K, M)); sx = np.zeros((K, M)); sy = np.zeros((K, M))
    sxx = np.zeros((K, M)); syy = np.zeros((K, M)); n = 0
    rel = []
    for t in range(0, mf["time"], every):
        xf = np.asarray(df[t], np.float64).reshape(mf["N_land"], K * M)
        x = (A @ xf).reshape(mc["N_land"], K, M)          # block mean of the 27 km Z_disp
        y = np.asarray(dc[t], np.float64)                  # natively rebuilt 54 km Z_disp
        sxy += (x * y).sum(0); sx += x.sum(0); sy += y.sum(0)
        sxx += (x * x).sum(0); syy += (y * y).sum(0); n += x.shape[0]
        rel.append(np.sqrt(((x - y) ** 2).sum((0, 2)) / (x ** 2).sum((0, 2))))
    cov = sxy / n - sx * sy / n ** 2
    r = cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))
    out["z_disp"] = {"years_used": len(range(0, mf["time"], every)),
                     "per_feature_corr_pooled_over_kernels": q(r.mean(0)),
                     "per_feature_corr_all_kernel_feature_pairs": q(r.ravel()),
                     "per_kernel_median_corr": {lab: float(np.median(r[k])) for k, lab in
                                                enumerate(mc["juvenile_kernel_labels"])},
                     "per_kernel_rel_rms_diff": dict(zip(mc["juvenile_kernel_labels"],
                                                         np.mean(rel, 0).round(4).tolist())),
                     "leading8_feature_corr_pooled": r.mean(0)[:8].round(4).tolist(),
                     "note": "27 km Z_disp used point kernels; 54 km uses cell-integrated kernels"}


def kernel_stats(K, cell):
    from src.model.build_kernels import toroidal_distance_grid
    Ly, Lx = K.shape
    r = np.asarray(toroidal_distance_grid(Lx, Ly, cell))
    return {"mass": float(K.sum()), "self": float(K[0, 0]), "mdd_centre_km": float((K * r).sum() / K.sum())}


def aggregate(K27, Ly54, Lx54):
    """27 km toroidal kernel -> 54 km block-to-block transfer for a uniform source block:
    K54(D) = 1/4 sum_{e in {-1,0,1}^2} m(ey) m(ex) K27(2D + e), m = (1, 2, 1)."""
    Ly, Lx = K27.shape
    m = {-1: 1.0, 0: 2.0, 1: 1.0}
    out = np.zeros((Ly54, Lx54))
    dy = np.concatenate([np.arange(Ly54 // 2 + 1), -np.arange(Ly54 // 2, 0, -1)])
    dx = np.concatenate([np.arange(Lx54 // 2 + 1), -np.arange(Lx54 // 2, 0, -1)])
    for ey in (-1, 0, 1):
        for ex in (-1, 0, 1):
            iy = (2 * dy + ey) % Ly; ix = (2 * dx + ex) % Lx
            out += 0.25 * m[ey] * m[ex] * K27[np.ix_(iy, ix)]
    return out


def test_kernels(mf, mc, out):
    import jax.numpy as jnp
    from src.model.build_kernels import (build_simulation_struct, make_juvenile_kernel_stack,
                                         radial_kernel_grid)
    d = mc["dispersal_spec"]
    disc = mc.get("kernel_discretization")
    res = {"discretization_54km_inputs": disc}
    # A large common toroidal frame (in km) so the 27 km kernel can be aggregated exactly onto 54 km.
    L54 = (265, 447)                        # 54 km cells ~ 14310 x 24138 km: no wrap-around mass
    L27 = (2 * L54[0] + 1, 2 * L54[1] + 1)
    integ = {"method": "cell_integrated", "subsamples": int((disc or {}).get("subsamples", 31))}
    for who, mdd, shp in (("adult", d["adult_mdd_km"], d["adult_shape"]),
                          ("juvenile", d["juvenile_mdd_km"], d["juvenile_shape"])):
        r = {}
        k27 = {}
        for name, dz in (("point", None), ("integrated", integ)):
            k27[name] = np.asarray(radial_kernel_grid(L27[1], L27[0], 27.0, mdd, shp, dz), np.float64)
            k54 = np.asarray(radial_kernel_grid(L54[1], L54[0], 54.0, mdd, shp, dz), np.float64)
            r[f"27km_{name}"] = kernel_stats(k27[name], 27.0)
            r[f"54km_{name}"] = kernel_stats(k54, 54.0)
            r[f"54km_{name}"]["k54"] = k54
        for name in ("point", "integrated"):
            agg = aggregate(k27[name], *L54)
            st = kernel_stats(agg, 54.0)
            r[f"27km_{name}_aggregated_to_54"] = st
            for nat in ("point", "integrated"):
                k54 = r[f"54km_{nat}"]["k54"]
                st[f"L1_vs_54km_{nat}"] = float(np.abs(agg - k54).sum())
        for nat in ("point", "integrated"):
            r[f"54km_{nat}"].pop("k54")
        res[who] = r
    # Juvenile radial band mass shares at the production frame sizes.
    shares = {}
    for km, (Ny, Nx) in ((27, (133, 224)), (54, (67, 112))):
        for name, dz in (("point", None), ("integrated", integ)):
            st, labels = make_juvenile_kernel_stack(2 * Nx - 1, 2 * Ny - 1, float(km),
                                                    d["juvenile_radial_splits_km"], mean_dist=d["juvenile_mdd_km"],
                                                    shape=d["juvenile_shape"], discretization=dz)
            st = np.asarray(st)
            bands = st.reshape(4, -1, *st.shape[1:]).sum((0, 2, 3))
            shares[f"{km}km_{name}"] = {"stack_mass": float(st.sum()), "band_shares": bands.round(4).tolist()}
    res["juvenile_band_shares"] = shares
    # The stored 54 km forward-model kernels are exactly what the shared builder gives.
    ss = build_simulation_struct(jnp.array(mc["land_mask"]), float(mc["path_feature_meta"]["cell_size_km"]),
                                 d["adult_mdd_km"], d["juvenile_mdd_km"], d["adult_shape"], d["juvenile_shape"],
                                 d["juvenile_radial_splits_km"], discretization=disc)
    res["stored_54km_kernels_match_builder"] = {
        k: float(np.abs(np.asarray(ss[k]) - mc[k]).max()) for k in
        ("adult_fft_kernel", "juvenile_fft_kernel_stack", "adult_edge_correction", "juvenile_edge_correction_stack")}
    res["path_meta"] = {k: mc["path_feature_meta"].get(k) for k in
                        ("kernel_mass", "realized_discrete_mdd_km", "kernel_discretization", "cell_size_km")}
    res["path_meta_27km"] = {k: mf["path_feature_meta"].get(k) for k in
                             ("kernel_mass", "realized_discrete_mdd_km", "cell_size_km")}
    out["kernels"] = res


def test_disease(mf, mc, A, out):
    f27 = mf["disease_onset"].astype(np.float64)
    c54 = mc["disease_onset"].astype(np.float64)
    diff = c54 - A @ f27
    out["disease"] = {"max_abs_diff_years": float(np.abs(diff).max()),
                      "mean_abs_diff_years": float(np.abs(diff).mean()),
                      "q99_abs_diff": float(np.percentile(np.abs(diff), 99)),
                      "map": mc["disease_arrival_map"]}
    regen = "/mnt/c/Users/brend/AppData/Local/Temp/claude/c--Dev-houfin-range-model/aecc69d0-0eb4-4a19-bcd2-5f6636dce536/scratchpad/disease27/disease_arrival_year_27km.tif"
    if os.path.exists(regen):
        with rasterio.open(regen) as s:
            a = s.read(1)[mf["land_rows"], mf["land_cols"]] - 1902.0
        out["disease"]["regenerated_27km_vs_stored_onset_max_abs"] = float(np.abs(a - f27).max())


def holdout(m, b, frac=0.1, seed=70):
    rows, cols = np.asarray(m["obs_rows"]).astype(np.int64), np.asarray(m["obs_cols"])
    bid = (rows // b) * 100000 + cols // b
    ids = np.unique(bid)
    rng = np.random.default_rng(seed)
    pick = rng.choice(ids, size=max(1, int(round(frac * ids.size))), replace=False)
    years = np.asarray(m["years"])[np.asarray(m["obs_time_indices"])]
    held = np.isin(bid, pick) | np.isin(years, np.arange(2021, 2026))
    return set(pick.tolist()), ids.size, held


def test_sizes(mf, mc, out):
    def nb(m):
        return {"Z_GB": m["time"] * m["N_land"] * m["M"] * 4 / 1e9,
                "Z_disp_GB": m["time"] * m["N_land"] * m["K"] * m["M"] * 4 / 1e9,
                "fft_grid": list(np.asarray(m["adult_fft_kernel"]).shape),
                "N_land": int(m["N_land"]), "grid": [int(m["Ny"]), int(m["Nx"])], "M": int(m["M"]), "K": int(m["K"])}
    a, b = nb(mf), nb(mc)
    ratio = (b["Z_GB"] + b["Z_disp_GB"]) / (a["Z_GB"] + a["Z_disp_GB"])
    # 27 km MAP (s64/map.log): inputs resident 6.63 GiB, device peak 12.74 GiB. Scale the input part by the
    # input ratio and the remainder (forward-pass/FFT working set, ~proportional to cells) by N_land ratio.
    cell_ratio = b["N_land"] / a["N_land"]
    est = 6.63 * ratio + (12.74 - 6.63) * cell_ratio
    s27, n27, h27 = holdout(mf, 8)
    s54, n54, h54 = holdout(mc, 4)
    # The fit code's pinned path (GRID54_LADDER snapshot data_loading, holdout_blocks.pinned_blocks).
    pin_path = f"{G}/holdout_blocks_216km_s64.json"
    pinned = None
    if os.path.exists(pin_path):
        pin = json.load(open(pin_path))
        pid = {int(r) * 100000 + int(c) for r, c in pin["blocks"]}
        rows, cols = np.asarray(mc["obs_rows"]).astype(np.int64), np.asarray(mc["obs_cols"])
        bid = (rows // 4) * 100000 + cols // 4
        years = np.asarray(mc["years"])[np.asarray(mc["obs_time_indices"])]
        sp_held = np.isin(bid, list(pid))
        held = sp_held | np.isin(years, np.arange(2021, 2026))
        real = ~(np.asarray(mc["obs_time_indices"]) < mc["inv_timestep"])
        rf = np.asarray(mf["obs_time_indices"]) >= mf["inv_timestep"]
        pinned = {"n_blocks": len(pid), "pinned_equals_27km_draw": pid == s27,
                  "withheld_share_54": float(held.mean()), "withheld_share_27": float(h27.mean()),
                  "spatial_withheld_share_54": float(sp_held.mean()),
                  "spatial_withheld_share_27": pin["source"]["share_withheld_spatial"],
                  "withheld_real_share_54": float(held[real].mean()),
                  "withheld_real_share_27": float(h27[rf].mean()),
                  "withheld_count_sum_54": int(np.asarray(mc["observed_results"])[held].sum()),
                  "withheld_count_sum_27": int(np.asarray(mf["observed_results"])[h27].sum())}
    out["sizes"] = {"27km": a, "54km": b, "input_ratio": ratio,
                    "device_peak_estimate_GiB": est, "device_peak_27km_GiB": 12.74,
                    "holdout": {"27km_blocks_with_obs": n27, "54km_blocks_with_obs": n54,
                                "picked_27": len(s27), "picked_54": len(s54),
                                "picked_identical": s27 == s54, "picked_overlap": len(s27 & s54),
                                "withheld_frac_27": float(h27.mean()), "withheld_frac_54_unpinned": float(h54.mean()),
                                "pinned": pinned},
                    "inv_locations_27": np.asarray(mf["inv_locations"]).tolist(),
                    "inv_locations_54": np.asarray(mc["inv_locations"]).tolist(),
                    "inv_parents_of_27": sorted({(int(r) // 2, int(c) // 2) for r, c in mf["inv_locations"]})}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z-disp-every", type=int, default=4)
    ap.add_argument("--out", default=f"{G}/grid54_verify.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    mf, mc = load_meta(SRC), load_meta(DST)
    A, parent, n, lf, lc = operator(mf, mc)
    out = {}
    for name, fn in (("land", lambda: test_land(n, A, lc, out)),
                     ("z", lambda: test_z(mf, mc, A, parent, rng, out)),
                     ("obs", lambda: test_obs(mf, mc, out)),
                     ("disease", lambda: test_disease(mf, mc, A, out)),
                     ("kernels", lambda: test_kernels(mf, mc, out)),
                     ("sizes", lambda: test_sizes(mf, mc, out)),
                     ("z_disp", lambda: test_zdisp(mf, mc, A, a.z_disp_every, out))):
        print(f"[verify] {name} ...", flush=True)
        fn()
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

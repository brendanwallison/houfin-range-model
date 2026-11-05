"""Verify the 54 km field_d_interp summer-64 input and compare its habitat Z with the clean-v2 s64 Z.

Record: research/experiments/GRID54_INPUTS_FDINTERP_2026-10-29.md. Mirrors grid54_verify.py /
grid54_obsroute_verify.py for the pieces that change with the cube:

  (1) Z: stored 54 km Z (cube and ingested Z_gathered) == block mean of the 27 km summer columns;
      kernel mean-embedding identity on sampled block pairs; |z_bar|^2 / mean |z_i|^2.
  (2) Z_disp: Z_raw inside each Z_disp npz == the cube; ingested Z_disp == the npz values on land;
      linearity: path operator applied to (Z_fd - Z_s64) == Z_disp_fd - Z_disp_s64 (same operator as the
      clean input); 54 km Z_disp vs block mean of a 27 km Z_disp of the SAME cube computed here with the
      production 27 km point kernels (none exists on disk), with the identical computation for the s64 cube
      as the reference (its 27 km Z_disp is stored; reproduced first to validate this code path).
  (3) metadata: every key except Z-derived ones identical to numpyro_input_s64_grid54_obsroute
      (observations, observer ids, pseudo-zeros, founders, kernels, disease, priors), pinned holdout mask.
  (4) tensor shapes / bytes identical to the obsroute input (device memory as measured there).
  (5) how different the habitat input is: field_d_interp vs clean-v2 s64 at 54 km, overall and in
      temporal change (1902-15 -> 2010-25), coordinate-wise and basis-invariant (Gram / linear map); the
      backcast40 control_s1 cube (same teacher basis as field_d_interp) as a reference.

    research/bin/rx py research/exp/grid54_fdinterp_verify.py
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

P = "/work/07980/bwa386/ls6/houfin/processed"
G = f"{P}/grid54"
CUBE = f"{G}/encoder/cube_fdinterp_s64_grid54"
CUBE_S64 = f"{G}/encoder/cube_s64_grid54"
ZD = f"{G}/latent_avian_paths_fdinterp_s64_grid54"
ZD_S64 = f"{G}/latent_avian_paths_s64_grid54"
NEW = f"{G}/model_inputs/numpyro_input_fdinterp_s64_grid54_obsroute"
OBS = f"{G}/model_inputs/numpyro_input_s64_grid54_obsroute"
ROOT = _REPO / ".research_runtime/community_z_repair_20261026/cubes/backcast40"
SRC27 = ROOT / "field_d_interp_s1"
CTRL27 = ROOT / "control_s1"
S64_27 = str(_REPO / ".research_runtime/shared_round1_inputs/numpyro_input_fd_clean_v2_m2_shared_s64")
LOW = 1e-4
FINE_MASK = "/home/breallis/houfin/scratch/houfin/data/land_mask/ocean_mask_27km.tif"
YEARS = list(range(1902, 2026))
EARLY, LATE = range(1902, 1916), range(2010, 2026)


def q(a, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    a = np.asarray(a, dtype=np.float64)
    return {f"q{x}": float(np.percentile(a, x)) for x in qs}


def meta(d):
    with open(os.path.join(d, "metadata.pkl"), "rb") as fh:
        return pickle.load(fh)


def zmem(d, m, disp=False):
    if disp:
        return np.memmap(os.path.join(d, m["z_disp_gathered_path"]), dtype="float32", mode="r",
                         shape=(m["time"], m["N_land"], m["K"], m["M"]))
    return np.memmap(os.path.join(d, m["z_gathered_path"]), dtype="float32", mode="r",
                     shape=(m["time"], m["N_land"], m["M"]))


def corr(x, y):
    x = np.asarray(x, np.float64).ravel(); y = np.asarray(y, np.float64).ravel()
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def pooled_coord_corr(x, y):
    """Centred per coordinate over rows, then one pooled Pearson over rows x coordinates."""
    x = x - x.mean(0); y = y - y.mean(0)
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def per_dim_corr(x, y):
    x = x - x.mean(0); y = y - y.mean(0)
    return (x * y).sum(0) / np.sqrt((x * x).sum(0) * (y * y).sum(0))


def gram_corr(x, y, centred=False):
    """Pearson of the off-diagonal inner products x_i.x_j vs y_i.y_j (basis-invariant)."""
    if centred:
        x = x - x.mean(0); y = y - y.mean(0)
    gx, gy = x @ x.T, y @ y.T
    iu = np.triu_indices(x.shape[0], 1)
    return corr(gx[iu], gy[iu])


def cross_fit_r2(x, y, groups, lam=1e-6):
    """Ridge map x -> y (with intercept) fitted on one half of the spatial groups, R^2 on the other, both
    ways round; pooled R^2 = 1 - SSE/SST over all target coordinates."""
    out = []
    for hold in (0, 1):
        tr, te = groups != hold, groups == hold
        X = np.hstack([x[tr], np.ones((tr.sum(), 1))]); Xt = np.hstack([x[te], np.ones((te.sum(), 1))])
        B = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y[tr])
        r = y[te] - Xt @ B
        out.append(1 - (r ** 2).sum() / ((y[te] - y[te].mean(0)) ** 2).sum())
    return float(np.mean(out))


# ---------------------------------------------------------------- (1) Z
def test_z(land_f, A, parent, mN, rng, out):
    rows_c, cols_c = mN["land_rows"], mN["land_cols"]
    zN = zmem(NEW, mN)
    fr, fc = np.where(land_f)
    children = [np.nonzero(parent == b)[0] for b in range(mN["N_land"])]
    errs, ratios, max_cube, max_ing = [], [], 0.0, 0.0
    for t in range(len(YEARS)):
        y = YEARS[t]
        g = np.load(SRC27 / f"Z_latent_{y}.npy")
        Zf = np.asarray(g[fr, fc, :64], np.float64)
        Zbar = A @ Zf
        cube = np.load(f"{CUBE}/Z_latent_{y}.npy")
        Zc = np.asarray(cube[rows_c, cols_c], np.float64)
        max_cube = max(max_cube, float(np.abs(Zc - Zbar).max()))
        max_ing = max(max_ing, float(np.abs(np.asarray(zN[t], np.float64) - Zc).max()))
        if t % 20 == 0:
            for _ in range(500):
                b1, b2 = rng.integers(mN["N_land"], size=2)
                lhs = Zc[b1] @ Zc[b2]
                rhs = (Zf[children[b1]] @ Zf[children[b2]].T).mean()
                errs.append(abs(lhs - rhs) / max(abs(rhs), 1e-12))
            sq = (Zf ** 2).sum(1)
            k = parent >= 0
            mean_sq = np.bincount(parent[k], weights=sq[k], minlength=mN["N_land"]) / np.bincount(parent[k], minlength=mN["N_land"])
            ratios.append((Zbar ** 2).sum(1) / mean_sq)
        # off-land is NaN, on-land finite
        if not (np.isfinite(cube[rows_c, cols_c]).all() and np.isnan(cube[~mN["land_mask"].astype(bool)]).all()):
            raise ValueError(f"cube {y}: land/ocean NaN pattern wrong")
    sel = np.load(SRC27 / "Z_latent_2000.npy")
    sel_err = float(np.abs((A @ np.asarray(sel[fr, fc, :], np.float64))[:, :64] - A @ np.asarray(sel[fr, fc, :64], np.float64)).max())
    out["z"] = {"cube_vs_block_mean_of_27km_summer_max_abs": max_cube,
                "ingested_Z_vs_cube_max_abs": max_ing,
                "mean_embedding_identity_max_rel_err": float(np.max(errs)),
                "mean_embedding_identity_n_pairs": len(errs),
                "select_then_average_vs_average_then_select_max_abs": sel_err,
                "norm_ratio_zbar2_over_mean_zi2": q(np.concatenate(ratios)),
                "years_checked": len(YEARS)}


# ---------------------------------------------------------------- (2) Z_disp
def build_ops():
    import jax.numpy as jnp
    from src.model.build_kernels import make_juvenile_kernel_stack
    from src.model.build_path_features import integrate_paths
    from src.data.masks import read_land_mask
    return jnp, make_juvenile_kernel_stack, integrate_paths, read_land_mask


def min_land_share(ks, land, steps):
    """Per (kernel, row, col): the minimum over integration steps of the share of the (rescaled) band's
    mass that lands on land. Where it is ~0 the normalised convolution divides FFT roundoff by ~0, so the
    stored Z_disp there is roundoff noise rather than a function of Z (a property of the existing operator,
    build_path_features.integrate_paths, at grid-edge cells)."""
    import jax.numpy as jnp
    from jax.numpy.fft import fft2
    from src.model.build_path_features import resize_kernel_stack, convolve_mask_step
    lm = jnp.array(land.astype(np.float32))
    wmin = None
    for s in np.linspace(1.0 / steps, 1.0, steps):
        sk = resize_kernel_stack(ks, s)
        w = np.asarray(convolve_mask_step(lm, fft2(sk))) / np.asarray(sk.sum((1, 2)))[:, None, None]
        wmin = w if wmin is None else np.minimum(wmin, w)
    return wmin


def test_zdisp(land_f, A, mN, mS, out, years27):
    jnp, mk, integ, read_land_mask = build_ops()
    pm = mN["path_feature_meta"]
    d = pm["dispersal"]
    land_c = mN["land_mask"].astype(bool)
    rows_c, cols_c = mN["land_rows"], mN["land_cols"]
    res = {"path_meta": {k: pm.get(k) for k in ("kernel_discretization", "kernel_mass", "realized_discrete_mdd_km",
                                                "integration_steps", "mask_sha256", "build_id")},
           "path_meta_matches_s64_except_build_and_contract": {
               k: pm.get(k) == mS["path_feature_meta"].get(k) for k in pm
               if k not in ("build_id", "kernel_contract")}}
    # (a) npz internal: Z_raw == cube; ingested Z_disp == npz on land
    zdN = zmem(NEW, mN, True)
    raw_err, ing_err = 0.0, 0.0
    for t in range(0, len(YEARS), 3):
        y = YEARS[t]
        f = np.load(f"{ZD}/Z_disp_{y}.npz")
        cube = np.load(f"{CUBE}/Z_latent_{y}.npy")
        zr = f["Z_raw"][0]
        raw_err = max(raw_err, float(np.abs(np.nan_to_num(zr) - np.nan_to_num(cube)).max()))
        disp = f["Z_disp"][0].transpose(0, 1, 3, 2)[rows_c, cols_c]
        ing_err = max(ing_err, float(np.abs(np.asarray(zdN[t]) - disp).max()))
    res["npz_Z_raw_vs_cube_max_abs"] = raw_err
    res["ingested_Z_disp_vs_npz_max_abs"] = ing_err
    # (b) linearity / same operator as the clean s64 build (one year, 54 km, cell-integrated)
    Ny, Nx = land_c.shape
    ks54, _ = mk(2 * Nx - 1, 2 * Ny - 1, 54.0, d["juvenile_radial_splits_km"], mean_dist=d["juvenile_mdd_km"],
                 shape=d["juvenile_shape"], discretization=pm["kernel_discretization"])
    lm54 = jnp.array(land_c.astype(np.float32))
    steps = int(pm["integration_steps"])
    low54 = (min_land_share(ks54, land_c, steps) < LOW).transpose(1, 2, 0)          # (Ny, Nx, K)
    good54 = land_c[..., None, None] & ~low54[:, :, None, :]                           # (Ny, Nx, M=1, K)
    res["low_land_support_pairs_54km"] = {"threshold_min_step_land_share": LOW,
                                          "land_cell_kernel_pairs": int(low54[land_c].sum()),
                                          "of": int(land_c.sum() * low54.shape[-1]),
                                          "land_cells_touched": int(low54[land_c].any(-1).sum())}
    y = 1990
    zf = np.load(f"{CUBE}/Z_latent_{y}.npy"); zs = np.load(f"{CUBE_S64}/Z_latent_{y}.npy")
    dif = np.nan_to_num(zf) - np.nan_to_num(zs)
    Ld = np.asarray(integ(jnp.array(dif[None]), ks54, lm54, steps=int(pm["integration_steps"])))[0]
    a = np.load(f"{ZD}/Z_disp_{y}.npz")["Z_disp"][0]; b = np.load(f"{ZD_S64}/Z_disp_{y}.npz")["Z_disp"][0]
    gm = np.broadcast_to(good54, a.shape)
    resid = (a - b - Ld)[gm]
    scale = np.abs((a - b)[gm]).max()
    res["linearity_residual_on_low_support_pairs_max_abs"] = float(np.abs(a - b - Ld)[land_c[..., None, None] & ~gm].max())
    Lf = np.asarray(integ(jnp.array(np.nan_to_num(zf)[None]), ks54, lm54, steps=int(pm["integration_steps"])))[0]
    res["linearity_year"] = y
    res["L(Zfd-Zs64)_vs_(Zdisp_fd-Zdisp_s64)_max_abs_over_max_abs_diff_excl_low_support"] = float(np.abs(resid).max() / scale)
    res["recomputed_vs_stored_Zdisp_fd_max_abs"] = float(np.abs((Lf - a)[land_c]).max())
    # (c) 27 km Z_disp of the same cube (production point kernels), block-meaned, vs 54 km
    m27 = meta(S64_27)
    pm27 = m27["path_feature_meta"]
    d27 = pm27["dispersal"]
    disc27 = pm27.get("kernel_discretization")
    lf_np = read_land_mask(FINE_MASK).astype(bool)
    assert np.array_equal(lf_np, land_f)
    Ny2, Nx2 = land_f.shape
    ks27, labels27 = mk(2 * Nx2 - 1, 2 * Ny2 - 1, 27.0, d27["juvenile_radial_splits_km"],
                        mean_dist=d27["juvenile_mdd_km"], shape=d27["juvenile_shape"], discretization=disc27)
    lm27 = jnp.array(land_f.astype(np.float32))
    fr, fc = np.where(land_f)
    low27 = (min_land_share(ks27, land_f, int(pm27["integration_steps"])) < LOW).transpose(1, 2, 0)   # (Ny, Nx, K)
    low27_land = low27[m27["land_rows"], m27["land_cols"]]                                           # (N27, K)
    res["low_land_support_pairs_27km"] = {"land_cell_kernel_pairs": int(low27_land.sum()),
                                          "of": int(low27_land.size)}
    # 54 km (cell, kernel) pairs used in the comparison: support OK at 54 km and at every 27 km child
    keep_c = ~low54[rows_c, cols_c] & (np.asarray(A @ low27[fr, fc].astype(np.float64)) == 0)        # (N54, K)
    res["comparison_pairs_kept"] = {"kept": int(keep_c.sum()), "of": int(keep_c.size)}
    # land_rows order of the 27 km input == np.where order? (A's column order is np.where order)
    same_order = bool(np.array_equal(m27["land_rows"], fr) and np.array_equal(m27["land_cols"], fc))
    zd27 = zmem(S64_27, m27, True)
    z27 = zmem(S64_27, m27)
    K, M = mN["K"], mN["M"]
    acc = {"fdinterp": [], "s64": []}
    repro = None
    for y in years27:
        t = YEARS.index(y)
        for tag in ("fdinterp", "s64"):
            if tag == "fdinterp":
                g = np.load(SRC27 / f"Z_latent_{y}.npy")[..., :64]
                coarse = np.load(f"{ZD}/Z_disp_{y}.npz")["Z_disp"][0]
                fine_land = np.asarray(integ(jnp.array(np.nan_to_num(g)[None]), ks27, lm27,
                                             steps=int(pm27["integration_steps"])))[0][fr, fc]  # (N, M, K)
                fine_land = fine_land.transpose(0, 2, 1)                                       # (N, K, M)
            else:
                coarse = np.load(f"{ZD_S64}/Z_disp_{y}.npz")["Z_disp"][0]
                fine_land = np.asarray(zd27[t], np.float64)                                    # (N, K, M)
                if repro is None:   # validate this code path once: recompute the stored 27 km s64 Z_disp
                    g = np.full(land_f.shape + (M,), 0.0, np.float32)
                    g[m27["land_rows"], m27["land_cols"]] = z27[t]
                    rc = np.asarray(integ(jnp.array(g[None]), ks27, lm27, steps=int(pm27["integration_steps"])))[0]
                    rc = rc[m27["land_rows"], m27["land_cols"]].transpose(0, 2, 1)
                    ok = ~low27_land[:, :, None]
                    dd = np.abs(rc - fine_land)
                    repro = {"year": y, "max_abs_excl_low_support": float(dd[np.broadcast_to(ok, dd.shape)].max()),
                             "max_abs_value": float(np.abs(fine_land).max()),
                             "max_abs_on_low_support_pairs": float(dd[np.broadcast_to(~ok, dd.shape)].max()),
                             "note": "stored 27 km Z_disp was integrated from the 128-d cube then column-selected"}
            x = (A @ np.asarray(fine_land, np.float64).reshape(fine_land.shape[0], K * M)).reshape(-1, K, M)
            yv = np.asarray(coarse.transpose(0, 1, 3, 2)[rows_c, cols_c], np.float64)   # (N, K, M)
            acc[tag].append((x, yv))
    for tag, pairs in acc.items():
        x = np.concatenate([p[0] for p in pairs]); yv = np.concatenate([p[1] for p in pairs])
        w = np.concatenate([keep_c for _ in pairs])[:, :, None].astype(np.float64)   # (N*years, K, 1)
        n = w.sum(0)
        xm = (w * x).sum(0) / n; ym = (w * yv).sum(0) / n
        xc = (x - xm) * w; yc = (yv - ym) * w
        r = (xc * yc).sum(0) / np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum(0))      # (K, M)
        x = x * w; yv = yv * w
        rel = np.sqrt(((x - yv) ** 2).sum((0, 2)) / (x ** 2).sum((0, 2)))
        res[f"vs_27km_blockmean_{tag}"] = {
            "per_feature_corr_pooled_over_kernels": q(r.mean(0)),
            "per_feature_corr_all_kernel_feature_pairs": q(r.ravel()),
            "per_kernel_rel_rms_diff": dict(zip(mN["juvenile_kernel_labels"], rel.round(4).tolist()))}
    res["27km_years"] = list(years27)
    res["27km_land_order_is_np_where"] = same_order
    res["27km_discretization"] = disc27 or "point (absent record)"
    res["27km_s64_stored_Zdisp_reproduced"] = repro
    out["z_disp"] = res


# ---------------------------------------------------------------- (3)/(4) metadata, holdout, sizes
def test_meta(mN, mO, out):
    zkeys = ("ingest_id", "z_gathered_path", "z_disp_gathered_path", "z_kernel_contract", "path_feature_meta")
    diff, same = [], []
    for k in sorted(set(mN) | set(mO)):
        if k in zkeys:
            continue
        va, vb = mN.get(k), mO.get(k)
        eq = (np.array_equal(np.asarray(va), np.asarray(vb)) if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray)
              else va == vb)
        (same if eq else diff).append(k)
    zc_n, zc_o = mN["z_kernel_contract"], mO["z_kernel_contract"]
    pin = json.load(open(f"{G}/holdout_blocks_216km_s64.json"))
    pid = {int(r) * 100000 + int(c) for r, c in pin["blocks"]}

    def held(m):
        rows, cols = np.asarray(m["obs_rows"]).astype(np.int64), np.asarray(m["obs_cols"])
        bid = (rows // 4) * 100000 + cols // 4
        years = np.asarray(m["years"])[np.asarray(m["obs_time_indices"])]
        return np.isin(bid, list(pid)) | np.isin(years, np.arange(2021, 2026))
    hN, hO = held(mN), held(mO)
    pre = np.asarray(mN["obs_time_indices"]) < mN["inv_timestep"]
    out["metadata"] = {
        "keys_differing_from_obsroute_input_excluding_Z_derived": diff,
        "n_keys_identical": len(same),
        "identical_keys": same,
        "z_kernel_contract_keys_differing": sorted(k for k in set(zc_n) | set(zc_o) if zc_n.get(k) != zc_o.get(k)),
        "z_kernel_contract_new": {k: zc_n.get(k) for k in ("kernel", "centered", "latent_dim", "source_latent_dim",
                                                         "truncation", "esk_basis_dir", "desk_checkpoint")},
        "pinned_holdout": {"identical_mask": bool(np.array_equal(hN, hO)), "withheld": int(hN.sum()),
                           "share": float(hN.mean())},
        "pseudo_zero": {"n_obs": int(pre.sum()),
                        "sites": int(np.unique(np.stack([mN["obs_rows"][pre], mN["obs_cols"][pre]], 1), axis=0).shape[0])},
        "n_obs": int(np.asarray(mN["observed_results"]).size),
        "inv_locations": np.asarray(mN["inv_locations"]).tolist(),
        "inv_timestep": int(mN["inv_timestep"]),
    }
    sizes = {}
    for tag, d, m in (("new", NEW, mN), ("obsroute", OBS, mO)):
        sizes[tag] = {k: os.path.getsize(os.path.join(d, m[k])) for k in ("z_gathered_path", "z_disp_gathered_path")}
        sizes[tag].update(M=int(m["M"]), K=int(m["K"]), N_land=int(m["N_land"]), time=int(m["time"]))
    sizes["identical"] = sizes["new"] == sizes["obsroute"]
    sizes["measured_device_peak_on_obsroute_input_GiB"] = 3.20
    sizes["measured_source"] = "grid54/runs/step3_obsroute/out/map.log ([memory] peak 3.2 GiB of 17.3 GiB)"
    out["sizes"] = sizes


# ---------------------------------------------------------------- (5) habitat comparison
def gather_cube(path, rows, cols, years):
    return np.stack([np.load(f"{path}/Z_latent_{y}.npy")[rows, cols] for y in years]).astype(np.float64)


def test_compare(land_f, A, mN, out):
    rows_c, cols_c = mN["land_rows"], mN["land_cols"]
    fr, fc = np.where(land_f)
    fd = gather_cube(CUBE, rows_c, cols_c, YEARS)          # (T, N, 64)
    s64 = gather_cube(CUBE_S64, rows_c, cols_c, YEARS)
    ctrl = np.stack([A @ np.asarray(np.load(CTRL27 / f"Z_latent_{y}.npy")[fr, fc, :64], np.float64) for y in YEARS])
    T, N, M = fd.shape
    groups = ((rows_c // 4) + (cols_c // 4)) % 2              # 216 km checkerboard halves
    ei = [YEARS.index(y) for y in EARLY]; li = [YEARS.index(y) for y in LATE]
    cubes = {"fdinterp": fd, "s64": s64, "control_s1": ctrl}
    res = {"design": "54 km land cells (4318), summer 64; change = mean 2010-2025 minus mean 1902-1915 per cell; "
                     "coordinate correlations are basis-dependent (fdinterp/control use the backcast40 summer teacher "
                     "basis, s64 the esk_all369_breed basis); Gram correlations and cross-fitted linear-map R^2 "
                     "(216 km checkerboard halves) are basis-invariant",
           "pairs": {}}
    for a_, b_ in (("fdinterp", "s64"), ("control_s1", "s64"), ("fdinterp", "control_s1")):
        x, y = cubes[a_], cubes[b_]
        lvl_x, lvl_y = x.reshape(T * N, M), y.reshape(T * N, M)
        t2000 = YEARS.index(2000)
        dx = x[li].mean(0) - x[ei].mean(0); dy = y[li].mean(0) - y[ei].mean(0)
        gT = np.tile(groups, T)
        r = {
            "level": {
                "pooled_coord_corr_all_cell_years": pooled_coord_corr(lvl_x, lvl_y),
                "per_dim_corr_2000": q(per_dim_corr(x[t2000], y[t2000]), (0, 25, 50, 75, 100)),
                "gram_corr_2000": gram_corr(x[t2000], y[t2000]),
                "gram_corr_2000_centred": gram_corr(x[t2000], y[t2000], centred=True),
                "linear_map_R2_cross_fit_a_to_b": cross_fit_r2(lvl_x[::7], lvl_y[::7], gT[::7]),
                "linear_map_R2_cross_fit_b_to_a": cross_fit_r2(lvl_y[::7], lvl_x[::7], gT[::7]),
            },
            "change": {
                "pooled_coord_corr": pooled_coord_corr(dx, dy),
                "per_dim_corr": q(per_dim_corr(dx, dy), (0, 25, 50, 75, 100)),
                "gram_corr": gram_corr(dx, dy),
                "gram_corr_centred": gram_corr(dx, dy, centred=True),
                "norm_corr_per_cell": corr(np.linalg.norm(dx, axis=1), np.linalg.norm(dy, axis=1)),
                "norm_spearman_per_cell": corr(np.argsort(np.argsort(np.linalg.norm(dx, axis=1))),
                                               np.argsort(np.argsort(np.linalg.norm(dy, axis=1)))),
                "linear_map_R2_cross_fit_a_to_b": cross_fit_r2(dx, dy, groups),
                "linear_map_R2_cross_fit_b_to_a": cross_fit_r2(dy, dx, groups),
                "mean_change_norm_a": float(np.linalg.norm(dx, axis=1).mean()),
                "mean_change_norm_b": float(np.linalg.norm(dy, axis=1).mean()),
                "change_norm_over_level_sd_a": float(np.linalg.norm(dx, axis=1).mean() / np.sqrt(((x[t2000] - x[t2000].mean(0)) ** 2).sum(1).mean())),
                "change_norm_over_level_sd_b": float(np.linalg.norm(dy, axis=1).mean() / np.sqrt(((y[t2000] - y[t2000].mean(0)) ** 2).sum(1).mean())),
            },
        }
        res["pairs"][f"{a_}_vs_{b_}"] = r
        print(a_, b_, json.dumps(r, indent=1), flush=True)
    out["compare"] = res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{G}/grid54_fdinterp_verify.json")
    ap.add_argument("--years27", default="1902,1925,1950,1966,1980,1995,2010,2025")
    ap.add_argument("--only", default="z,meta,compare,zdisp")
    a = ap.parse_args()
    from src.data.masks import read_land_mask
    from src.processing.coarsen_model_grid import averaging_operator
    land_f = read_land_mask(FINE_MASK).astype(bool)
    mN, mO = meta(NEW), meta(OBS)
    land_c = mN["land_mask"].astype(bool)
    A, parent, _ = averaging_operator(land_f, land_c, 2)
    assert abs(sp.load_npz(f"{G}/land_mask/averaging_operator.npz") - A).max() == 0
    assert np.array_equal(np.where(land_c)[0], mN["land_rows"]) and np.array_equal(np.where(land_c)[1], mN["land_cols"])
    out = {}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
    rng = np.random.default_rng(0)
    steps = a.only.split(",")
    if "z" in steps:
        print("[verify] z", flush=True); test_z(land_f, A, parent, mN, rng, out)
    if "meta" in steps:
        print("[verify] meta", flush=True); test_meta(mN, mO, out)
    if "compare" in steps:
        print("[verify] compare", flush=True); test_compare(land_f, A, mN, out)
    if "zdisp" in steps:
        print("[verify] zdisp", flush=True)
        test_zdisp(land_f, A, mN, mO, out, [int(y) for y in a.years27.split(",")])
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "compare"}, indent=2, default=str))


if __name__ == "__main__":
    main()

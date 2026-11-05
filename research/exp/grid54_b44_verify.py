"""Verify the 54 km 4 directions x 4 bands input (outer juvenile band split at 1,000 km) against the 3-band obsroute input.

Run from the code54 snapshot (cell_integrated kernel builder), e.g.
    fd_snaprun.py <code54> research/exp/grid54_b44_verify.py      (AGE_MODEL_CONFIG = grid54_age_b44_obsroute.json)

Checks (output grid54/grid54_b44_verify.json):
  1. operator reproduction: the 3-band kernels rebuilt by this builder (original splits) + integrate_paths reproduce the
     STORED 3-band Z_disp (path-feature npz) in sample years, all 64 features;
  2. the new 16 kernels: bands 0-1 identical to the old bands 0-1, new bands 2 + 3 == old band 2 cell by cell, the
     per-direction mass of the 16-kernel stack equals the 12-kernel stack's (forward kernels from both metadata.pkl);
  3. new Z_disp bands 0-1 equal the old Z_disp bands 0-1 (same kernels -> same path features);
  4. every other metadata array equals the 3-band input (same observations, cells, adult kernel, disease, ...);
  5. band masses and mean distances (cell-centre distance, stored kernels), sizes.
"""
import json
import os
import pickle

import numpy as np

P = "/work/07980/bwa386/ls6/houfin/processed/grid54"
OLD_IN = f"{P}/model_inputs/numpyro_input_s64_grid54_obsroute"
NEW_IN = f"{P}/model_inputs/numpyro_input_s64_grid54_b44_obsroute"
OLD_PF = f"{P}/latent_avian_paths_s64_grid54"
NEW_PF = f"{P}/latent_avian_paths_s64_grid54_b44"
OLD_SPLITS = [0.0, 155.36162529769288, 482.7446923028151, 1000000000.0]
YEARS = (1902, 1966, 2025)
KERNEL_KEYS = {"juvenile_fft_kernel_stack", "juvenile_edge_correction_stack", "juvenile_kernel_labels", "K",
               "dispersal_spec", "z_disp_gathered_path", "z_gathered_path", "ingest_id", "population_model_spec"}


def meta(d):
    with open(os.path.join(d, "metadata.pkl"), "rb") as fh:
        return pickle.load(fh)


def real_stack(m):
    return np.real(np.fft.ifft2(np.asarray(m["juvenile_fft_kernel_stack"])))


def main():
    import jax.numpy as jnp
    from src.model.build_kernels import make_juvenile_kernel_stack
    from src.model.build_path_features import integrate_paths
    from src.processing.generate_all_path_features import load_land_mask_and_meta
    out = {}
    a, b = meta(OLD_IN), meta(NEW_IN)
    Ny, Nx = int(a["Ny"]), int(a["Nx"])
    land, cell_km, _ = load_land_mask_and_meta(f"{P}/land_mask/ocean_mask_54km.tif")
    disc = a["kernel_discretization"]
    # 1. operator reproduction with the original splits
    ks_old, lab_old = make_juvenile_kernel_stack(2 * Nx - 1, 2 * Ny - 1, cell_km, OLD_SPLITS, mean_dist=330.0,
                                                 shape=0.468, discretization=disc)
    ks_old_stored = real_stack(a)
    out["old_kernels_rebuilt_vs_stored_max_abs"] = float(np.max(np.abs(np.asarray(ks_old) - ks_old_stored)))
    rep = {}
    for y in YEARS:
        z = np.load(f"{OLD_PF}/Z_disp_{y}.npz")
        rebuilt = np.asarray(integrate_paths(jnp.asarray(z["Z_raw"]), ks_old, jnp.asarray(land), steps=20))
        stored = np.asarray(z["Z_disp"])
        d = np.abs(rebuilt - stored)[0][land > 0.5]                                  # (N_land, M, K)
        rep[y] = {"max_abs": float(d.max()), "median_abs": float(np.median(d)),
                  "p999_abs": float(np.quantile(d, 0.999)), "max_abs_value": float(np.abs(stored).max()),
                  "corr": float(np.corrcoef(rebuilt.ravel(), stored.ravel())[0, 1])}
    out["old_splits_rebuild_vs_stored_Zdisp"] = rep
    # 2. new kernels
    ka, kb = ks_old_stored, real_stack(b)
    na, nb = ka.shape[0] // 4, kb.shape[0] // 4
    assert (na, nb) == (3, 4), (na, nb)
    kc = {}
    for dd in range(4):
        A = ka[dd * 3:(dd + 1) * 3]; B = kb[dd * 4:(dd + 1) * 4]
        kc[str(b["juvenile_kernel_labels"][dd * 4]).split("_")[1]] = {
            "mass_old": float(A.sum()), "mass_new": float(B.sum()),
            "bands01_max_abs": float(np.abs(A[:2] - B[:2]).max()),
            "band2_eq_new2plus3_max_abs": float(np.abs(A[2] - B[2] - B[3]).max())}
    out["kernels_per_direction"] = kc
    out["new_kernel_total_mass"] = float(kb.sum())
    out["new_labels"] = [str(x) for x in b["juvenile_kernel_labels"]]
    out["edge_correction_bands01_max_abs"] = float(np.max(np.abs(
        np.asarray(a["juvenile_edge_correction_stack"]).reshape(4, 3, Ny, Nx)[:, :2]
        - np.asarray(b["juvenile_edge_correction_stack"]).reshape(4, 4, Ny, Nx)[:, :2])))
    # 3. new Z_disp bands 0-1 vs old (ingested arrays)
    za = np.memmap(os.path.join(OLD_IN, a["z_disp_gathered_path"]), dtype="float32", mode="r",
                   shape=(a["time"], a["N_land"], a["K"], a["M"]))
    zb = np.memmap(os.path.join(NEW_IN, b["z_disp_gathered_path"]), dtype="float32", mode="r",
                   shape=(b["time"], b["N_land"], b["K"], b["M"]))
    io, inew = [d * 3 + j for d in range(4) for j in (0, 1)], [d * 4 + j for d in range(4) for j in (0, 1)]
    mx = 0.0
    for t in range(0, a["time"], 10):
        mx = max(mx, float(np.abs(np.asarray(za[t][:, io]) - np.asarray(zb[t][:, inew])).max()))
    out["Zdisp_bands01_new_vs_old_max_abs_every10y"] = mx
    zga = np.memmap(os.path.join(OLD_IN, a["z_gathered_path"]), dtype="float32", mode="r", shape=(a["time"], a["N_land"], a["M"]))
    zgb = np.memmap(os.path.join(NEW_IN, b["z_gathered_path"]), dtype="float32", mode="r", shape=(b["time"], b["N_land"], b["M"]))
    out["Z_identical"] = bool(all(np.array_equal(zga[t], zgb[t]) for t in range(a["time"])))
    # 4. other metadata
    diff = []
    for k in sorted(set(a) | set(b)):
        if k in KERNEL_KEYS:
            continue
        va, vb = a.get(k), b.get(k)
        try:
            same = (np.array_equal(np.asarray(va), np.asarray(vb)) if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray)
                    else va == vb)
        except Exception:
            same = False
        if not same:
            diff.append(k)
    out["metadata_keys_differing_beyond_kernels"] = diff
    out["dispersal_spec_new"] = b.get("dispersal_spec")
    # 5. band masses / mean distances
    Ly, Lx = kb.shape[1:]
    dy = np.minimum(np.arange(Ly), Ly - np.arange(Ly)); dx = np.minimum(np.arange(Lx), Lx - np.arange(Lx))
    r = cell_km * np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
    bands = {}
    for tag, k, n in (("old", ka, 3), ("new", kb, 4)):
        bands[tag] = []
        for j in range(n):
            kk = k[[d * n + j for d in range(4)]].sum(0)
            bands[tag].append({"mass": float(kk.sum()), "mean_km": float((kk * r).sum() / kk.sum())})
    out["bands"] = bands
    out["sizes_GB"] = {"old_Zdisp": os.path.getsize(os.path.join(OLD_IN, a["z_disp_gathered_path"])) / 1e9,
                       "new_Zdisp": os.path.getsize(os.path.join(NEW_IN, b["z_disp_gathered_path"])) / 1e9,
                       "Z": os.path.getsize(os.path.join(NEW_IN, b["z_gathered_path"])) / 1e9}
    with open(f"{P}/grid54_b44_verify.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

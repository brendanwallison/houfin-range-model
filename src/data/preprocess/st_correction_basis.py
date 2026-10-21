"""Spatial basis for the age model's smooth spatiotemporal habitat correction (population_model.st_correction,
E067), with its PATH-FEATURE image.

The correction is h(cell, t) = sum_b S_b(cell) w_b(t): a low-frequency, land-centred cosine basis S (the same family
as the disease term's, model_inputs.generate_spatial_basis) times smooth time coefficients, added to every habitat
field H. Journey survival reads PATH habitat Z_disp = A Z, where A is the land-normalised path-integration operator of
src/processing/generate_all_path_features.py. A is linear in its input (the normaliser depends only on the land
mask), so the path image of the correction is sum_b (A S_b)(cell, k) w_b(t): A applied ONCE to each basis function,
here, through the identical operator (same juvenile kernel stack, mask, splits and integration steps). The weights
stay learnable; nothing about Z changes.

Output ``st_correction_basis.npz`` next to the model inputs:
    S       (n_basis, N_land)        land-centred spatial basis
    S_disp  (N_land, K, n_basis)     its path image, kernel order as Z_disp
    n_freq, labels

    AGE_MODEL_CONFIG=<overlay> python -m src.data.preprocess.st_correction_basis --n-freq 3
"""
import argparse
import os
import pickle

import jax.numpy as jnp
import numpy as np


def build(input_dir, n_freq, out_path):
    from src.config_utils import load_age_model_config
    from src.data.combine.model_inputs import generate_spatial_basis
    from src.model.build_kernels import dispersal_spec, make_juvenile_kernel_stack
    from src.model.build_path_features import integrate_paths
    from src.processing.generate_all_path_features import load_land_mask_and_meta

    with open(os.path.join(input_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    rows, cols = np.asarray(meta["land_rows"]), np.asarray(meta["land_cols"])
    Ny, Nx = int(meta["Ny"]), int(meta["Nx"])
    S = generate_spatial_basis(Ny, Nx, rows, cols, n_freq, label=" (st_correction)")      # (nb, N_land)

    cfg = load_age_model_config()
    disp = dispersal_spec(cfg)
    land_mask_np, cell_size_km, _ = load_land_mask_and_meta(cfg["ocean_mask"])
    if land_mask_np.shape != (Ny, Nx):
        raise ValueError(f"mask grid {land_mask_np.shape} != model grid {(Ny, Nx)}")
    Ly, Lx = 2 * Ny - 1, 2 * Nx - 1
    kernel_stack, labels = make_juvenile_kernel_stack(
        Lx, Ly, cell_size_km, disp["juvenile_radial_splits_km"],
        mean_dist=disp["juvenile_mdd_km"], shape=disp["juvenile_shape"])
    # Guard: the operator here must be the one that built Z_disp. Rebuild the path features of the first 4 DESK
    # features in one year from Z and compare with the stored Z_disp.
    t0 = 0
    zmem = np.memmap(os.path.join(input_dir, meta["z_gathered_path"]), dtype="float32", mode="r",
                     shape=(meta["time"], meta["N_land"], meta["M"]))
    zdmem = np.memmap(os.path.join(input_dir, meta["z_disp_gathered_path"]), dtype="float32", mode="r",
                      shape=(meta["time"], meta["N_land"], meta["K"], meta["M"]))
    zg = np.zeros((1, Ny, Nx, 4), dtype=np.float32)
    zg[0, rows, cols, :] = np.asarray(zmem[t0, :, :4])
    rebuilt = np.asarray(integrate_paths(jnp.asarray(zg), kernel_stack, jnp.asarray(land_mask_np),
                                         steps=int(disp["path_integration_steps"])))[0][rows, cols]
    # integrate_paths returns (..., M, K); model inputs store Z_disp as (N_land, K, M) (model_inputs transposes)
    rebuilt = np.transpose(rebuilt, (0, 2, 1))
    stored = np.asarray(zdmem[t0, :, :, :4])
    # The guard must catch a WRONG operator (transposed kernels, other splits/steps, another Z), not float32 FFT noise
    # in low-land-support coastal/island cells, which Z_disp itself carries (diagnostic 2026-10-20: corr .99999, median
    # 0; 1% of cells differ by a few % of the global max). A wrong operator destroys the correlation and the median.
    a, b = rebuilt.ravel(), stored.ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    med = float(np.median(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))
    print(f"[st_correction_basis] operator check vs stored Z_disp: corr {corr:.6f}, median rel err {med:.2e}")
    if corr < 0.999 or med > 1e-5:
        raise ValueError(f"path operator does not reproduce the stored Z_disp (corr {corr:.6f}, median {med:.2e})")
    grid = np.zeros((1, Ny, Nx, S.shape[0]), dtype=np.float32)
    grid[0, rows, cols, :] = S.T
    S_disp_grid = integrate_paths(jnp.asarray(grid), kernel_stack, jnp.asarray(land_mask_np),
                                  steps=int(disp["path_integration_steps"]))          # (1, Ny, Nx, K, nb)
    S_disp = np.transpose(np.asarray(S_disp_grid)[0][rows, cols], (0, 2, 1))          # (N_land, nb, K) -> (N_land, K, nb)
    np.savez_compressed(out_path, S=S.astype(np.float32), S_disp=S_disp.astype(np.float32),
                        n_freq=int(n_freq), labels=np.asarray(labels))
    print(f"[st_correction_basis] {out_path}: S {S.shape}, S_disp {S_disp.shape}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-freq", type=int, default=3)
    a = ap.parse_args()
    from src.config_utils import load_age_model_config
    input_dir = load_age_model_config()["input_dir"]
    build(input_dir, a.n_freq, os.path.join(input_dir, "st_correction_basis.npz"))


if __name__ == "__main__":
    main()

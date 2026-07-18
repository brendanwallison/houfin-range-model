"""ESK: build the habitat-similarity ground truth Z from eBird via Ruzicka kernel-PCA.

The first stage of the community encoder. For the one richly-sampled eBird year,
it treats each land cell's weekly per-species abundance vector as a point,
computes a Ruzicka (generalized-Jaccard) similarity kernel between cells, and
takes its kernel-PCA -- a Nystrom landmark approximation makes the full pairwise
kernel tractable. The result, swept over temporal-smoothing bandwidths and latent
dimensions, is ``Z.npy`` + ``valid_mask.npy``: the "real" habitat-similarity
space that DESK (``desk_training``) later learns to predict from covariates alone.

Abundance rasters are aggregated to the model grid by reprojection as they load
(:func:`load_tifs_structured`), so Z is built directly at the model resolution.
"""
import json
import os
import glob
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from scipy.ndimage import gaussian_filter1d

from .config_utils import load_config
from src.config_utils import load_data_config
from src.processing import regrid


def load_tifs_structured(folder, pattern="*_abundance_median_*.tif", target_res_m=None):
    """
    Parses filenames to enforce strict (Species, Time) ordering.
    Returns: stack (H, W, S*T), meta dict.

    If ``target_res_m`` is given, each raster is reprojected onto the model
    reference grid (``grid.ref_raster``) as it loads, via ``reproject_match``
    with ``average`` -- the linear areal aggregate. Relative abundance
    aggregates linearly, so this is the correct order: aggregate abundance
    first, then build the (nonlinear) Ruzicka kernel + PCA at the target
    resolution -- giving Z directly at the model grid rather than mean-pooling
    the finished embedding downstream (meaningless for a kernel-PCA latent).

    Reproject (not integer ``block_reduce``) is used so this works for any
    native:target ratio -- eBird's ~2.96 km cells do not divide the 25 km grid
    evenly -- and so the eBird CRS (EPSG:8857) is resolved onto the Albers grid
    in the same step, going straight from finest native resolution to the model
    grid without an intermediate fixed-resolution reprojection.
    """
    import rioxarray  # noqa: F401  (registers .rio)
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise ValueError(f"No files found in {folder} matching {pattern}")

    regex = re.compile(r"([a-z0-9]+)_abundance_median_(\d{4}-\d{2}-\d{2})")
    records = []
    for fpath in files:
        fname = os.path.basename(fpath)
        match = regex.match(fname)
        if match:
            records.append({
                "species": match.group(1),
                "date": datetime.strptime(match.group(2), "%Y-%m-%d"),
                "path": fpath,
            })
        else:
            print(f"Skipping non-matching file: {fname}")

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No filenames matched regex.")

    n_species = df["species"].nunique()
    n_weeks = df["date"].nunique()
    if len(df) != n_species * n_weeks:
        raise ValueError(f"Grid incomplete. Expected {n_species * n_weeks} files, found {len(df)}.")

    df_sorted = df.sort_values(by=["species", "date"])
    ordered_paths = df_sorted["path"].tolist()

    with rasterio.open(ordered_paths[0]) as src:
        H, W = src.shape
        native_res = abs(src.transform.a)

    ref = None
    if target_res_m is not None:
        ref = regrid.load_ref()
        H, W = int(ref.rio.height), int(ref.rio.width)
        print(f"Reprojecting {len(ordered_paths)} rasters {native_res:.0f}m -> "
              f"model grid {H}x{W} @ {target_res_m}m as they load "
              f"({n_species} sp x {n_weeks} wks)...")
    else:
        print(f"Loading {len(ordered_paths)} rasters at native {native_res:.0f}m "
              f"({n_species} sp x {n_weeks} wks)...")

    full_stack = np.zeros((H, W, len(ordered_paths)), dtype=np.float32)

    for i, p in enumerate(ordered_paths):
        if ref is not None:
            da = rioxarray.open_rasterio(p, masked=True).squeeze()
            band = regrid.reproject_to_ref(da, ref, resampling="average").values
        else:
            with rasterio.open(p) as src:
                band = src.read(1).astype(np.float64)
                if src.nodata is not None:
                    band[band == src.nodata] = np.nan
        full_stack[:, :, i] = band.astype(np.float32)

    # Species block order = sorted unique species (matches df_sorted column blocks),
    # so downstream code (e.g. the BBS amplitude modulation) can align per-species
    # scalars to the right 52-week block.
    species = sorted(df["species"].unique().tolist())
    return full_stack, {"n_species": n_species, "n_weeks": n_weeks,
                        "native_res_m": native_res, "target_res_m": target_res_m,
                        "species": species}


def smooth_abundances(ebird_flat, n_weeks, sigma):
    """
    1. Reshapes to (N, Species, Weeks)
    2. Applies Gaussian blur along Time axis (if sigma > 0)
    3. Returns flattened array (N, S*T) preserving absolute abundance.
    """
    N, D = ebird_flat.shape
    n_species = D // n_weeks

    data_3d = ebird_flat.reshape(N, n_species, n_weeks)

    if sigma > 1e-5:
        data_smoothed = gaussian_filter1d(data_3d, sigma=sigma, axis=-1, mode="wrap")
    else:
        data_smoothed = data_3d

    data_smoothed = np.maximum(data_smoothed, 0.0)
    return data_smoothed.reshape(N, -1)


def compute_optimal_latent_z_ruzicka(ebird_flat, n_species, n_weeks, latent_dim, n_landmarks=10000, device="cuda"):
    """
    Computes Mercer features using the GLOBAL Ruzicka Kernel (Generalized Jaccard).
    """
    N, D = ebird_flat.shape
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    idx_lm = np.random.choice(N, min(N, n_landmarks), replace=False)
    X_lm_np = ebird_flat[idx_lm]
    M = X_lm_np.shape[0]

    print(f"Computing Exact Global Kernel on {M} landmarks (Dim={D})...")

    try:
        T_lm = torch.tensor(X_lm_np, device=device, dtype=torch.float32)
    except RuntimeError:
        print("VRAM limit warning: Landmarks too large. Reduce n_landmarks.")
        return None

    sum_lm = T_lm.sum(dim=1, keepdim=True)
    l1_dist = torch.cdist(T_lm, T_lm, p=1)

    sum_plus_sum = sum_lm + sum_lm.T
    numerator = 0.5 * (sum_plus_sum - l1_dist)
    denominator = 0.5 * (sum_plus_sum + l1_dist)

    mask = denominator > 1e-6
    K_mm_total = torch.zeros_like(numerator)
    K_mm_total[mask] = numerator[mask] / denominator[mask]

    L, U = torch.linalg.eigh(K_mm_total.cpu())

    idx_sort = torch.argsort(L, descending=True)[:latent_dim]
    L = L[idx_sort]
    U = U[:, idx_sort].to(device)

    L = torch.clamp(L, min=1e-10).to(device)
    proj_mat = U * torch.rsqrt(L)

    Z_opt = np.zeros((N, latent_dim), dtype=np.float32)
    batch_size = 5000

    print(f"Projecting {N} points in batches...")

    X_all_np = ebird_flat

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_np = X_all_np[start:end]
            T_batch = torch.tensor(batch_np, device=device, dtype=torch.float32)

            sum_b = T_batch.sum(dim=1, keepdim=True)
            l1_rect = torch.cdist(T_batch, T_lm, p=1)
            sum_plus_sum_rect = sum_b + sum_lm.T

            num = 0.5 * (sum_plus_sum_rect - l1_rect)
            den = 0.5 * (sum_plus_sum_rect + l1_rect)

            mask = den > 1e-6
            K_batch_lm = torch.zeros_like(num)
            K_batch_lm[mask] = num[mask] / den[mask]

            z_batch = K_batch_lm @ proj_mat
            Z_opt[start:end] = z_batch.cpu().numpy()

    return Z_opt


def compute_kernel_diagnostics_ruzicka(z, ebird_flat, n_species, n_weeks, max_samples=500):
    """
    Computes RMSE between the Nyström approximation (ZZ^T) and the
    Exact GLOBAL Ruzicka Kernel (K_true).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = z.shape[0]
    idx = np.random.choice(N, min(N, max_samples), replace=False)

    z_s = torch.tensor(z[idx], device=device, dtype=torch.float32)
    X_s = torch.tensor(ebird_flat[idx], device=device, dtype=torch.float32)

    K_approx = z_s @ z_s.T

    sum_s = X_s.sum(dim=1, keepdim=True)
    l1_dist = torch.cdist(X_s, X_s, p=1)

    sum_plus_sum = sum_s + sum_s.T
    num = 0.5 * (sum_plus_sum - l1_dist)
    den = 0.5 * (sum_plus_sum + l1_dist)

    mask = den > 1e-6
    K_true = torch.zeros_like(num)
    K_true[mask] = num[mask] / den[mask]

    diff = K_approx - K_true
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    k_scale = torch.sqrt(torch.mean(K_true**2)).item()

    svals = torch.linalg.svd(z_s, full_matrices=False)[1]
    svals_sq = svals**2
    eff_rank_Z = (svals_sq.sum()**2) / torch.sum(svals_sq**2)

    return {
        "rmse": rmse,
        "rmse_norm": rmse / (k_scale + 1e-8),
        "effective_rank": eff_rank_Z.item(),
    }


def visualize_nystrom_component(Z_k, cmap="viridis", fade_continuous=True, iqr_factor=0.5, title=None, show_colorbar=True):
    """Visualize a single Nyström eigenfeature."""
    valid = ~np.isnan(Z_k)
    vals = Z_k[valid]

    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 1, len(vals))

    pct_map = np.full_like(Z_k, np.nan, dtype=float)
    pct_map[valid] = ranks

    abs_vals = np.abs(vals)
    med = np.median(abs_vals)
    iqr = np.percentile(abs_vals, 75) - np.percentile(abs_vals, 25)
    threshold = med + iqr_factor * iqr
    support_mask = np.full_like(Z_k, False, dtype=bool)
    support_mask[valid] = abs_vals >= threshold

    alpha = np.ones_like(Z_k, dtype=float)
    if fade_continuous:
        sorted_idx = np.argsort(abs_vals)
        abs_ranks = np.empty_like(sorted_idx, dtype=float)
        abs_ranks[sorted_idx] = np.linspace(0, 1, len(abs_vals))
        alpha[valid] = abs_ranks
    else:
        alpha[valid] = support_mask[valid].astype(float)

    plt.figure(figsize=(5, 4))
    plt.imshow(pct_map, cmap=cmap, alpha=alpha)
    if show_colorbar:
        plt.colorbar(label="Signed percentile along component")
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    return pct_map, support_mask


def run_esk_experiment(config=None):
    """Build the ESK ground-truth Z: Ruzicka kernel-PCA over eBird abundance.

    Loads the weekly per-species eBird rasters (reprojected to the model grid),
    computes the Nystrom-approximated Ruzicka-similarity kernel-PCA latent over a
    sweep of temporal-smoothing bandwidths and latent dimensions, and writes
    ``Z.npy`` + ``valid_mask.npy`` (plus diagnostics). ``config`` is the encoder
    config (dict/path; defaults to the repo config).
    """
    if config is None:
        config = load_config()
    elif isinstance(config, (str, os.PathLike)):
        config = load_config(config)

    paths = config["paths"]
    esk_cfg = config["esk"]

    # Build Z at the model grid: aggregate abundance to grid.target_res_m before
    # the Ruzicka kernel/PCA (C3 — no downstream pooling of the embedding). Use the
    # shared cache (reproject once; NaN preserved) rather than re-reprojecting here.
    from src.community_encoder.train_DESK.ebird_cache import load_ebird_stack
    ebird_stack, meta = load_ebird_stack(config)

    H, W, D = ebird_stack.shape
    valid_mask = np.any(~np.isnan(ebird_stack), axis=-1)
    valid_flat = valid_mask.flatten()
    ebird_flat_raw = np.nan_to_num(ebird_stack).reshape(-1, D)[valid_flat]

    out_dir = paths["esk_output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing {len(ebird_flat_raw)} valid pixels. Shape: {ebird_flat_raw.shape}")

    sigmas = esk_cfg.get("sigmas", [0.0, 0.5, 1.0, 1.5])
    latent_dims = esk_cfg.get("latent_dims", [8, 16, 32])
    n_landmarks = esk_cfg.get("n_landmarks", 30000)

    total_runs = len(sigmas) * len(latent_dims)
    run_count = 0
    results = []

    print(f"\n{'Run':<4} | {'Dim':<4} | {'Sig':<4} | {'Eff Rank':<10} | {'RMSE (N)':<10} | {'RMSE (U)':<10}")
    print("-" * 55)

    for sig in sigmas:
        X_smooth = smooth_abundances(ebird_flat_raw, meta["n_weeks"], sigma=sig)
        max_dim = max(latent_dims)

        try:
            Z_max = compute_optimal_latent_z_ruzicka(
                X_smooth,
                meta["n_species"],
                meta["n_weeks"],
                max_dim,
                n_landmarks=n_landmarks,
            )

            for dim in sorted(latent_dims):
                run_count += 1
                Z_slice = Z_max[:, :dim]
                diag = compute_kernel_diagnostics_ruzicka(
                    Z_slice,
                    X_smooth,
                    meta["n_species"],
                    meta["n_weeks"],
                )

                print(f"{run_count}/{total_runs:<3} | {dim:<4} | {sig:<4.1f} | {diag['effective_rank']:<10.2f} | {diag['rmse_norm']:<10.4f} | {diag['rmse']:<10.4f}")
                results.append({
                    "dim": dim,
                    "sigma": sig,
                    "rank": diag["effective_rank"],
                    "rmse_norm": diag["rmse_norm"],
                    "rmse_unnorm": diag["rmse"],
                })

                if dim == max_dim:
                    sigma_dir = os.path.join(out_dir, f"sigma_{sig}")
                    os.makedirs(sigma_dir, exist_ok=True)

                    np.save(os.path.join(sigma_dir, "Z.npy"), Z_slice)
                    np.save(os.path.join(sigma_dir, "valid_mask.npy"), valid_mask)

                    meta_out = {
                        "sigma": sig,
                        "latent_dim": dim,
                        "n_species": meta["n_species"],
                        "n_weeks": meta["n_weeks"],
                        "kernel": "global_ruzicka",
                    }
                    with open(os.path.join(sigma_dir, "meta.json"), "w", encoding="utf-8") as f:
                        json.dump(meta_out, f, indent=2)

                    Z_full = np.full((H * W, dim), np.nan, dtype=np.float32)
                    Z_full[valid_flat] = Z_slice

                    base_name = f"d{dim}_s{sig}_ruzicka"
                    for k in range(min(dim, 10)):
                        latent_map = Z_full[:, k].reshape(H, W)
                        base_title = f"Z{k + 1} (Sig={sig})\nR:{diag['effective_rank']:.1f}"
                        out_path = os.path.join(out_dir, f"map_{base_name}_c{k + 1}.png")

                        visualize_nystrom_component(
                            latent_map,
                            cmap="viridis",
                            fade_continuous=True,
                            iqr_factor=0.5,
                            title=base_title,
                            show_colorbar=True,
                        )
                        plt.savefig(out_path, dpi=100)
                        plt.close()

        except Exception as exc:
            print(f"Error for sigma {sig}: {exc}")
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "sweep_summary.csv"), index=False)

    if not df.empty:
        plt.figure(figsize=(10, 6))
        for sig in sigmas:
            subset = df[df["sigma"] == sig]
            plt.plot(subset["dim"], subset["rank"], marker="o", label=f"Sigma={sig}")

        plt.xlabel("Latent Dimension")
        plt.ylabel("Effective Rank")
        plt.title("Global Ruzicka Kernel: Rank vs Dimension")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "summary_plot.png"))
        plt.close()
        print("\nSummary plot saved.")


if __name__ == "__main__":
    run_esk_experiment()

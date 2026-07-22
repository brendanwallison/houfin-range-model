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
from .ebird_io import load_tifs_structured  # noqa: F401  torch-free loader (re-export)
from src.config_utils import load_data_config
from src.processing import regrid


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


def stratified_landmarks(strata, n_landmarks, rng, recent_label=0, recent_frac=0.5):
    """Recent-heavy landmark indices across strata (for the joint spatiotemporal ESK).

    ``strata`` (N,) integer labels (e.g. 0 = recent eBird, 1..k = historical decade bins).
    The ``recent_label`` stratum receives up to ``recent_frac`` of the landmark budget (so
    the modern spatial structure stays dominant in the basis despite recent points being a
    small share of N); the remainder is split across the other strata proportional to their
    counts. If ``n_landmarks >= N`` all points are landmarks (exact). Returns a shuffled
    index array; ordering is irrelevant to the kernel-PCA.
    """
    strata = np.asarray(strata)
    N = len(strata)
    if n_landmarks >= N:
        return rng.permutation(N)
    idx = np.arange(N)
    rec = idx[strata == recent_label]
    n_rec = min(len(rec), int(round(recent_frac * n_landmarks)))
    picks = [rng.choice(rec, n_rec, replace=False)] if n_rec else []
    others = [l for l in np.unique(strata) if l != recent_label]
    counts = np.array([int((strata == l).sum()) for l in others], dtype=float)
    n_oth = n_landmarks - n_rec
    if counts.sum() > 0 and n_oth > 0:
        alloc = np.floor(n_oth * counts / counts.sum()).astype(int)
        while alloc.sum() < n_oth:                       # largest-remainder fill
            alloc[int(np.argmax(n_oth * counts / counts.sum() - alloc))] += 1
        for l, a in zip(others, alloc):
            li = idx[strata == l]
            picks.append(rng.choice(li, min(int(a), len(li)), replace=False))
    return rng.permutation(np.concatenate(picks)) if picks else rng.permutation(N)


def compute_optimal_latent_z_ruzicka(ebird_flat, n_species, n_weeks, latent_dim, n_landmarks=10000,
                                     device="cuda", seed=0, return_proj=False, landmark_idx=None):
    """
    Computes Mercer features using the GLOBAL Ruzicka Kernel (Generalized Jaccard).

    ``seed`` makes the landmark draw reproducible (at 25 km N < n_landmarks so ALL pixels
    are landmarks anyway -- exact). ``landmark_idx`` (if given) uses a caller-chosen landmark
    set (e.g. ``stratified_landmarks`` for the joint spatiotemporal ESK) instead of a random
    draw. ``return_proj`` additionally returns the landmark rows and the projection matrix so
    the SAME basis can later project out-of-sample vectors via ``project_into_z`` -- the
    pinned basis that makes z_DESK and z_obs directly comparable.
    """
    N, D = ebird_flat.shape
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    rng = np.random.default_rng(seed)
    if landmark_idx is not None:
        idx_lm = np.asarray(landmark_idx, dtype=int)
    else:
        idx_lm = rng.choice(N, min(N, n_landmarks), replace=False)
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

    # Eigendecompose on the GPU (cuSOLVER) when it fits -- far faster than CPU LAPACK
    # and identical precision (K_mm is float32 either way). Fall back to CPU only if
    # the eigenvector workspace OOMs (e.g. a small card at finer resolution / larger N);
    # at 25 km (N ~16.5k) the matrix is ~1 GB and fits an A100 with room to spare.
    try:
        L, U = torch.linalg.eigh(K_mm_total)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        if device == "cuda":
            torch.cuda.empty_cache()
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

    if return_proj:
        return Z_opt, X_lm_np.astype(np.float32), proj_mat.detach().cpu().numpy().astype(np.float32)
    return Z_opt


def project_into_z(x_flat, landmarks, proj_mat, device="cuda", batch_size=5000):
    """Project rows of ``x_flat`` into a SAVED ESK basis: ``z = Ruzicka(x, landmarks) @ proj_mat``.

    ``landmarks`` (M, D) and ``proj_mat`` (M, latent) come from a prior
    ``compute_optimal_latent_z_ruzicka(..., return_proj=True)``. Mirrors that function's own
    in-sample projection exactly, so out-of-sample vectors (e.g. observed BBS-amplitude
    communities) land in the SAME pinned basis as z_DESK -- the whole point of comparing in
    z-space. Projecting the original training rows reproduces their Z (validity check).
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    T_lm = torch.tensor(np.asarray(landmarks), device=device, dtype=torch.float32)
    P = torch.tensor(np.asarray(proj_mat), device=device, dtype=torch.float32)
    sum_lm = T_lm.sum(dim=1, keepdim=True)
    N, L = x_flat.shape[0], P.shape[1]
    Z = np.zeros((N, L), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            Tb = torch.tensor(np.asarray(x_flat[s:e]), device=device, dtype=torch.float32)
            l1 = torch.cdist(Tb, T_lm, p=1)
            sp = Tb.sum(1, keepdim=True) + sum_lm.T
            num = 0.5 * (sp - l1); den = 0.5 * (sp + l1)
            K = torch.zeros_like(num); m = den > 1e-6; K[m] = num[m] / den[m]
            Z[s:e] = (K @ P).cpu().numpy()
    return Z


def project_amplitude_to_z(X, z_dir, latent_dim, batch=20000):
    """Project amplitude community vectors ``X`` into the SAVED ESK basis in ``z_dir``.

    Loads ``esk_landmarks.npy``/``esk_projmat.npy``/``meta.json`` (written by
    ``run_esk_experiment``), applies the SAME weekly smoothing the ESK used, and projects
    batched -> ``(N, latent_dim)``. Single source of truth for z_obs, shared by the enrich
    trainer (supervised targets) and validate (reconstruction eval), so targets and eval
    are guaranteed to live in the identical pinned basis. Returns None if no projection saved.
    """
    import json as _json
    lmp, pmp = os.path.join(z_dir, "esk_landmarks.npy"), os.path.join(z_dir, "esk_projmat.npy")
    if not (os.path.exists(lmp) and os.path.exists(pmp)):
        return None
    landmarks, projmat = np.load(lmp), np.load(pmp)
    meta = _json.load(open(os.path.join(z_dir, "meta.json")))
    sigma, n_weeks = float(meta.get("sigma", 0.0)), int(meta["n_weeks"])
    N = X.shape[0]
    z = np.zeros((N, latent_dim), dtype="float32")
    for s in range(0, N, batch):
        e = min(s + batch, N)
        xb = smooth_abundances(X[s:e], n_weeks, sigma) if sigma > 0 else X[s:e]
        z[s:e] = project_into_z(xb, landmarks, projmat)[:, :latent_dim]
    return z


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
            Z_max, esk_landmarks, esk_projmat = compute_optimal_latent_z_ruzicka(
                X_smooth,
                meta["n_species"],
                meta["n_weeks"],
                max_dim,
                n_landmarks=n_landmarks,
                seed=esk_cfg.get("seed", 0),
                return_proj=True,
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
                    # Save the projection so observed communities can be mapped into THIS
                    # exact basis later (z-space reconstruction metric). Landmarks are the
                    # sigma-smoothed rows; project new x with the same smoothing.
                    np.save(os.path.join(sigma_dir, "esk_landmarks.npy"), esk_landmarks)
                    np.save(os.path.join(sigma_dir, "esk_projmat.npy"), esk_projmat)

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


def run_spacetime_esk(config=None):
    """Joint spatiotemporal ESK (``bbs_mode=enrich``): Ruzicka kernel-PCA over the eBird-recent
    + BBS-historical amplitude points, with **recent-heavy stratified landmarks**, so the basis
    spans historical CHANGE directions while the modern eBird spatial structure stays dominant.

    Writes the SAME layout the eBird ESK writes -- ``Z.npy`` (recent 2023 embedding on the grid,
    ``Z[valid_mask]`` order), ``valid_mask.npy``, ``esk_landmarks``/``esk_projmat`` (the joint
    projection), ``meta.json`` -- into ``esk/spacetime``, so DESK / validate / cube consume it
    unchanged, just pointed here. z_obs for any point is then ``project_amplitude_to_z`` through
    this joint projection (= its joint-ESK embedding).
    """
    if config is None:
        config = load_config()
    elif isinstance(config, (str, os.PathLike)):
        config = load_config(config)
    bc, esk_cfg, paths = config["bbs"], config["esk"], config["paths"]
    sc = esk_cfg.get("spacetime", {})
    sigma = float(sc.get("sigma", 1.0)); latent_dim = int(sc.get("latent_dim", 32))
    landmark_mode = str(sc.get("landmark_mode", "random"))
    n_landmarks = int(bc.get("n_landmarks", 30000)); seed = int(esk_cfg.get("seed", 0))

    zt = bc["z_dir"]
    X = np.nan_to_num(np.load(os.path.join(zt, "X_points.npy"))).astype("float32")
    pidx = np.load(os.path.join(zt, "point_index.npy"))
    with open(os.path.join(zt, "points_meta.json")) as fh:
        pmeta = json.load(fh)
    n_species, n_weeks = int(pmeta["n_species"]), int(pmeta["n_weeks"])
    recent_year = int(pmeta["recent_year"])
    print(f"[st-esk] joint points {X.shape}: {pmeta['n_recent']} recent + {pmeta['n_hist']} historical")

    X = smooth_abundances(X, n_weeks, sigma) if sigma > 0 else X   # match the eBird-ESK weekly smoothing
    yrs = pidx[:, 2]
    strata = np.where(yrs == recent_year, 0, ((yrs // 10) * 10).astype(int))   # 0=recent, else decade
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if landmark_mode == "stratified":
        # proportional-by-stratum: recent gets ONLY its natural share (no upweighting)
        lm_idx = stratified_landmarks(strata, n_landmarks, rng, recent_label=0,
                                      recent_frac=float((strata == 0).mean()))
    else:                                                          # 'random' (default): uniform over all points
        lm_idx = rng.permutation(N)[:min(N, n_landmarks)]
    print(f"[st-esk] {len(lm_idx)} landmarks ({landmark_mode}); recent share {np.mean(strata[lm_idx] == 0):.2f} "
          f"(population {np.mean(strata == 0):.2f})")

    Zj, lm, pm = compute_optimal_latent_z_ruzicka(
        X, n_species, n_weeks, latent_dim, n_landmarks=n_landmarks, seed=seed,
        return_proj=True, landmark_idx=lm_idx)
    diag = compute_kernel_diagnostics_ruzicka(Zj, X, n_species, n_weeks)
    print(f"[st-esk] joint kernel: effective_rank={diag['effective_rank']:.1f} "
          f"rmse_norm={diag['rmse_norm']:.4f} (vs eBird-only ESK diagnostics for comparison)")

    # Recent (2023) embedding onto the grid, in the eBird-ESK Z.npy layout (Z[valid_mask]).
    import rasterio
    from src.config_utils import load_data_config
    with rasterio.open(load_data_config()["grid"]["ref_raster"]) as src:
        H, W = src.height, src.width
    rec = yrs == recent_year
    rr, cc = pidx[rec, 0], pidx[rec, 1]
    vm = np.zeros((H, W), bool); vm[rr, cc] = True
    Zg = np.full((H, W, latent_dim), np.nan, np.float32); Zg[rr, cc] = Zj[rec]
    Z_flat = Zg[vm]

    out_dir = os.path.join(paths["esk_output_dir"], "spacetime")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "Z.npy"), Z_flat)
    np.save(os.path.join(out_dir, "valid_mask.npy"), vm)
    np.save(os.path.join(out_dir, "esk_landmarks.npy"), lm)
    np.save(os.path.join(out_dir, "esk_projmat.npy"), pm)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"sigma": sigma, "latent_dim": latent_dim, "n_species": n_species,
                   "n_weeks": n_weeks, "kernel": "joint_ruzicka_stratified",
                   "recent_frac": recent_frac, "recent_year": recent_year,
                   "n_landmarks": int(len(lm_idx))}, fh, indent=2)
    print(f"[st-esk] saved joint basis -> {out_dir} (recent Z {Z_flat.shape}, latent_dim {latent_dim})")
    return out_dir


if __name__ == "__main__":
    run_esk_experiment()

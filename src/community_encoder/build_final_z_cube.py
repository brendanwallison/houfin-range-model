"""Apply the trained DESK model to every year to build the Z spacetime cube.

Runs the fitted DESK autoencoder's encoder over each year's smoothed covariate
state (climate/land-use/soil), producing ``Z_latent_{year}.npy`` for the whole
timeline -- the habitat-quality cube the population model consumes. Missing or
edge cells are filled in three passes: spatial interpolation within a radius,
backfill from the static ESK ground-truth where available, then nearest-neighbor
cleanup (``fill_gaps_stage1/2/3``). CRS/mask/normalization anchor come from the
data + encoder configs.
"""
import glob
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from torch import nn
from tqdm import tqdm

from community_encoder.train_DESK.config_utils import load_config
from community_encoder.train_DESK import covariate_io as cio
from community_encoder.train_DESK.model_arch import MultiStreamAutoencoder


def fill_gaps_stage1_spatial(z_cube, valid_mask, land_mask, radius_px=25):
    """Fill land gaps within a local radius using linear interpolation."""
    dist_map = distance_transform_edt(~valid_mask)
    target_mask = land_mask & (~valid_mask) & (dist_map <= radius_px)

    if target_mask.sum() == 0:
        return z_cube, valid_mask

    print(f"   -> Stage 1 (Spatial): Interpolating {target_mask.sum()} pixels within {radius_px}px...")

    y_valid, x_valid = np.where(valid_mask)
    points = np.column_stack((y_valid, x_valid))
    y_target, x_target = np.where(target_mask)

    z_filled = z_cube.copy()
    for d in range(z_cube.shape[2]):
        values = z_cube[y_valid, x_valid, d]
        interp_vals = griddata(points, values, (y_target, x_target), method="linear")
        z_filled[y_target, x_target, d] = interp_vals

    new_valid = ~np.isnan(z_filled).any(axis=-1)
    return z_filled, new_valid


def fill_gaps_stage2_static(z_cube, valid_mask, land_mask, z_static_ref, z_static_mask):
    """Backfill remaining land gaps with the static reference latent field."""
    target_mask = land_mask & (~valid_mask) & z_static_mask

    if target_mask.sum() == 0:
        return z_cube, valid_mask

    print(f"   -> Stage 2 (Static): Backfilling {target_mask.sum()} pixels with reference Z...")

    z_filled = z_cube.copy()
    z_filled[target_mask] = z_static_ref[target_mask]
    new_valid = valid_mask | target_mask
    return z_filled, new_valid


def fill_gaps_stage3_nearest(z_cube, valid_mask, land_mask):
    """Fill any remaining land gaps with the nearest available value."""
    target_mask = land_mask & (~valid_mask)

    if target_mask.sum() == 0:
        return z_cube

    print(f"   -> Stage 3 (Cleanup): NN filling remaining {target_mask.sum()} pixels...")

    y_valid, x_valid = np.where(valid_mask)
    points = np.column_stack((y_valid, x_valid))
    y_target, x_target = np.where(target_mask)

    z_filled = z_cube.copy()
    for d in range(z_cube.shape[2]):
        values = z_cube[y_valid, x_valid, d]
        interp_vals = griddata(points, values, (y_target, x_target), method="nearest")
        z_filled[y_target, x_target, d] = interp_vals

    return z_filled


def build_spacetime_cube(config: Optional[Union[Dict[str, Any], str, os.PathLike]] = None):
    """Encode every year's covariate state with the trained DESK model into Z.

    Loads the fitted DESK network + per-stream normalization stats, encodes each
    year's ``state_{year}.npz`` to its latent Z on the model grid, runs the
    three-stage gap fill, and writes ``Z_latent_{year}.npy`` (plus the valid
    mask). ``config`` is the encoder config (dict or path); defaults to the repo
    config.
    """
    if config is None:
        config = load_config()
    elif isinstance(config, (str, os.PathLike)):
        config = load_config(config)

    paths = config.get("paths", {})
    cube_cfg = config.get("latent_cube", {})
    desk_cfg = config.get("desk", {})

    # Gap-fill radius in km -> pixels at the model grid, so the fill footprint
    # is resolution-independent (was a hardcoded 25 px = 100 km at 4 km, but
    # 400 km at 16 km).
    from src.config_utils import load_data_config
    _res_km = load_data_config()["grid"]["target_res_m"] // 1000
    radius_px = int(round(cube_cfg.get("radius_km", 100) / _res_km))

    data_dir = cube_cfg.get("data_dir") or paths.get("data_dir") or load_data_config()["datasets_root"]
    hist_dir = cube_cfg.get("hist_dir") or paths.get("hist_dir")
    if not hist_dir:
        raise KeyError("latent_cube.hist_dir (or paths.hist_dir) must be set in esk_desk_config")
    if os.path.basename(hist_dir) != "yearly_states" and os.path.isdir(os.path.join(hist_dir, "yearly_states")):
        hist_dir = os.path.join(hist_dir, "yearly_states")

    z_dir = cube_cfg.get("z_dir") or desk_cfg.get("z_dir") or paths.get("desk_output_dir", "")
    model_path = cube_cfg.get("model_path") or os.path.join(paths.get("desk_output_dir", ""), "env_model_semisup.pth")
    z_ref_path = cube_cfg.get("z_ref_path") or os.path.join(z_dir, "Z.npy")
    mask_ref_path = cube_cfg.get("mask_ref_path") or os.path.join(z_dir, "valid_mask.npy")
    water_mask_path = cube_cfg.get("water_mask_path") or os.path.join(data_dir, "land_mask", f"ocean_mask_{_res_km}km.tif")
    output_dir = cube_cfg.get("output_dir") or os.path.join(paths.get("desk_output_dir", ""), "spacetime_cube")

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading masks and reference data...")
    with rasterio.open(water_mask_path) as src:
        water_data = src.read(1)
        land_mask = (water_data == 0).astype(bool)
        H, W = land_mask.shape

    z_ref_flat = np.load(z_ref_path)
    z_ref_mask = np.load(mask_ref_path)

    # Normalization + architecture come from the trainer's desk_meta.npz — one
    # source of truth, so the cube standardizes exactly as training did.
    import json as _json
    meta_path = cube_cfg.get("desk_meta") or os.path.join(paths.get("desk_output_dir", ""), "desk_meta.npz")
    dm = np.load(meta_path, allow_pickle=True)
    mu, sd = dm["mu"].astype(np.float32), dm["sd"].astype(np.float32)
    stream_dims = [int(d) for d in dm["stream_dims"]]
    latent_dim = int(dm["latent_dim"])
    spatial_kernel = int(dm["spatial_kernel"]) if "spatial_kernel" in dm else 0
    schema = _json.loads(str(dm["schema"]))

    # DESK may have trained on a truncation of the ESK Z (desk.latent_dim); the ESK
    # reference is saved at the max swept dim. Match the model: kernel-PCA columns are
    # eigenvalue-ordered, so Z[:, :latent_dim] is the exact top-latent_dim embedding.
    if z_ref_flat.shape[1] < latent_dim:
        raise ValueError(f"ESK z_ref has {z_ref_flat.shape[1]} dims < desk_meta latent_dim {latent_dim}")
    z_ref_flat = z_ref_flat[:, :latent_dim]
    z_dim = latent_dim
    z_static_grid = np.full((H, W, z_dim), np.nan, dtype=np.float32)
    z_static_grid[z_ref_mask] = z_ref_flat
    z_static_valid = ~np.isnan(z_static_grid).any(axis=-1)

    print(f"Loading N-stream model ({stream_dims}, spatial_kernel={spatial_kernel}) from {model_path}...")
    model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    year_files = sorted(glob.glob(os.path.join(hist_dir, cube_cfg.get("state_pattern", "state_*.npz"))))
    if not year_files:
        raise FileNotFoundError(f"No state files found in {hist_dir}")

    for fpath in tqdm(year_files, desc="Processing Years"):
        year = int(os.path.basename(fpath).split("_")[1].split(".")[0])   # state_{year}.npz

        cov = cio.load_state_stack(year, hist_dir, schema)   # (H, W, C), transforms applied
        # Grid-native forward: the spatial residual conv needs the whole grid, so
        # normalize in place (invalid cells zero-filled + masked) rather than gather.
        covn, valid_pixels = cio.norm_grid(cov, mu, sd)
        z_year = np.full((H, W, z_dim), np.nan, dtype=np.float32)

        if valid_pixels.sum() > 0:
            xg = torch.tensor(covn[None], dtype=torch.float32, device=device)
            mg = torch.tensor(valid_pixels[None], device=device)
            with torch.no_grad():
                z_out, _ = model(xg, mg)                       # (1, H, W, L)
            z_year[valid_pixels] = z_out[0].cpu().numpy()[valid_pixels]

        z_s1, mask_s1 = fill_gaps_stage1_spatial(
            z_year,
            valid_pixels,
            land_mask,
            radius_px=radius_px,
        )
        z_s2, mask_s2 = fill_gaps_stage2_static(z_s1, mask_s1, land_mask, z_static_grid, z_static_valid)
        z_final = fill_gaps_stage3_nearest(z_s2, mask_s2, land_mask)
        z_final[~land_mask] = np.nan

        out_name = f"Z_latent_{year}.npy"
        np.save(os.path.join(output_dir, out_name), z_final.astype(np.float32))

    print("Spatiotemporal Cube Generation Complete.")
    return output_dir


def main():
    """CLI entry: build the Z cube using the config at $ESK_DESK_CONFIG (or default)."""
    config_path = os.environ.get("ESK_DESK_CONFIG")
    build_spacetime_cube(config_path)


if __name__ == "__main__":
    main()

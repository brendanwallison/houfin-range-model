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
from community_encoder.train_DESK.model_arch import BMLPBlock, MultiInputAutoencoder


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
    if config is None:
        config = load_config()
    elif isinstance(config, (str, os.PathLike)):
        config = load_config(config)

    paths = config.get("paths", {})
    cube_cfg = config.get("latent_cube", {})
    desk_cfg = config.get("desk", {})

    data_dir = cube_cfg.get("data_dir") or paths.get("data_dir", "/home/breallis/datasets")
    hist_dir = cube_cfg.get("hist_dir") or paths.get("hist_dir", "/home/breallis/datasets/smoothed_prism_bui")
    if os.path.basename(hist_dir) != "yearly_states" and os.path.isdir(os.path.join(hist_dir, "yearly_states")):
        hist_dir = os.path.join(hist_dir, "yearly_states")

    z_dir = cube_cfg.get("z_dir") or desk_cfg.get("z_dir") or paths.get("desk_output_dir", "")
    model_path = cube_cfg.get("model_path") or os.path.join(paths.get("desk_output_dir", ""), "env_model_semisup.pth")
    z_ref_path = cube_cfg.get("z_ref_path") or os.path.join(z_dir, "Z.npy")
    mask_ref_path = cube_cfg.get("mask_ref_path") or os.path.join(z_dir, "valid_mask.npy")
    water_mask_path = cube_cfg.get("water_mask_path") or os.path.join(data_dir, "land_mask", "ocean_mask_4km.tif")
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

    z_dim = z_ref_flat.shape[1]
    z_static_grid = np.full((H, W, z_dim), np.nan, dtype=np.float32)
    z_static_grid[z_ref_mask] = z_ref_flat
    z_static_valid = ~np.isnan(z_static_grid).any(axis=-1)

    print("Loading 2023 state to derive normalization stats...")
    state_2023 = np.load(os.path.join(hist_dir, "state_2023_bio_ema10.npz"))
    p_temp = state_2023["prism"]
    b_temp = state_2023["bui"]

    mask_intersect = (~np.isnan(p_temp).any(-1)) & (~np.isnan(b_temp).any(-1)) & z_ref_mask
    p_flat = p_temp[mask_intersect]
    b_flat = b_temp[mask_intersect]

    p_mu = p_flat.mean(0).astype(np.float32)
    p_sd = p_flat.std(0).astype(np.float32)
    b_mu = (b_flat**0.1).mean(0).astype(np.float32)
    b_sd = (b_flat**0.1).std(0).astype(np.float32)

    print(f"Loading model from {model_path}...")
    model = MultiInputAutoencoder(prism_dim=p_temp.shape[2], bui_dim=b_temp.shape[2], latent_dim=z_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    year_files = sorted(glob.glob(os.path.join(hist_dir, cube_cfg.get("state_pattern", "state_*.npz"))))
    if not year_files:
        raise FileNotFoundError(f"No state files found in {hist_dir}")

    for fpath in tqdm(year_files, desc="Processing Years"):
        fname = os.path.basename(fpath)
        year = int(fname.split("_")[1])

        data = np.load(fpath)
        p_raw = data["prism"]
        b_raw = data["bui"]

        valid_pixels = (~np.isnan(p_raw).any(-1)) & (~np.isnan(b_raw).any(-1))
        z_year = np.full((H, W, z_dim), np.nan, dtype=np.float32)

        if valid_pixels.sum() > 0:
            p_in = torch.tensor(p_raw[valid_pixels], dtype=torch.float32)
            b_in = torch.tensor(b_raw[valid_pixels], dtype=torch.float32)
            p_in = (p_in - p_mu) / (p_sd + 1e-6)
            b_in = (b_in**0.1 - b_mu) / (b_sd + 1e-6)

            with torch.no_grad():
                z_out, _ = model(p_in.to(device), b_in.to(device))
            z_year[valid_pixels] = z_out.cpu().numpy()

        z_s1, mask_s1 = fill_gaps_stage1_spatial(
            z_year,
            valid_pixels,
            land_mask,
            radius_px=cube_cfg.get("radius_px", 25),
        )
        z_s2, mask_s2 = fill_gaps_stage2_static(z_s1, mask_s1, land_mask, z_static_grid, z_static_valid)
        z_final = fill_gaps_stage3_nearest(z_s2, mask_s2, land_mask)
        z_final[~land_mask] = np.nan

        out_name = f"Z_latent_{year}.npy"
        np.save(os.path.join(output_dir, out_name), z_final.astype(np.float32))

    print("Spatiotemporal Cube Generation Complete.")
    return output_dir


def main():
    config_path = os.environ.get("ESK_DESK_CONFIG")
    build_spacetime_cube(config_path)


if __name__ == "__main__":
    main()

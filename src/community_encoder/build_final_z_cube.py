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
from src.data.masks import read_land_mask

# Per-cell provenance codes written to ``Z_fill_stage_{year}.npy`` (uint8).
# ``FILL_PREDICTED`` is the only class that is actual model output; it doubles as the
# per-year valid mask (``stage == FILL_PREDICTED``).
FILL_PREDICTED = 0     # DESK forward on finite covariates
FILL_SPATIAL = 1       # stage 1: linear interpolation from THIS year's valid cells
FILL_STATIC = 2        # stage 2: the year-INVARIANT static ESK field -> no temporal signal
FILL_NEAREST = 3       # stage 3: nearest from THIS year's valid cells
FILL_NODATA = 255      # ocean / off-grid (Z is NaN)
FILL_LABELS = {FILL_PREDICTED: "predicted", FILL_SPATIAL: "spatial",
               FILL_STATIC: "static", FILL_NEAREST: "nearest", FILL_NODATA: "nodata"}


def fill_provenance(land_mask, valid_pixels, mask_s1, mask_s2):
    """Per-cell fill provenance from the masks the three fill stages already return.

    Each stage reports the validity it achieved, so the classes are exact differences
    rather than a re-derivation: what stage 1 added over the model's own cells, what
    stage 2 added over stage 1, and whatever land remained for stage 3. The four land
    classes partition ``land_mask`` exactly; everything else is ``FILL_NODATA``.
    """
    stage = np.full(land_mask.shape, FILL_NODATA, dtype=np.uint8)
    stage[land_mask & (~mask_s2)] = FILL_NEAREST
    stage[land_mask & mask_s2 & (~mask_s1)] = FILL_STATIC
    stage[land_mask & mask_s1 & (~valid_pixels)] = FILL_SPATIAL
    stage[land_mask & valid_pixels] = FILL_PREDICTED
    return stage


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
    three-stage gap fill, and writes ``Z_latent_{year}.npy`` plus
    ``Z_fill_stage_{year}.npy`` (uint8 provenance, see ``FILL_*`` below).
    ``config`` is the encoder config (dict or path); defaults to the repo config.

    **Read the provenance before interpreting the cube.** Only ``FILL_PREDICTED``
    cells are model output; the covariate footprint is much smaller than the land
    mask (validity is all-or-nothing over ~295 channels), and the rest is filled.
    ``FILL_STATIC`` cells in particular get a *year-invariant* reference field, so
    their Z is bit-identical across years and any temporal statistic on them is
    degenerate -- e.g. turnover ``1 - Z.Z'`` collapses to ``1 - ||Z||^2`` ~= 0.
    That artifact is why this file is written: it was previously indistinguishable
    from a real prediction of "no community change".
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
    # roughly 675 km at the current 27 km grid).
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
    land_mask = read_land_mask(water_mask_path)
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
    kernel = str(dm["kernel"]) if "kernel" in dm else ""
    centered = bool(dm["centered"]) if "centered" in dm else True
    if kernel != "ruzicka" or centered:
        raise ValueError(f"cube requires uncentered Ružička DESK metadata; "
                         f"got kernel={kernel!r}, centered={centered}")

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

    # The schema above came from desk_meta.npz (what the model was TRAINED on); the
    # states about to be encoded are whatever is on disk now. mu/sd are positional,
    # so a states rebuild with different channels must fail loudly here.
    cio.assert_schema_compatible(schema, cio.load_schema(hist_dir),
                                 context="build_final_z_cube")

    # Pass 1: forward every year (temporal order) to per-year raw Z + valid mask.
    years, z_raws, valids = [], [], []
    for fpath in tqdm(year_files, desc="Encoding Years"):
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
        years.append(year); z_raws.append(z_year); valids.append(valid_pixels)

    # DESK trains a raw instantaneous encoder whose output-EMA is used only to match
    # lagged community targets during training. The mechanistic population model below
    # this cube already supplies demographic lag, so export z_raw here; applying the EMA
    # again would double-count lagged response.
    hl = float(dm["ema_half_life"]) if "ema_half_life" in dm else float("nan")
    ema_on = bool(dm["output_ema"]) if "output_ema" in dm else False
    if ema_on:
        print(f"Exporting instantaneous z_raw (training EMA half-life={hl:.2f} yr is not applied).")

    fill_counts = {}
    for year, z_year, valid_pixels in tqdm(list(zip(years, z_raws, valids)), desc="Filling Years"):
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

        # Provenance: without it a gap-filled cell is indistinguishable from a
        # prediction, and stage-2 cells silently read as "no community change".
        stage = fill_provenance(land_mask, valid_pixels, mask_s1, mask_s2)
        np.save(os.path.join(output_dir, f"Z_fill_stage_{year}.npy"), stage)
        counts = {name: int((stage == code).sum()) for code, name in FILL_LABELS.items()}
        fill_counts[int(year)] = counts
        n_land = int(land_mask.sum())
        print(f"   -> {year} provenance: predicted {counts['predicted']}/{n_land} "
              f"({100 * counts['predicted'] / max(n_land, 1):.1f}%), spatial {counts['spatial']}, "
              f"static {counts['static']} (year-invariant), nearest {counts['nearest']}", flush=True)

    with open(os.path.join(output_dir, "cube_meta.json"), "w", encoding="utf-8") as fh:
        _json.dump({
            "kernel": kernel, "centered": centered, "latent_dim": latent_dim,
            "kernel_contract": "Z(x) dot Z(x') ~= uncentered Ruzicka(x,x')",
            "years": years,
            "fill_stage_codes": {str(k): v for k, v in FILL_LABELS.items()},
            "fill_counts_by_year": fill_counts,
            "fill_note": ("only 'predicted' cells are model output; 'static' cells carry a "
                          "year-invariant reference field, so temporal statistics on them are "
                          "degenerate (turnover 1-Z.Z' collapses to ~0)"),
            "temporal_output": "raw_instantaneous",
            "training_output_ema": ema_on,
            "training_ema_half_life": hl if np.isfinite(hl) else None,
        }, fh, indent=2)

    print("Spatiotemporal Cube Generation Complete.")
    return output_dir


def main():
    """CLI entry: build the Z cube using the config at $ESK_DESK_CONFIG (or default)."""
    config_path = os.environ.get("ESK_DESK_CONFIG")
    build_spacetime_cube(config_path)


if __name__ == "__main__":
    main()

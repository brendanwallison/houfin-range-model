import os
import glob
import json
import hashlib
import uuid
import pickle
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as project_coords

import jax.numpy as jnp
from src.model.build_kernels import build_simulation_struct, dispersal_spec
from src.config_utils import load_age_model_config
from src.data.masks import read_land_mask
from src.temporal import (assert_contiguous, disease_timestep, invasion_timestep,
                          load_timeline, model_years, year_to_index)

_cfg = load_age_model_config()
RAW_Z_DIR = _cfg["raw_z_dir"]
BBS_DATA_NPZ = _cfg["bbs_npz"]
MASK_FILE = _cfg["ocean_mask"]
DISEASE_ARRIVAL_MAP = _cfg["disease_arrival_map"]
OUTPUT_DIR = _cfg["input_dir"]

# No AGG_FACTOR: every input (Z/Z_disp, BBS grid, mask) is already produced at
# the model grid (grid.target_res_m, see data_config.json). This stage consumes
# them as-is and is resolution-agnostic. The old code built Z at 4 km and
# mean-pooled it 4x4 here, which is meaningless for a kernel-PCA embedding.
# The timeline (first/end year, invasion) comes from src/temporal.py; the
# realized model years are read from the Z_disp files on disk and reconciled
# against it (see ingest_data). Nothing here hardcodes a start/end year.
MODEL_LATENT_DIM = int(_cfg.get("latent_dim", 64))
SOURCE_LATENT_DIM = int(_cfg.get("source_latent_dim", MODEL_LATENT_DIM))
KERNEL_CONTRACT = dict(_cfg.get("kernel_contract", {}))
DISPERSAL_SPEC = dispersal_spec(_cfg)
POPULATION_SPEC = dict(_cfg["population_model"])

# --- DISEASE-TERM SPATIAL BASIS SETTINGS ---
# These replace the old st_basis_space/time_frequencies. The disease effect on K is
# no longer a generic spatiotemporal field (967 free cosine coefficients over
# space x time, which annihilated eastern K by absorbing every kind of spatial
# misfit); it is now a structured severity x onset x recovery form whose only
# spatially varying pieces are two SMOOTH, TIME-INDEPENDENT fields. See
# src/model/age_fields.py.
DISEASE_PRIOR_SPEC = dict(POPULATION_SPEC["disease_prior"])
N_FREQ_SEVERITY = int(DISEASE_PRIOR_SPEC["severity_space_frequencies"])
N_FREQ_LAG = int(DISEASE_PRIOR_SPEC["lag_space_frequencies"])
# Continental time trend on K (see generate_k_trend_basis).
K_TREND_SPEC = dict(POPULATION_SPEC["k_trend"])


def generate_spatial_basis(Ny, Nx, land_rows, land_cols, n_freq, label=""):
    """Smooth 2-D cosine basis on the land cells, centered over land.

    Returns ``(n_basis, N_land)`` with ``n_basis = (n_freq+1)^2 - 1`` (the global
    constant is dropped). Used for the disease term's two spatial fields:
    severity and onset lag.

    **Each function is centered over land cells.** The cosines are orthogonal over
    the full rectangular grid, but land is an irregular subset of it, so their
    land-restricted means are NOT zero -- an uncentered basis lets its
    coefficients shift the field's continental LEVEL, which then trades off
    against the scalar that is supposed to own that level (``disease_mu_sev`` for
    severity, ``disease_lag0`` for timing). Centering makes the split exact: the
    scalars carry the level, the coefficients carry only regional deviation from
    it, so both are reportable.

    There is deliberately NO time axis. The old basis was
    ``O(n_basis * Time * N_land)`` and cost ~2 GiB of VRAM; these are a few MB.
    Anything the disease effect does in time now goes through the structured
    onset gate and recovery curve, not through free coefficients.
    """
    print(f"  Constructing spatial basis{label}: n_freq={n_freq} "
          f"-> {(n_freq + 1) ** 2 - 1} functions...")
    y_coord = np.linspace(0, 1, Ny)[land_rows]  # (N_land,)
    x_coord = np.linspace(0, 1, Nx)[land_cols]  # (N_land,)

    basis_list = []
    for i in range(n_freq + 1):
        for j in range(n_freq + 1):
            if i == 0 and j == 0:
                continue  # the constant is owned by the level scalar
            wave = np.cos(i * np.pi * y_coord) * np.cos(j * np.pi * x_coord)
            basis_list.append((wave - wave.mean()).astype(np.float32))

    basis = np.stack(basis_list, axis=0)  # (n_basis, N_land)
    land_means = np.abs(basis.mean(axis=1))
    if not (land_means < 1e-5).all():
        raise ValueError(f"spatial basis{label} is not land-centered "
                         f"(max |land mean| = {land_means.max():.2e})")
    return basis


def generate_k_trend_basis(Time, n_basis):
    """Continental (spatially uniform) smooth time basis for K, centered over time.

    Returns ``(n_basis, Time)``: ``cos(m*pi*t)`` for ``m = 1..n_basis`` over the FULL
    model timeline, each centered so its temporal mean is zero. Centering is what
    keeps ``alpha_k`` the owner of K's level -- these coefficients can only say how
    capacity drifted, not where it sits.

    Why this exists: K's only temporal degree of freedom used to be the disease term,
    so "modern capacity is below 1970s capacity" -- which BBS demands, and which has
    causes besides conjunctivitis -- could only be expressed as disease, and the
    disease term saturated trying. Kept to a handful of basis functions and NO
    spatial dependence on purpose: it must not be able to compete with the disease
    term's spatial pattern or manufacture year-to-year wiggle.
    """
    t = np.linspace(0.0, 1.0, Time)
    rows = []
    for m in range(1, int(n_basis) + 1):
        wave = np.cos(m * np.pi * t)
        rows.append((wave - wave.mean()).astype(np.float32))
    basis = np.stack(rows, axis=0)
    if not (np.abs(basis.mean(axis=1)) < 1e-5).all():
        raise ValueError("K trend basis is not time-centered")
    return basis


def load_disease_onset(tif_path, Ny, Nx, land_rows, land_cols, first_year):
    """Per-land-cell disease arrival, as a (fractional) MODEL TIMESTEP index.

    Reads the arrival-year surface produced by
    ``scripts/build_disease_arrival_map.py`` and converts calendar years to
    timestep units by subtracting ``first_year`` -- the model timeline is
    contiguous (``assert_contiguous``), so this is the same mapping
    ``year_to_index`` performs, extended to the fractional values kernel
    smoothing produces. The forward model's onset gate compares this directly
    against ``t_idx``, so the conversion must happen here and exactly once.

    The raster must be cell-for-cell on the model grid; a shape mismatch raises
    rather than being resampled, so a stale map from an older grid resolution
    cannot silently misalign the epidemic front by hundreds of km.
    """
    with rasterio.open(tif_path) as src:
        if (src.height, src.width) != (Ny, Nx):
            raise ValueError(f"{tif_path} is {src.height}x{src.width}, model grid is "
                             f"{Ny}x{Nx}; rebuild it with scripts/build_disease_arrival_map.py")
        arrival = src.read(1).astype(np.float64)
        nodata = src.nodata
    if nodata is not None:
        arrival = np.where(arrival == nodata, np.nan, arrival)
    onset = arrival[land_rows, land_cols] - float(first_year)
    if not np.isfinite(onset).all():
        raise ValueError(f"{tif_path} has nodata on {int((~np.isfinite(onset)).sum())} "
                         f"land cells; the arrival surface must cover all land")
    return onset.astype(np.float32)


def load_land_metadata(tif_path):
    with rasterio.open(tif_path) as src:
        res_x = src.res[0]
        if (src.crs and 'metre' in src.crs.linear_units.lower()) or (res_x > 100):
            cell_size_km = res_x / 1000.0
        else:
            cell_size_km = res_x * 111.0
    return cell_size_km


def load_ocean_land_mask(tif_path):
    """Land boolean grid (True = land) from an ocean-mask raster (water encoded nonzero).

    Matches the convention used by bbs.py / build_final_z_cube.py (``land = raster == 0``),
    so the age-model mask and the BBS npz's embedded land mask can be compared cell-for-cell.
    """
    return read_land_mask(tif_path)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# --- Ingestion guards (see plan E1/E2/E3): turn silent grid/timeline mismatches, created by
# the 25->27 km / year-span migration, into loud failures. Pure so they unit-test directly. ---

def require_same_grid(name, got_hw, expected_hw):
    """Raise unless a product's (H,W) matches the BBS/model grid; else silent misalignment."""
    if tuple(got_hw) != tuple(expected_hw):
        raise ValueError(f"{name} grid {tuple(got_hw)} != BBS/model grid {tuple(expected_hw)}; "
                         f"regenerate {name} and the BBS npz at the same grid "
                         f"(grid.target_res_m in data_config.json).")


def require_mask_match(mask_land, bbs_land, path):
    """Raise unless the age-model ocean mask's land cells equal the BBS npz's land mask."""
    mask_land = np.asarray(mask_land, bool); bbs_land = np.asarray(bbs_land, bool)
    if mask_land.shape != bbs_land.shape:
        raise ValueError(f"ocean_mask {path} shape {mask_land.shape} != BBS land grid "
                         f"{bbs_land.shape}; regenerate the BBS npz and mask at the same grid.")
    if not np.array_equal(mask_land, bbs_land):
        n_diff = int(np.sum(mask_land != bbs_land))
        raise ValueError(f"ocean_mask {path} land cells differ from the BBS npz land mask "
                         f"({n_diff} cells); they must be the identical grid.")


def require_pseudo_zero_coverage(start_year, first_year, invasion_year, end_year):
    """Raise if the cube starts after the last pre-invasion year, which would silently
    drop ALL pseudo-zero absence slices [first_year, invasion_year-1] the BBS model needs."""
    if start_year > invasion_year - 1:
        raise ValueError(
            f"Z cube starts at {start_year}, after the last pre-invasion year "
            f"{invasion_year - 1}: all pseudo-zero absence slices "
            f"({first_year}-{invasion_year - 1}) would be dropped. Rebuild states + cube "
            f"over the full timeline ({first_year}-{end_year}).")

def get_grid_location(tif_path, lat, lon):
    with rasterio.open(tif_path) as src:
        if src.crs != 'EPSG:4326':
            xs, ys = project_coords('EPSG:4326', src.crs, [lon], [lat])
            x, y = xs[0], ys[0]
        else:
            x, y = lon, lat
        row, col = src.index(x, y)
        return int(row), int(col)

# Main Execution
def ingest_data():
    print(f"--- Starting Data Ingestion (grid-native, latent_dim={MODEL_LATENT_DIM}, "
          f"kernel={KERNEL_CONTRACT.get('kernel', 'unspecified')}, "
          f"centered={KERNEL_CONTRACT.get('centered', 'unspecified')}) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path_meta_path = os.path.join(RAW_Z_DIR, "path_feature_meta.json")
    if not os.path.exists(path_meta_path):
        raise FileNotFoundError(f"path features lack complete provenance: {path_meta_path}")
    with open(path_meta_path, encoding="utf-8") as fh:
        path_meta = json.load(fh)
    source_contract = path_meta.get("kernel_contract") or {}
    expected_contract = {
        "kernel": KERNEL_CONTRACT.get("kernel", "ruzicka"),
        "centered": bool(KERNEL_CONTRACT.get("centered", False)),
        "latent_dim": SOURCE_LATENT_DIM,
    }
    for key, expected in expected_contract.items():
        if source_contract.get(key) != expected:
            raise ValueError(f"path-feature contract {key}={source_contract.get(key)!r} "
                             f"!= age-model expectation {expected!r}")
    if source_contract.get("temporal_output") != "raw_instantaneous":
        raise ValueError("age model requires instantaneous raw DESK Z; rebuild the cube/path "
                         f"features (got temporal_output={source_contract.get('temporal_output')!r})")
    if path_meta.get("dispersal") != DISPERSAL_SPEC:
        raise ValueError("path-feature dispersal specification differs from age_model_config; "
                         "regenerate Z_disp before ingestion")
    if int(path_meta.get("integration_steps", -1)) != DISPERSAL_SPEC["path_integration_steps"]:
        raise ValueError("path-feature integration step count differs from age_model_config")
    if path_meta.get("mask_sha256") != _sha256(MASK_FILE):
        raise ValueError("path features were generated with a different ocean/land mask")
    if abs(float(path_meta.get("kernel_mass", np.nan)) - 1.0) > 2e-5:
        raise ValueError(f"path-feature juvenile kernel is not mass-conserving: "
                         f"{path_meta.get('kernel_mass')}")
    
    # 1. Load Raw Data (Fine Grid)
    if not os.path.exists(BBS_DATA_NPZ):
        raise FileNotFoundError(f"BBS data not found: {BBS_DATA_NPZ}")
        
    bbs_data = np.load(BBS_DATA_NPZ)
    land_mask = (bbs_data['land'].astype(np.float32) > 0.5).astype(int)
    # Seed map is a DIMENSIONLESS shape (core=1, margin=observed ratio); the model
    # scales it by the fitted capacity level times a configured fraction, so the seed
    # is always a known fraction of capacity. Older npz files carry absolute values
    # (initpop_route_counts, or initpop_density in relative units) -- normalize those
    # to a shape so a legacy file still runs, since only the pattern is used now.
    if 'initpop_shape' in bbs_data:
        initpop_map = bbs_data['initpop_shape'] * land_mask
    else:
        legacy = bbs_data['initpop_route_counts'] if 'initpop_route_counts' in bbs_data \
            else bbs_data['initpop_density']
        peak = float(np.max(legacy)) or 1.0
        initpop_map = (legacy / peak) * land_mask
        print(f"  [compat] legacy initpop normalized to a shape (peak was {peak:g})")
    print(f"  Init shape: core {initpop_map.max():.2f}, "
          f"nonzero cells {int((initpop_map > 0).sum())}")  # already at grid res

    Ny, Nx = land_mask.shape
    land_rows, land_cols = np.where(land_mask)
    N_land = len(land_rows)
    print(f"  Grid: {Ny}x{Nx}, Land Pixels: {N_land}")

    # Guard (E2): the age-model ocean mask (used for cell-size km + invasion location) and
    # the BBS npz's embedded land mask (used for Ny/Nx + land indexing) must be the SAME grid
    # -- both derive from ocean_mask_{res}km.tif. A stale/mismatched mask (e.g. a leftover
    # 25 km file) would compute cell size + the invasion cell on a different lattice, silently.
    require_mask_match(load_ocean_land_mask(MASK_FILE), land_mask, MASK_FILE)

    # 3. Process Observations
    print("  Processing Observations...")
    orig_rows = bbs_data['obs_rows']
    orig_cols = bbs_data['obs_cols']
    orig_years = bbs_data['obs_year']
    orig_counts = bbs_data['observed_results']
    n_pseudo_orig = int(bbs_data['N_pseudo'])
    # Per-observation quality tier (0 = standard, 1 = mx_unprocessed). Older BBS
    # npz files predate this field -> default everything to standard.
    orig_quality = (bbs_data['obs_quality'] if 'obs_quality' in bbs_data.files
                    else np.zeros_like(orig_rows))

    # Split Real vs Pseudo
    real_indices = slice(n_pseudo_orig, None)
    pseudo_indices = slice(0, n_pseudo_orig)

    # -- Real Data (already at grid resolution) --
    r_rows_coarse = orig_rows[real_indices]
    r_cols_coarse = orig_cols[real_indices]
    r_years = orig_years[real_indices]
    r_counts = orig_counts[real_indices]
    r_quality = orig_quality[real_indices]
    
    # -- Pseudo Data Subsampling --
    # a. Calculate Density of Real Data
    real_locs = np.vstack((r_rows_coarse, r_cols_coarse)).T
    unique_real_locs = np.unique(real_locs, axis=0)
    sampling_density = len(unique_real_locs) / N_land
    
    # b. Get Unique Coarse Locations of Pseudo Data
    p_rows_coarse = orig_rows[pseudo_indices]
    p_cols_coarse = orig_cols[pseudo_indices]
    p_years_fine = orig_years[pseudo_indices]

    pseudo_locs = np.vstack((p_rows_coarse, p_cols_coarse)).T
    unique_pseudo_locs = np.unique(pseudo_locs, axis=0)
    
    # c. Subsample
    n_target = int(len(unique_pseudo_locs) * sampling_density)
    n_target = max(n_target, 50)
    
    print(f"  Subsampling Pseudo-Zeros: Target {n_target} sites (Density {sampling_density:.4f})")
    
    rng = np.random.default_rng(42)
    if len(unique_pseudo_locs) > n_target:
        chosen_indices = rng.choice(len(unique_pseudo_locs), n_target, replace=False)
        chosen_locs = unique_pseudo_locs[chosen_indices]
    else:
        chosen_locs = unique_pseudo_locs
    
    # d. Expand Chosen Locs over Years
    # Note: p_years_fine contains all years. We just need the unique years range.
    years_range = np.unique(p_years_fine)
    
    final_p_rows, final_p_cols, final_p_years = [], [], []
    for yr in years_range:
        final_p_rows.append(chosen_locs[:, 0])
        final_p_cols.append(chosen_locs[:, 1])
        final_p_years.append(np.full(len(chosen_locs), yr))
        
    final_p_rows = np.concatenate(final_p_rows)
    final_p_cols = np.concatenate(final_p_cols)
    final_p_years = np.concatenate(final_p_years)
    final_p_counts = np.zeros_like(final_p_years)
    final_p_quality = np.zeros_like(final_p_years)  # pseudo-zeros are standard tier

    # -- Merge --
    obs_rows = np.concatenate([final_p_rows, r_rows_coarse])
    obs_cols = np.concatenate([final_p_cols, r_cols_coarse])
    obs_year = np.concatenate([final_p_years, r_years])
    observed_results = np.concatenate([final_p_counts, r_counts])
    obs_quality = np.concatenate([final_p_quality, r_quality])

    # Bounds Check
    valid_locs = (obs_rows >= 0) & (obs_rows < Ny) & (obs_cols >= 0) & (obs_cols < Nx)
    obs_rows = obs_rows[valid_locs]
    obs_cols = obs_cols[valid_locs]
    obs_year = obs_year[valid_locs]
    observed_results = observed_results[valid_locs]
    obs_quality = obs_quality[valid_locs]
    
    print(f"  Final Observations: {len(observed_results)}")

    # 4. Stream Z Data
    z_files = sorted(glob.glob(os.path.join(RAW_Z_DIR, "Z_disp_*.npz")))
    file_map = {int(os.path.basename(f).split('_')[2].split('.')[0]): f for f in z_files}
    sorted_years = sorted(file_map.keys())
    if not sorted_years:
        raise FileNotFoundError(f"no Z_disp_*.npz files in {RAW_Z_DIR}")
    assert_contiguous(sorted_years)  # the year->index mapping requires no gaps
    start_year_model, end_year_model = min(sorted_years), max(sorted_years)
    realized_years = np.array(sorted_years)
    Time = len(realized_years)
    _tl = load_timeline()
    print(f"  Timeline: {start_year_model}-{end_year_model} ({Time} years); "
          f"config timeline {_tl['first_year']}-{_tl['end_year']}")
    expected_years = model_years(_tl)
    if sorted_years != expected_years:
        missing = sorted(set(expected_years) - set(sorted_years))
        extra = sorted(set(sorted_years) - set(expected_years))
        raise ValueError(f"production model inputs require the complete canonical timeline "
                         f"{expected_years[0]}-{expected_years[-1]}; "
                         f"missing={missing[:10]}, extra={extra[:10]}")
    if [int(y) for y in path_meta.get("years", [])] != expected_years:
        raise ValueError("path_feature_meta years do not match the canonical timeline")

    # Guard (E3): the BBS model's pre-invasion pseudo-zeros live in [first_year, invasion-1].
    # Obs are filtered to years present in the cube (below), so if the cube starts after the
    # last pre-invasion year, ALL pseudo-zeros are silently dropped -- gutting the pre-invasion
    # absence signal. Require the cube to cover the pre-invasion span.
    _inv_year, _first_year = int(_tl["invasion_year"]), int(_tl["first_year"])
    require_pseudo_zero_coverage(start_year_model, _first_year, _inv_year, int(_tl["end_year"]))

    peek = np.load(file_map[start_year_model])
    # Guard (E1): the Z cube and the BBS/model grid must share the exact lattice -- the cube
    # is gathered onto BBS-derived (land_rows, land_cols), so a shape mismatch (e.g. a stale
    # 25 km BBS npz vs a fresh 27 km cube) would IndexError or silently gather wrong cells.
    require_same_grid("Z cube", peek['Z_raw'].shape[1:3], (Ny, Nx))
    require_same_grid("Z_disp", peek['Z_disp'].shape[1:3], (Ny, Nx))
    available_M = int(peek['Z_raw'].shape[-1])
    if available_M < MODEL_LATENT_DIM:
        raise ValueError(f"Z cube has {available_M} features but age-model latent_dim="
                         f"{MODEL_LATENT_DIM}; rerun ESK -> DESK -> cube at the contracted width")
    M = MODEL_LATENT_DIM
    if available_M != SOURCE_LATENT_DIM:
        raise ValueError(f"path features provide {available_M} dimensions but "
                         f"age_model_config.source_latent_dim={SOURCE_LATENT_DIM}")
    if M < available_M:
        print(f"  Explicit configured truncation: top {M}/{available_M} uncentered "
              "Ružička eigenfeatures (age_model_config.latent_dim)")
    K = peek['Z_disp'].shape[-1]
    expected_labels = [str(x) for x in path_meta.get("kernel_labels", [])]
    expected_build_id = path_meta.get("build_id")
    if not expected_build_id:
        raise ValueError("path-feature metadata lacks a transactional build_id")
    if K != int(path_meta.get("kernel_count", -1)) or len(expected_labels) != K:
        raise ValueError("path-feature kernel count/labels are inconsistent")
    
    ingest_id = uuid.uuid4().hex
    z_gather_name = f"Z_gathered_{ingest_id}.dat"
    z_disp_name = f"Z_disp_gathered_{ingest_id}.dat"
    z_gather_path = os.path.join(OUTPUT_DIR, z_gather_name)
    Z_gathered = np.memmap(z_gather_path, dtype='float32', mode='w+', shape=(Time, N_land, M))
    z_disp_path = os.path.join(OUTPUT_DIR, z_disp_name)
    Z_disp_gathered = np.memmap(z_disp_path, dtype='float32', mode='w+', shape=(Time, N_land, K, M))

    print("  Streaming Z Data (already at grid resolution; no pooling)...")
    for t, year in enumerate(realized_years):
        data = np.load(file_map[year])
        expected_raw_shape = (1, Ny, Nx, SOURCE_LATENT_DIM)
        expected_disp_shape = (1, Ny, Nx, SOURCE_LATENT_DIM, K)
        if (tuple(data['Z_raw'].shape) != expected_raw_shape or
                tuple(data['Z_disp'].shape) != expected_disp_shape):
            raise ValueError(f"{file_map[year]} violates source_latent_dim={SOURCE_LATENT_DIM}: "
                             f"Z_raw {data['Z_raw'].shape}, Z_disp {data['Z_disp'].shape}")
        labels = [str(x) for x in data["labels"].tolist()]
        if labels != expected_labels:
            raise ValueError(f"{file_map[year]} kernel labels/order differ from path metadata")
        if str(data["build_id"].item()) != expected_build_id:
            raise ValueError(f"{file_map[year]} belongs to an incomplete/different path build")
        if not np.array_equal(np.asarray(data["land_mask"]) > 0.5, land_mask > 0):
            raise ValueError(f"{file_map[year]} land mask differs from BBS/model grid")
        if not np.isclose(float(data["cell_size_km"]), float(path_meta["cell_size_km"])):
            raise ValueError(f"{file_map[year]} cell size differs from path metadata")
        raw_land = data["Z_raw"][0][land_mask > 0]
        disp_land = data["Z_disp"][0][land_mask > 0]
        if not np.isfinite(raw_land).all() or not np.isfinite(disp_land).all():
            raise ValueError(f"{file_map[year]} contains non-finite Z values on land")
        z = np.nan_to_num(data['Z_raw'][0])
        disp = np.nan_to_num(data['Z_disp'][0].transpose(0, 1, 3, 2))

        Z_gathered[t] = z[land_rows, land_cols, :M]
        Z_disp_gathered[t] = disp[land_rows, land_cols, :, :M]
        if t % 5 == 0: print(f"    Processed {year}...", end='\r')

    Z_gathered.flush(); Z_disp_gathered.flush()
    print("\n  Data Streaming Complete.")

    # 5. Disease term inputs (see src/model/age_fields.py). The effect on K is
    # severity(x) * onset_gate(x,t) * (1 - recovery(t - arrival)); the only
    # spatially varying free pieces are two smooth, time-independent fields, so
    # what used to be a 2 GiB spatiotemporal array is now a few MB.
    dis_timestep = disease_timestep(_tl, first_year=start_year_model)
    if dis_timestep < 0 or dis_timestep >= Time:
        raise ValueError(f"disease_start_year {_tl['disease_start_year']} lies outside the "
                         f"realized timeline {start_year_model}..{start_year_model + Time - 1}")
    disease_sev_basis = generate_spatial_basis(Ny, Nx, land_rows, land_cols,
                                               N_FREQ_SEVERITY, label=" (severity)")
    disease_lag_basis = generate_spatial_basis(Ny, Nx, land_rows, land_cols,
                                               N_FREQ_LAG, label=" (onset lag)")
    print(f"  Disease basis footprint: "
          f"{(disease_sev_basis.nbytes + disease_lag_basis.nbytes) / 1e6:.2f} MB "
          f"({disease_sev_basis.shape[0]} severity + {disease_lag_basis.shape[0]} lag "
          f"coefficients; epizootic window starts at index {dis_timestep})")

    # 5b. Exogenous arrival map -> the onset gate (see load_disease_onset).
    disease_onset = load_disease_onset(DISEASE_ARRIVAL_MAP, Ny, Nx,
                                       land_rows, land_cols, start_year_model)
    print(f"  Disease onset: arrival years "
          f"{disease_onset.min() + start_year_model:.1f}-"
          f"{disease_onset.max() + start_year_model:.1f} over {N_land} land cells")
    # Arrival year centered and scaled to DECADES, so the severity model can carry
    # "populations reached later were hit less hard" (more genetic diversity in the
    # west) as a single coefficient instead of spending spatial-field capacity on a
    # pattern that is essentially the arrival gradient itself.
    disease_onset_decades = ((disease_onset - disease_onset.mean()) / 10.0).astype(np.float32)

    # 5c. Continental time basis for K's drift, over the FULL timeline (unlike the
    # disease term, capacity drift is not tied to the epizootic window).
    k_trend_basis = generate_k_trend_basis(Time, K_TREND_SPEC["n_basis"])
    print(f"  K trend basis: {k_trend_basis.shape[0]} time-centered cosines over "
          f"{Time} years ({k_trend_basis.nbytes / 1e3:.1f} kB)")

    # 6. Build Kernels
    # MASK_FILE must be the canonical 27 km model-grid mask so cell size / invasion
    # location are on the same grid as Z and the observations.
    cell_size_km = load_land_metadata(MASK_FILE)
    print(f"  Cell Size: {cell_size_km:.2f} km")
    
    sim_struct = build_simulation_struct(
        land=jnp.array(land_mask),
        cell_size=cell_size_km,
        adult_mdd=DISPERSAL_SPEC["adult_mdd_km"],
        juvenile_mdd=DISPERSAL_SPEC["juvenile_mdd_km"],
        adult_shape=DISPERSAL_SPEC["adult_shape"],
        juvenile_shape=DISPERSAL_SPEC["juvenile_shape"],
        radii_splits=DISPERSAL_SPEC["juvenile_radial_splits_km"],
    )
    if [str(x) for x in sim_struct["labels"]] != expected_labels:
        raise ValueError("forward-model juvenile kernel labels differ from Z_disp labels")

    inv_row, inv_col = get_grid_location(
        MASK_FILE,
        float(POPULATION_SPEC["invasion_lat"]),
        float(POPULATION_SPEC["invasion_lon"]),
    )

    # Keep obs whose year is actually in the model timeline, then map year->index
    # via a gap-safe lookup (not year - start subtraction). See src/temporal.py.
    year_set = set(int(y) for y in realized_years)
    valid_obs_mask = np.array([int(y) in year_set for y in obs_year])
    _n_drop = int((~valid_obs_mask).sum())
    _n_drop_pre = int(np.sum(obs_year[~valid_obs_mask] < _inv_year)) if _n_drop else 0
    print(f"  Obs kept {int(valid_obs_mask.sum())}/{len(obs_year)}; dropped {_n_drop} "
          f"outside cube span ({_n_drop_pre} pre-invasion).")
    final_obs_time_idx = year_to_index(list(realized_years), obs_year[valid_obs_mask])
    
    model_metadata = {
        "Ny": Ny, "Nx": Nx,
        "land_mask": np.array(land_mask).astype(int),
        "land_rows": np.array(land_rows), "land_cols": np.array(land_cols),
        "time": Time, "years": realized_years,
        "M": M, "K": K, "N_land": N_land,
        # The Ružička/uncentered/isotropic contract holds EXACTLY for the raw-Z (local)
        # block: Z.Z^T ~= uncentered Ružička => an isotropic coefficient prior induces a GP
        # with the Ružička kernel. It is propagated over the Z_disp dispersal block too, but
        # z_disp = A.Z is a smoothed convolution, so z_disp.z_disp^T ~= A.K.A^T (a smoothed
        # kernel), not Ružička -- the identity is only approximate there. See age_fields.py.
        "z_kernel_contract": {
            "kernel": KERNEL_CONTRACT.get("kernel", "ruzicka"),
            "centered": bool(KERNEL_CONTRACT.get("centered", False)),
            "feature_prior": KERNEL_CONTRACT.get("feature_prior", "isotropic"),
            "latent_dim": M,
            "source_latent_dim": available_M,
            "truncation": "top_eigenfeatures" if M < available_M else "none",
            "disp_kernel_note": "exact for raw-Z (local); z_disp=A.Z is a smoothed A.K.A^T",
        },
        # Disease term: two smooth land-centered spatial bases, the exogenous
        # arrival map (timestep units) and its decade-scaled version, and the
        # window start before which the term is identically zero.
        "disease_sev_basis": disease_sev_basis,
        "disease_lag_basis": disease_lag_basis,
        "N_sev_basis": int(disease_sev_basis.shape[0]),
        "N_lag_basis": int(disease_lag_basis.shape[0]),
        "disease_timestep": dis_timestep,
        "disease_onset": disease_onset,
        "disease_onset_decades": disease_onset_decades,
        "disease_arrival_map": DISEASE_ARRIVAL_MAP,
        "disease_prior_spec": DISEASE_PRIOR_SPEC,
        "k_trend_basis": k_trend_basis,
        "k_trend_spec": K_TREND_SPEC,
        "ingest_id": ingest_id,
        "z_gathered_path": z_gather_name, "z_disp_gathered_path": z_disp_name,
        "adult_fft_kernel": np.array(sim_struct['adult_fft_kernel']),
        "juvenile_fft_kernel_stack": np.array(sim_struct['juvenile_fft_kernel_stack']),
        "adult_edge_correction": np.array(sim_struct['adult_edge_correction']),
        "juvenile_edge_correction_stack": np.array(sim_struct['juvenile_edge_correction_stack']),
        "juvenile_kernel_labels": expected_labels,
        "dispersal_spec": DISPERSAL_SPEC,
        "path_feature_meta": path_meta,
        "age_structure_prior": dict(_cfg["age_structure_prior"]),
        "population_model_spec": POPULATION_SPEC,
        "obs_time_indices": np.array(final_obs_time_idx),
        "obs_rows": np.array(obs_rows[valid_obs_mask]),
        "obs_cols": np.array(obs_cols[valid_obs_mask]),
        "observed_results": np.array(observed_results[valid_obs_mask]),
        "obs_quality": np.array(obs_quality[valid_obs_mask]),
        "initpop_latent": initpop_map,
        # THE GAUGE (expected BBS route counts per relative density unit). Renamed
        # from ..._birds_per_relative_unit, which it never was -- a BBS count is a
        # 50-stop roadside index, not an absolute population. Old key accepted as a
        # fallback so existing metadata still loads.
        "pop_scalar": float(POPULATION_SPEC.get(
            "population_scale_route_counts_per_relative_unit",
            POPULATION_SPEC.get("population_scale_birds_per_relative_unit"))),
        "inv_location": (inv_row, inv_col),
        # The invasion pulse only. This was previously reused as the K-correction
        # basis's window start; the disease term now has its own, later window
        # (``disease_timestep`` above), so the two are no longer coupled.
        "inv_timestep": invasion_timestep(_tl, first_year=start_year_model),
        "inv_window": int(POPULATION_SPEC["invasion_window_years"]),
        "dispersal_target_fraction": float(
            POPULATION_SPEC["dispersal_target_capacity_fraction"]
        ),
    }
    
    meta_path = os.path.join(OUTPUT_DIR, "metadata.pkl")
    print(f"Saving metadata to {meta_path}...")
    tmp_meta_path = meta_path + ".tmp"
    with open(tmp_meta_path, "wb") as f:
        pickle.dump(model_metadata, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_meta_path, meta_path)
    print("Success. Data ingested to disk.")

def main():
    ingest_data()


if __name__ == "__main__":
    main()

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
from src.temporal import assert_contiguous, invasion_timestep, load_timeline, model_years, year_to_index

_cfg = load_age_model_config()
RAW_Z_DIR = _cfg["raw_z_dir"]
BBS_DATA_NPZ = _cfg["bbs_npz"]
MASK_FILE = _cfg["ocean_mask"]
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

# --- SPATIOTEMPORAL BASIS SETTINGS ---
N_FREQ_SPACE = int(POPULATION_SPEC["st_basis_space_frequencies"])
N_FREQ_TIME = int(POPULATION_SPEC["st_basis_time_frequencies"])

def generate_spatiotemporal_basis(Ny, Nx, Time, land_rows, land_cols, n_freq_space=4, n_freq_time=8):
    """
    Generates a 3D Spectral Basis (Cosine series).
    Space: n_freq_space=4 captures regional patterns.
    Time: n_freq_time=8 captures decadal cycles.

    ``Time`` here is whatever span the caller passes -- it is NOT necessarily
    the full model timeline. The K-correction basis (see age_fields.py /
    ingest_data below) is deliberately built over only the post-invasion
    window (invasion_year..end_year), both to bound VRAM (this array's size
    is O(N_basis * Time * N_land)) and because a correction meant to capture
    disease dynamics has nothing to explain before the species even arrives.
    The frequency-to-resolution mapping (e.g. "n_freq_time=20 -> ~4.3yr
    half-wavelength") is relative to whatever ``Time`` span is actually passed.
    """
    print(f"  Constructing 3D Basis: Space={n_freq_space}, Time={n_freq_time}...")
    
    # Create normalized coordinate grids [0, 1]
    t_coord = np.linspace(0, 1, Time)[:, None] # (Time, 1)
    y_coord = np.linspace(0, 1, Ny)[land_rows] # (N_land,)
    x_coord = np.linspace(0, 1, Nx)[land_cols] # (N_land,)
    
    basis_list = []
    
    for k in range(n_freq_time + 1):
        t_wave = np.cos(k * np.pi * t_coord) # (Time, 1)
        
        for i in range(n_freq_space + 1):
            for j in range(n_freq_space + 1):
                if i == 0 and j == 0 and k == 0:
                    continue # Skip the constant offset
                
                # Spatial component
                s_wave = np.cos(i * np.pi * y_coord) * np.cos(j * np.pi * x_coord) # (N_land,)
                
                # Outer product creates (Time, N_land) volume
                st_volume = (t_wave * s_wave[None, :]).astype(np.float32)
                basis_list.append(st_volume)
    
    st_basis = np.stack(basis_list, axis=0) # (N_basis, Time, N_land)
    return st_basis


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
    initpop_map = bbs_data['initpop_density'] * land_mask  # already at grid res

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

    # 5. Generate 3D Spatiotemporal Basis (K-correction only; post-invasion window
    # only, both to bound VRAM and because there is nothing for it to correct
    # before the species arrives -- see generate_spatiotemporal_basis's docstring
    # and age_fields.py's use of inv_timestep to bypass it for earlier timesteps).
    inv_timestep_for_basis = invasion_timestep(_tl, first_year=start_year_model)
    Time_basis_active = Time - inv_timestep_for_basis
    st_basis = generate_spatiotemporal_basis(Ny, Nx, Time_basis_active, land_rows, land_cols,
                                             n_freq_space=N_FREQ_SPACE,
                                             n_freq_time=N_FREQ_TIME)
    N_basis = st_basis.shape[0]
    print(f"  Basis Footprint: {st_basis.nbytes / 1e6:.2f} MB "
          f"(post-invasion window: {Time_basis_active}/{Time} years)")

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
        "st_basis": st_basis, 
        "N_basis": N_basis,
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
        "pop_scalar": float(POPULATION_SPEC["population_scale_birds_per_relative_unit"]),
        "inv_location": (inv_row, inv_col),
        # Reused as the K-correction basis's active-window start (see step 5
        # above / age_fields.py) -- both are "when does the species/its
        # disease dynamics first become relevant" by construction.
        "inv_timestep": inv_timestep_for_basis,
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

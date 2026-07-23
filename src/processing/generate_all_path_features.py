import sys
import os
import time
import argparse
import glob
import json
import hashlib
import uuid
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
from tqdm import tqdm

# --- Setup Paths ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 1. Import Kernels (shared juvenile-kernel builder: same family the forward sim disperses with)
from src.model.build_kernels import (
    dispersal_spec, make_juvenile_kernel_stack, toroidal_distance_grid,
)
from src.data.masks import read_land_mask
from src.temporal import load_timeline, model_years

# 2. Import Path Integration (Ensure resize_kernel_stack is fixed in here!)
from src.model.build_path_features import integrate_paths

# Helper Functions

def load_land_mask_and_meta(tif_path):
    """Loads TIF, returns Land Mask (1=Land, 0=Water) and Cell Size (km)."""
    land_mask, mask_meta = read_land_mask(tif_path, return_meta=True)
    with rasterio.open(tif_path) as src:
        res_x = src.res[0]
        units = src.crs.linear_units if src.crs else None
        
        # Heuristic unit conversion
        if (units and 'metre' in units.lower()) or (res_x > 100):
            print(f"  [Metadata] Detected units in Meters (Resolution: {res_x:.2f}m)")
            cell_size_km = res_x / 1000.0
        elif res_x < 10:
            print(f"  [Warning] Resolution {res_x} appears to be in Degrees.")
            print("  Approximating 1 deg ~ 111 km. PROJECTION RECOMMENDED.")
            cell_size_km = res_x * 111.0
        else:
            cell_size_km = res_x
            
    return land_mask.astype(np.float32), cell_size_km, mask_meta


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def visualize_results(Z_disp, Z_raw, labels, land_mask, output_dir, year):
    """
    Generates figures comparing Base Map vs Path Integrals.
    Layout: 
      Top: Base Feature (Local)
      Grid: Directional/Radial Integrals (Upstream)
    """
    # Z_disp: (1, Ny, Nx, M, K) -> take index 0 -> (Ny, Nx, M, K)
    # Z_raw:  (1, Ny, Nx, M)    -> take index 0 -> (Ny, Nx, M)
    
    data_disp = Z_disp[0]
    data_base = Z_raw[0]
    
    Ny, Nx, M, K = data_disp.shape
    
    # Masking: Set Ocean to NaN so it plots transparently
    land_mask_bc_disp = land_mask[:, :, None, None] 
    land_mask_bc_base = land_mask[:, :, None]
    
    data_vis_disp = np.where(land_mask_bc_disp > 0.5, data_disp, np.nan)
    data_vis_base = np.where(land_mask_bc_base > 0.5, data_base, np.nan)
    
    directions = ['NORTH', 'SOUTH', 'EAST', 'WEST']
    
    # Infer number of bins based on K (kernels)
    n_bins = K // 4
    
    num_feats_to_plot = min(M, 3)
    
    # print(f"    Generating plots for {num_feats_to_plot} features...")
    
    for f in range(num_feats_to_plot):
        # Create GridSpec: 
        # Row 0: Base Map (Large)
        # Row 1-4: Directions
        fig = plt.figure(figsize=(5*n_bins, 20), constrained_layout=True)
        gs = gridspec.GridSpec(5, n_bins, figure=fig, height_ratios=[1.5, 1, 1, 1, 1])
        
        fig.suptitle(f"Path Integrals: Year {year} | Feature Z_{f}", fontsize=20)
        
        # --- 1. Plot Base Map (Top Row) ---
        ax_base = fig.add_subplot(gs[0, :])
        base_img = data_vis_base[:, :, f]
        
        # Calculate shared color limits for THIS feature
        all_vals = np.concatenate([base_img.flatten(), data_vis_disp[:, :, f, :].flatten()])
        vmin = np.nanpercentile(all_vals, 2)
        vmax = np.nanpercentile(all_vals, 98)
        
        im_base = ax_base.imshow(base_img, origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        ax_base.set_title(f"BASE MAP (Local Z_{f})\nWhat is the environment HERE?", fontsize=14, fontweight='bold')
        ax_base.axis('off')
        plt.colorbar(im_base, ax=ax_base, orientation='vertical', shrink=0.8, label=f"Z_{f} Value")
        
        # --- 2. Plot Path Integrals (Rows 1-4) ---
        for i, direction in enumerate(directions):
            for j in range(n_bins):
                k_idx = i * n_bins + j
                if k_idx >= len(labels): continue
                
                label_text = labels[k_idx]
                ax = fig.add_subplot(gs[i+1, j])
                
                map_data = data_vis_disp[:, :, f, k_idx]
                im = ax.imshow(map_data, origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f"{direction} - Bin {j+1}\n({label_text})", fontsize=10)
                ax.axis('off')
                
        save_path = os.path.join(output_dir, f"vis_year_{year}_feature_{f}_comparison.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

def main(args):
    print(f"--- Starting Full Path Integration Pipeline ---")

    cube_meta_path = os.path.join(args.input_dir, "cube_meta.json")
    if not os.path.exists(cube_meta_path):
        raise FileNotFoundError(f"missing cube kernel contract: {cube_meta_path}")
    with open(cube_meta_path, encoding="utf-8") as fh:
        cube_meta = json.load(fh)
    if cube_meta.get("kernel") != "ruzicka" or bool(cube_meta.get("centered", True)):
        raise ValueError(f"path integration requires uncentered Ružička Z; got {cube_meta}")
    
    # 1. SETUP & FIND FILES
    # Mask at the model grid (matches the Z_latent cube it path-integrates).
    from src.config_utils import load_age_model_config
    age_cfg = load_age_model_config()
    disp_spec = dispersal_spec(age_cfg)
    tif_path = age_cfg["ocean_mask"]
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"mask file not found: {tif_path}")
    build_id = uuid.uuid4().hex

    # Find years
    if args.year == 'all':
        pattern = os.path.join(args.input_dir, "Z_latent_*.npy")
        files = glob.glob(pattern)
        years = []
        for f in files:
            # Filename format: Z_latent_1990.npy
            try:
                y = os.path.basename(f).split('_')[-1].split('.')[0]
                if y.isdigit():
                    years.append(y)
            except:
                continue
        years = sorted(years, key=int)
        print(f"Found {len(years)} years to process: {years}")
    else:
        years = [args.year]

    if not years:
        raise FileNotFoundError(f"No files found matching {args.input_dir}/Z_latent_*.npy")
    expected_years = [str(y) for y in model_years(load_timeline())]
    if args.year == "all" and years != expected_years:
        missing = sorted(set(expected_years) - set(years))
        extra = sorted(set(years) - set(expected_years))
        raise ValueError(f"cube timeline must be exactly {expected_years[0]}-{expected_years[-1]}; "
                         f"missing={missing[:10]}, extra={extra[:10]}")
    cube_years = [str(int(y)) for y in cube_meta.get("years", [])]
    if args.year == "all" and cube_years != expected_years:
        raise ValueError(f"cube_meta years do not match the canonical timeline "
                         f"{expected_years[0]}-{expected_years[-1]}")

    # 2. LOAD STATIC DATA (Geometry & Kernels) - Run Once!
    print("\n--- Initializing Geometry & Kernels (Shared) ---")
    land_mask_np, cell_size_km, mask_meta = load_land_mask_and_meta(tif_path)
    print(f"Grid: {land_mask_np.shape} | Cell: {cell_size_km:.2f} km")
    
    Ny, Nx = land_mask_np.shape
    Ly, Lx = 2*Ny-1, 2*Nx-1
    land_mask = jnp.array(land_mask_np, dtype=jnp.float32)

    splits = disp_spec["juvenile_radial_splits_km"]
    print(f"Using configured radial splits (km): {[f'{x:.1f}' for x in splits]}")
    
    # Build the juvenile kernel stack through the SAME helper the forward simulation uses,
    # so the base dispersal PDF + splits match and Q[p,k] pairs with the right flux kernel.
    kernel_stack, labels = make_juvenile_kernel_stack(
        Lx, Ly, cell_size_km, splits,
        mean_dist=disp_spec["juvenile_mdd_km"],
        shape=disp_spec["juvenile_shape"],
    )
    kernel_mass = float(jnp.sum(kernel_stack))
    r_grid = toroidal_distance_grid(Lx, Ly, cell_size_km)
    realized_mdd = float(jnp.sum(jnp.sum(kernel_stack, axis=0) * r_grid) / kernel_mass)
    print(f"Juvenile stack: mass={kernel_mass:.8f}, realized discrete MDD={realized_mdd:.2f} km")
    # Force compilation on a dummy input to avoid recompiling inside the loop
    print("Pre-compiling JAX kernels...")
    dummy_Z = jnp.zeros((1, Ny, Nx, 1)) # Small dummy
    integrate_paths(dummy_Z, kernel_stack, land_mask, steps=2).block_until_ready()
    print("Compilation complete.\n")

    # 3. PROCESSING LOOP
    # The contract (uncentered Ružička, isotropic prior) is copied through verbatim from the
    # cube. It describes the RAW Z (Z_raw), where Z.Z^T ~= Ružička exactly. The Z_disp features
    # written below are a land-normalized spatial convolution A.Z, so z_disp.z_disp^T ~= A.K.A^T
    # (a smoothed kernel) -- the Ružička identity is only approximate on the dispersal block.
    # Downstream (age_fields.py) reuses beta_s there by design. See disp_kernel_note.
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_start = time.time()
    
    for year in tqdm(years, desc="Processing Years"):
        z_filename = f"Z_latent_{year}.npy"
        z_path = os.path.join(args.input_dir, z_filename)
        out_name = f"Z_disp_{year}.npz"
        save_path = os.path.join(args.output_dir, out_name)
        
        # Check if already exists (optional skip)
        # if os.path.exists(save_path): continue 

        # A. Load Z
        Z_year = jnp.load(z_path)
        if Z_year.ndim == 3: Z_year = Z_year[None, ...]
        if tuple(Z_year.shape[1:3]) != (Ny, Nx):
            raise ValueError(f"{z_path} grid {tuple(Z_year.shape[1:3])} != mask grid {(Ny, Nx)}")
        if int(Z_year.shape[-1]) != int(cube_meta["latent_dim"]):
            raise ValueError(f"{z_path} width {Z_year.shape[-1]} != cube contract "
                             f"{cube_meta['latent_dim']}")
        
        # B. Integrate
        # Note: kernel_stack and land_mask are reused from memory!
        if not bool(jnp.isfinite(Z_year[0][land_mask_np > 0.5]).all()):
            raise ValueError(f"{z_path} contains non-finite values on land")
        Z_disp = integrate_paths(Z_year, kernel_stack, land_mask, steps=args.steps)
        Z_disp.block_until_ready()
        
        # C. Save
        tmp_path = save_path + ".tmp"
        with open(tmp_path, "wb") as fh:
            np.savez_compressed(
                fh,
                Z_disp=np.asarray(Z_disp),
                Z_raw=np.asarray(Z_year),
                cell_size_km=cell_size_km,
                labels=labels,
                land_mask=land_mask_np,
                build_id=build_id,
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, save_path)
        
        # D. Visualize (First 3 features)
        if args.viz:
            visualize_results(Z_disp, Z_year, labels, land_mask_np, args.output_dir, year)

    path_meta = {
        "schema_version": 1,
        "build_id": build_id,
        "kernel_contract": cube_meta,
        "years": [int(y) for y in years],
        "grid_shape": [Ny, Nx],
        "cell_size_km": cell_size_km,
        "mask_path": tif_path,
        "mask_sha256": _sha256(tif_path),
        "mask_semantics": "0=land,1=ocean",
        "mask_crs": mask_meta["crs"],
        "mask_transform": list(mask_meta["transform"]),
        "source_latent_dim": int(cube_meta["latent_dim"]),
        "kernel_labels": labels,
        "kernel_count": len(labels),
        "dispersal": disp_spec,
        "integration_steps": int(args.steps),
        "kernel_mass": kernel_mass,
        "realized_discrete_mdd_km": realized_mdd,
    }
    meta_path = os.path.join(args.output_dir, "path_feature_meta.json")
    tmp_meta_path = meta_path + ".tmp"
    with open(tmp_meta_path, "w", encoding="utf-8") as fh:
        json.dump(path_meta, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_meta_path, meta_path)

    print(f"\nAll done! Total time: {(time.time() - total_start)/60:.2f} minutes.")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    from src.config_utils import load_age_model_config
    _pf = load_age_model_config()["path_features"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default="all", help="Year to process (e.g. '1990' or 'all')")
    parser.add_argument("--input_dir", type=str, default=_pf["input_dir"])
    parser.add_argument("--output_dir", type=str, default=_pf["output_dir"])
    parser.add_argument("--steps", type=int, default=None, help="Integration steps (default: config)")
    parser.add_argument("--viz", action="store_true", help="Generate PNG visualizations")
    
    args = parser.parse_args()
    if args.steps is None:
        args.steps = dispersal_spec(load_age_model_config())["path_integration_steps"]
    main(args)

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import jax
import jax.numpy as jnp
from jax.numpy.fft import fft2, ifft2
from scipy.special import gammainc  # Needed for theoretical mass calculation

# --- Setup Paths ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config_utils import load_age_model_config, load_data_config
_CFG = load_data_config()
_DR = _CFG["datasets_root"]
_PR = _CFG["processed_root"]
_RES_KM = _CFG["grid"]["target_res_m"] // 1000

from src.data.masks import read_land_mask
from src.model.build_kernels import build_simulation_struct, dispersal_spec, get_gamma_scale

def print_mass_distribution(splits, mean_dist, shape):
    """
    Calculates and prints the theoretical probability mass in each radial bin
    under the Generalized Gamma dispersal kernel.
    """
    # 1. Calculate Scale Parameter (Lambda)
    scale = get_gamma_scale(mean_dist, shape)
    
    print("\n--- Theoretical Mass Distribution ---")
    print(f"Kernel: GenGamma(Mean={mean_dist:.1f}km, Shape={shape:.3f})")
    print(f"Scale Parameter (Lambda): {scale:.3f} km")
    print("-" * 65)
    print(f"{'Bin':<5} | {'Range (km)':<20} | {'Mass Fraction':<15} | {'Cum. Mass':<10}")
    print("-" * 65)
    
    cdf_prev = 0.0
    
    for i in range(len(splits) - 1):
        r_min = splits[i]
        r_max = splits[i+1]
        
        # Calculate CDF at r_max
        # Formula: P(R < r) = gammainc(2/shape, (r/scale)^shape)
        if r_max >= 1e9:
            cdf_curr = 1.0
            range_str = f"{r_min:.1f} - Inf"
        else:
            # Argument for Regularized Gamma Function
            z = (r_max / scale) ** shape
            cdf_curr = gammainc(2.0 / shape, z)
            range_str = f"{r_min:.1f} - {r_max:.1f}"
            
        mass = cdf_curr - cdf_prev
        
        print(f"{i+1:<5} | {range_str:<20} | {mass:.4%}      | {cdf_curr:.4f}")
        
        cdf_prev = cdf_curr
    print("-" * 65 + "\n")

def load_land_metadata(tif_path):
    with rasterio.open(tif_path) as src:
        res_x = src.res[0]
        units = src.crs.linear_units if src.crs else None
        
        if (units and 'metre' in units.lower()) or (res_x > 100):
            cell_size_km = res_x / 1000.0
        elif res_x < 10:
            print("Warning: Units appear to be degrees. Approximating 111km/deg.")
            cell_size_km = res_x * 111.0
        else:
            cell_size_km = res_x
            
        land_mask = read_land_mask(tif_path).astype(np.float32)
        
    return land_mask, cell_size_km

def save_grid_plot(data_stack, labels, title, output_path):
    """Plots a grid of maps with PER-IMAGE normalization."""
    n_kernels = len(data_stack)
    rows = 4 
    cols = n_kernels // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    
    direction_names = ["NORTH", "SOUTH", "EAST", "WEST"]
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx >= n_kernels: continue
            
            ax = axes[i, j]
            img = data_stack[idx]
            lbl = labels[idx]
            
            if "Kernel" in title:
                img = np.fft.fftshift(img)
            
            local_max = np.nanmax(img)
            if local_max == 0: local_max = 1.0
            
            im = ax.imshow(img, origin='upper', cmap='plasma', vmin=0, vmax=local_max)
            ax.set_title(f"{direction_names[i]} - Bin {j+1}\n{lbl}", fontsize=9)
            ax.axis('off')
            
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")

def main(args):
    print(f"--- Testing Kernel Physics ---")
    
    # 1. Load Geometry
    age_cfg = load_age_model_config()
    spec = dispersal_spec(age_cfg)
    tif_path = age_cfg["ocean_mask"]
    if not os.path.exists(tif_path):
        print(f"Error: Missing {tif_path}")
        return
        
    land_mask_np, cell_size_km = load_land_metadata(tif_path)
    Ny, Nx = land_mask_np.shape
    print(f"Grid: {Ny}x{Nx} | Cell Size: {cell_size_km:.2f} km")
    
    land_mask = jnp.array(land_mask_np)

    splits = spec["juvenile_radial_splits_km"]
    
    # --- DIAGNOSTIC ADDED HERE ---
    print_mass_distribution(
        splits, spec["juvenile_mdd_km"], spec["juvenile_shape"]
    )
    
    # 3. Build Kernels
    print("Building Simulation Struct...")
    sim_data = build_simulation_struct(
        land=land_mask,
        cell_size=cell_size_km,
        adult_mdd=spec["adult_mdd_km"],
        juvenile_mdd=spec["juvenile_mdd_km"],
        adult_shape=spec["adult_shape"],
        juvenile_shape=spec["juvenile_shape"],
        radii_splits=splits
    )
    
    kernel_stack_fft = sim_data["juvenile_fft_kernel_stack"]
    labels = sim_data["labels"]
    edge_corrections = sim_data["juvenile_edge_correction_stack"]
    _, Ly, Lx = kernel_stack_fft.shape
    
    # 4. Visualize Raw Kernels
    print("\nVisualizing Raw Kernels...")
    raw_kernels = jnp.real(ifft2(kernel_stack_fft, axes=(-2, -1)))
    
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    save_grid_plot(
        raw_kernels, labels, 
        "Raw Kernels (FFT Shifted Center)", 
        os.path.join(out_dir, "diagnostics_kernels_raw.png")
    )
    
    # 5. Run Spike Test
    print("\nRunning Spike Simulation...")
    pop_before = jnp.zeros((Ny, Nx))
    cy, cx = Ny // 2, Nx // 2
    
    if land_mask[cy, cx] == 0:
        print("Warning: Center pixel is Water! Moving spike to nearest Land.")
        y_land, x_land = jnp.where(land_mask > 0.5)
        idx = len(y_land) // 2
        cy, cx = y_land[idx], x_land[idx]
        
    pop_before = pop_before.at[cy, cx].set(1.0)
    print(f"Spike location: ({cy}, {cx})")
    
    pop_results = []
    
    for k in range(len(labels)):
        # Forward-model convention: divide each source cohort by the fraction
        # of its normalized cohort shape that can land on valid habitat.
        pop_w_edge = pop_before / (edge_corrections[k] + 1e-6)
        pop_padded = jnp.pad(pop_w_edge, ((0, Ly - Ny), (0, Lx - Nx)))
        
        fft_pop = fft2(pop_padded) 
        conv_fft = fft_pop * kernel_stack_fft[k] 
        
        pop_after_full = jnp.real(ifft2(conv_fft))
        pop_after = pop_after_full[:Ny, :Nx]
        
        # Calculate Stats (ON LAND ONLY)
        pop_on_land = pop_after * land_mask
        mass = jnp.sum(pop_on_land)
        
        y_idx, x_idx = jnp.meshgrid(jnp.arange(Ny), jnp.arange(Nx), indexing="ij")
        dy = (y_idx - cy) * cell_size_km
        dx = (x_idx - cx) * cell_size_km
        dist_grid = jnp.sqrt(dy**2 + dx**2)
        
        mean_dist = jnp.sum(pop_on_land * dist_grid) / jnp.maximum(mass, 1e-9)
        
        pop_results.append(pop_on_land)
        
        print(f"  Kernel {labels[k]:<25} | Land Mass: {mass:.4f} | Mean Disp: {mean_dist:.2f} km")

    pop_stack = jnp.stack(pop_results, axis=0)
    
    save_grid_plot(
        pop_stack, labels, 
        "Population After Dispersal (Land Only)", 
        os.path.join(out_dir, "diagnostics_spike_response.png")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=f"{_DR}/latent_avian_community_similarities")
    parser.add_argument("--output_dir", type=str, default=f"{_PR}/datasets/latent_avian_path_diagnostics")
    
    args = parser.parse_args()
    main(args)

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import warnings
import glob

# --- Setup Paths ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.data_loading import load_data_to_gpu
from src.config_utils import load_age_model_config
from src.vis._age_vis_common import (
    plot_posterior_weights,
    plot_demographic_response_curves,
    plot_spatial_uncertainty,
    plot_demographic_baselines,
    create_animation,
    scatter_to_grid_robust,
    scatter_observations_to_grid,
    analyze_source_sink_mortality,
)

# --- CONFIGURATION ---
PRECISION = 'float32'
_cfg = load_age_model_config()
INPUT_DIR = _cfg["input_dir"]
RESULT_DIR = os.path.join(_cfg["results_dir"], _cfg["run_names"]["vis_svi"].format(precision=PRECISION))
OUTPUT_PLOT_DIR = os.path.join(RESULT_DIR, "plots_analysis")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

def plot_comprehensive_age_structure(Na_grid_mean, Nj_grid_mean, years,
                                     Sa_grid_mean, Sj_grid_mean, Fmax_grid_mean, 
                                     land_mask, output_dir, recent_years_window=10):
    """
    Generates both the trajectory trend plot and the recent 2D spatial manifold map
    for the theoretical age structure at carrying capacity, processing full 3D tensors.
    """
    print("Processing Comprehensive Age Structure Diagnostics (Trajectories & Spatial Manifolds)...")
    
    # 1. COMPUTE TRAJECTORY METRICS (Realized vs Theoretical over time)
    # Realized metrics from actual simulated bird allocations
    total_Na = np.sum(Na_grid_mean, axis=(1, 2))
    total_Nj = np.sum(Nj_grid_mean, axis=(1, 2))
    total_pop = total_Na + total_Nj + 1e-9
    global_rho_realized = total_Nj / total_pop

    T_sliced = Sa_grid_mean.shape[0]
    years_sliced = years[-T_sliced:]
    global_rho_theoretical_at_K = np.zeros(T_sliced)
    
    # Pre-allocate a 3D matrix to hold full 2D rho_K maps for every timestep
    Ny, Nx = land_mask.shape
    all_rho_K_maps = np.zeros((T_sliced, Ny, Nx))

    for t in range(T_sliced):
        # Extract full 2D arrays for this timestep
        Sa_t = Sa_grid_mean[t]
        Sj_t = Sj_grid_mean[t]
        Fmax_t = Fmax_grid_mean[t]
        
        # Reconstruct dynamic density-dependence scaling (using 1e-6 to prevent 0/0)
        c_dynamic_t = (Fmax_t * Sj_t) / (1.0 - Sa_t + 1e-6) - 1.0
        c_dynamic_t = np.maximum(c_dynamic_t, 0.0)
        
        F_eff_K_t = Fmax_t / (1.0 + c_dynamic_t)
        lambda_K_t = (Sa_t + np.sqrt(Sa_t**2 + 4.0 * F_eff_K_t * Sj_t)) / 2.0
        
        # Safeguard the denominator against unlivable 0/0 channels
        denominator_t = F_eff_K_t + lambda_K_t
        rho_K_t = np.where(denominator_t > 0, F_eff_K_t / (denominator_t + 1e-9), np.nan)
        
        # Mask to land-only to calculate an accurate continental average for this year
        global_rho_theoretical_at_K[t] = np.nanmean(rho_K_t[land_mask])
        
        # Store the full 2D map slice
        all_rho_K_maps[t, :, :] = rho_t_fixed = np.where(land_mask, rho_K_t, np.nan)

    # PLOT 1: Trajectory Comparison Trendline
    plt.figure(figsize=(10, 6))
    plt.plot(years, global_rho_realized, color='forestgreen', linewidth=2, 
             label="Realized (Actual Population)")
    plt.plot(years_sliced, global_rho_theoretical_at_K, color='navy', linestyle=':', linewidth=2, 
             label="Theoretical Potential (Spatially-Averaged at K)")
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5, label="Prior Anchor (Ideal Habitat)")
    
    plt.title("Continental Juvenile Fraction: Realized vs. Theoretical Potential")
    plt.xlabel("Year")
    plt.ylabel("Proportion of Population")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "trend_age_structure_comparison.png"), dpi=200)
    plt.close()

    # 2. COMPUTE SPATIAL MANIFOLD (Temporal average of recent years)
    window_clamped = min(recent_years_window, T_sliced)
    rho_K_recent_slices = all_rho_K_maps[-window_clamped:, :, :]
    
    # Take the temporal average across the 3D stack, preserving 2D spatial dimensions
    rho_K_map = np.nanmean(rho_K_recent_slices, axis=0)

    # PLOT 2: 2D Geographic Heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(rho_K_map, cmap='RdYlBu', vmin=0.3, vmax=0.7)
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(r"Theoretical Juvenile Fraction ($\rho_K$)", rotation=270, labelpad=15)
    cbar.ax.axhline(0.5, color='black', linestyle='--', linewidth=1.5)
    
    plt.title(f"Theoretical Age Structure Potential at Carrying Capacity\n"
              f"(Spatial Manifold Averaged Over Past {window_clamped} Years)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "map_theoretical_age_structure.png"), dpi=300)
    plt.close()
    
    print("Diagnostics successfully rendered and saved to output directory.")

# --- MAIN CONTROLLER ---
def plot_results():
    data = load_data_to_gpu(INPUT_DIR, precision=PRECISION)

    sample_path = os.path.join(OUTPUT_PLOT_DIR, "reconstructed_samples_stats.npz")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"SVI summary stats not found at {sample_path}.")
        
    print(f"Loading SVI statistics from {sample_path}...")
    with np.load(sample_path) as loader:
        spatial_stats = {k: loader[k] for k in loader.files if k.endswith('_mean') or k.endswith('_var')}
        raw_samples = {k: loader[k] for k in loader.files if not (k.endswith('_mean') or k.endswith('_var'))}

   # 1. Load Directional Dispersal Kernel Labels (K = 12)
    PATH_INTEGRATION_DIR = _cfg["path_features"]["output_dir"]
    disp_files = glob.glob(os.path.join(PATH_INTEGRATION_DIR, "Z_disp_*.npz"))
    
    if not disp_files:
        print("Warning: No Z_disp files found. Labels will be generic.")
        kernel_labels = [f"Kernel_{i}" for i in range(data['juvenile_fft_kernel_stack'].shape[0])] 
    else:
        with np.load(disp_files[0]) as loader:
            kernel_labels = [str(lbl) for lbl in loader['labels']]
            
    # 2. Generate Environmental Feature Labels (M = 16)
    M = data['Z_gathered'].shape[-1]
    env_labels = [f"Env_PC_{i+1}" for i in range(M)]

    Ny, Nx = data['Ny'], data['Nx']
    rows, cols = data['land_rows'], data['land_cols']
    years = data['years']
    start_idx = np.where(years >= 1960)[0][0]
    
    # 1. Re-scatter pre-aggregated means
    Sa_grid_mean = scatter_to_grid_robust(spatial_stats['Sa_flat_mean'], rows, cols, (Ny, Nx))
    Sj_grid_mean = scatter_to_grid_robust(spatial_stats['Sj_flat_mean'], rows, cols, (Ny, Nx))
    Fmax_grid_mean = scatter_to_grid_robust(spatial_stats['Fmax_flat_mean'], rows, cols, (Ny, Nx))
    K_grid_mean = scatter_to_grid_robust(spatial_stats['K_flat_mean'], rows, cols, (Ny, Nx))
    
    # 2. Re-scatter exact pre-aggregated variances
    Sa_grid_var = scatter_to_grid_robust(spatial_stats['Sa_flat_var'], rows, cols, (Ny, Nx))
    Sj_grid_var = scatter_to_grid_robust(spatial_stats['Sj_flat_var'], rows, cols, (Ny, Nx))
    Fmax_grid_var = scatter_to_grid_robust(spatial_stats['Fmax_flat_var'], rows, cols, (Ny, Nx))
    K_grid_var = scatter_to_grid_robust(spatial_stats['K_flat_var'], rows, cols, (Ny, Nx))
    
    # 3. State variable means (No rebuilding needed - avoiding Jensen's Inequality)
    density_mean = spatial_stats['simulated_density_mean'] * data['pop_scalar']
    Na_grid_mean = scatter_to_grid_robust(spatial_stats['Na_flat_mean'], rows, cols, (Ny, Nx))
    Nj_grid_mean = scatter_to_grid_robust(spatial_stats['Nj_flat_mean'], rows, cols, (Ny, Nx))
    
    obs_grid = scatter_observations_to_grid(data['observed_results'], data['obs_time_indices'], data['obs_rows'], data['obs_cols'], (Ny, Nx), data['time'])
    source_prob_mean = scatter_to_grid_robust(spatial_stats['source_probability_mean'], rows, cols, (Ny, Nx))
    
    # --- Execute Diagnostic Suite ---
    
    # These plot environmental features (M=16) -> Pass env_labels
    plot_posterior_weights(raw_samples, M, env_labels, OUTPUT_PLOT_DIR, label="SVI", fname="1_posterior_weights_vi.png")
    
    for target_idx in range(min(3, M)):
        plot_demographic_response_curves(raw_samples, M, env_labels, target_idx, OUTPUT_PLOT_DIR)
    
    # Trigger the new analysis
    analyze_source_sink_mortality(
        data, 
        Sa_grid_mean[start_idx:], 
        Sj_grid_mean[start_idx:], 
        Fmax_grid_mean[start_idx:], 
        source_prob_mean[start_idx:], 
        OUTPUT_PLOT_DIR
    )
    plot_demographic_baselines(Sa_grid_mean[start_idx:], Sj_grid_mean[start_idx:], Fmax_grid_mean[start_idx:], K_grid_mean[start_idx:], OUTPUT_PLOT_DIR, data['land_mask'], data['pop_scalar'])
    plot_spatial_uncertainty(Sa_grid_var[start_idx:], Sj_grid_var[start_idx:], Fmax_grid_var[start_idx:], K_grid_var[start_idx:], OUTPUT_PLOT_DIR, data['land_mask'])
    plot_comprehensive_age_structure(
        Na_grid_mean, 
        Nj_grid_mean, 
        years, 
        Sa_grid_mean[start_idx:],  # New argument
        Sj_grid_mean[start_idx:],  # New argument
        Fmax_grid_mean[start_idx:],# New argument
        data['land_mask'],
        OUTPUT_PLOT_DIR, 
        recent_years_window=10
    )
    create_animation(density_mean, obs_grid, years, OUTPUT_PLOT_DIR, data['land_mask'])

    # This analyzes directional path integrations (K=12) -> Pass kernel_labels
    # plot_directional_asymmetry(data, Sa_grid_mean[start_idx:], Sj_grid_mean[start_idx:], Fmax_grid_mean[start_idx:], Q_grid_mean[start_idx:], kernel_labels, OUTPUT_PLOT_DIR)
    print(f"All probabilistic diagnostics complete. Results at: {OUTPUT_PLOT_DIR}")

if __name__ == "__main__":
    plot_results()
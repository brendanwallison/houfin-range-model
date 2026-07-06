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

from src.model.age_run_map import load_data_to_gpu

# --- CONFIGURATION ---
PRECISION = 'float32' 
INPUT_DIR = "/home/breallis/processed_data/model_inputs/numpyro_input"
RESULT_DIR = f"/home/breallis/processed_data/model_results/age_vi_{PRECISION}_run_5"
OUTPUT_PLOT_DIR = os.path.join(RESULT_DIR, "plots_analysis")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# --- 1. DIRECT POSTERIORS (RAW SAMPLES) ---
def plot_posterior_weights(raw_samples, M, z_names, output_dir):
    """Plots SVI estimates for beta_s and beta_r with 90% Credible Intervals."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bs_samples = raw_samples['w_env'][..., 0]
    br_samples = raw_samples['w_env'][..., 1]
    
    bs_mean, br_mean = np.mean(bs_samples, axis=0), np.mean(br_samples, axis=0)
    
    bs_err = np.vstack([bs_mean - np.percentile(bs_samples, 5, axis=0), 
                        np.percentile(bs_samples, 95, axis=0) - bs_mean])
    br_err = np.vstack([br_mean - np.percentile(br_samples, 5, axis=0), 
                        np.percentile(br_samples, 95, axis=0) - br_mean])
    
    y_pos = np.arange(M)
    for i in range(M):
        ax.plot([bs_mean[i], br_mean[i]], [y_pos[i], y_pos[i]], color='gray', linestyle=':', alpha=0.7, zorder=1)
    
    ax.errorbar(bs_mean, y_pos, xerr=bs_err, fmt='o', color='dodgerblue', label=r'Survival ($\beta_s$)', markersize=8, capsize=3, zorder=3)
    ax.errorbar(br_mean, y_pos, xerr=br_err, fmt='s', color='darkorange', label=r'Reproduction ($\beta_r$)', markersize=8, capsize=3, zorder=2)
    
    ax.axvline(0, color='black', linestyle='-', alpha=0.3, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(z_names)
    ax.set_xlabel("Learned Weight (90% CI)")
    ax.set_title("Environmental Profile: Survival vs. Reproduction (SVI Posterior)")
    ax.set_ylim(-0.5, M - 0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_posterior_weights_vi.png"), dpi=300)
    plt.close()

def plot_demographic_response_curves(raw_samples, M, z_names, target_z_idx, output_dir):
    """Sweeps a target Z axis to show the non-linear biological response with 90% HDI."""
    z_sweep = np.linspace(-3, 3, 100)
    Z_matrix = np.zeros((100, M))
    Z_matrix[:, target_z_idx] = z_sweep
    
    # Sigmoid and Softplus transformations in pure numpy
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    def softplus(x): return np.log1p(np.exp(x))
    
    alpha_a = raw_samples['alpha_a'][:, None]
    alpha_j = raw_samples['alpha_j'][:, None]
    alpha_f = raw_samples['alpha_f'][:, None]
    
    gamma_a = softplus(raw_samples['gamma_a_raw'])[:, None]
    gamma_j = gamma_a + raw_samples['gamma_j_diff'][:, None]
    gamma_f = softplus(raw_samples['gamma_f_raw'])[:, None]
    
    H_s = np.dot(Z_matrix, raw_samples['w_env'][..., 0].T).T
    H_r = np.dot(Z_matrix, raw_samples['w_env'][..., 1].T).T
    
    S_a_curves = sigmoid(alpha_a + gamma_a * H_s)
    S_j_curves = sigmoid(alpha_j + gamma_j * H_s)
    F_max_curves = np.exp(alpha_f + gamma_f * H_r)
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    def plot_band(ax, x, curves, color, label, linestyle='-'):
        mean_c = np.mean(curves, axis=0)
        low_c = np.percentile(curves, 5, axis=0)
        high_c = np.percentile(curves, 95, axis=0)
        ax.plot(x, mean_c, color=color, linewidth=2, linestyle=linestyle, label=label)
        ax.fill_between(x, low_c, high_c, color=color, alpha=0.2, edgecolor='none')

    plot_band(ax1, z_sweep, S_a_curves, 'navy', r'Adult Survival ($S_a$)')
    plot_band(ax1, z_sweep, S_j_curves, 'royalblue', r'Juvenile Survival ($S_j$)', '--')
    
    ax1.set_xlabel(f"{z_names[target_z_idx]} Gradient (Standardized)")
    ax1.set_ylabel("Annual Survival Probability", color='navy')
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='y', labelcolor='navy')
    
    ax2 = ax1.twinx()
    plot_band(ax2, z_sweep, F_max_curves, 'darkorange', r'Max Fecundity ($F_{max}$)')
    ax2.set_ylabel("Maximum Fecundity", color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    plt.title(f"Demographic Response to {z_names[target_z_idx]} (90% CI)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"2_response_curve_{z_names[target_z_idx]}.png"), dpi=300)
    plt.close()

# --- 2. SPATIAL UNCERTAINTY MAPPING (EXACT VARIANCE) ---
def plot_spatial_uncertainty(Sa_grid_var, Sj_grid_var, Fmax_grid_var, K_grid_var, output_dir, land_mask):
    print("Generating Spatial Uncertainty Maps (Standard Deviations)...")
    
    # Convert exact running variance into standard deviations
    Sa_std = np.sqrt(np.mean(Sa_grid_var, axis=0))
    Sj_std = np.sqrt(np.mean(Sj_grid_var, axis=0))
    Fmax_std = np.sqrt(np.mean(Fmax_grid_var, axis=0))
    K_std = np.sqrt(np.mean(K_grid_var, axis=0))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plots = [
        (Sa_std, r"Uncertainty: Adult Survival ($\sigma_{S_a}$)", "magma", axes[0, 0]),
        (Sj_std, r"Uncertainty: Juvenile Survival ($\sigma_{S_j}$)", "magma", axes[0, 1]),
        (Fmax_std, r"Uncertainty: Max Fecundity ($\sigma_{F_{max}}$)", "inferno", axes[1, 0]),
        (K_std, r"Uncertainty: Carrying Capacity ($\sigma_K$)", "cividis", axes[1, 1])
    ]

    for grid, title, cmap, ax in plots:
        masked_grid = np.ma.masked_where(land_mask == 0, grid)
        im = ax.imshow(masked_grid, cmap=cmap, origin='upper')
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label="Standard Deviation", fraction=0.046, pad=0.04)

    plt.suptitle("Spatiotemporal Environmental Uncertainty (Model Confidence)", fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "analysis_spatial_uncertainty.png"), dpi=200)
    plt.close()

def plot_demographic_baselines(Sa_grid_mean, Sj_grid_mean, Fmax_grid_mean, K_grid_mean, output_dir, land_mask, scalar):
    print("Generating Age-Structured Demographic Baselines (Means)...")
    Sa_avg = np.mean(Sa_grid_mean, axis=0)
    Sj_avg = np.mean(Sj_grid_mean, axis=0)
    Fmax_avg = np.mean(Fmax_grid_mean, axis=0)
    K_avg = np.mean(K_grid_mean, axis=0) * scalar
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plots = [
        (Sa_avg, r"Adult Survival ($S_a$)", "plasma", axes[0, 0], (0, 1)),
        (Sj_avg, r"Juvenile Survival ($S_j$)", "plasma", axes[0, 1], (0, 1)),
        (Fmax_avg, r"Maximum Fecundity ($F_{\max}$)", "viridis", axes[1, 0], None),
        (K_avg, r"Carrying Capacity ($K$)", "cividis", axes[1, 1], 'log')
    ]

    for grid, title, cmap, ax, scale in plots:
        masked_grid = np.ma.masked_where(land_mask == 0, grid)
        if scale == 'log':
            im = ax.imshow(np.log1p(masked_grid), cmap=cmap, origin='upper')
            cbar_label = "Log(1 + Expected Birds)"
        elif isinstance(scale, tuple):
            im = ax.imshow(masked_grid, cmap=cmap, origin='upper', vmin=scale[0], vmax=scale[1])
            cbar_label = "Probability"
        else:
            im = ax.imshow(masked_grid, cmap=cmap, origin='upper')
            cbar_label = "Offspring per Adult"

        ax.set_title(title, fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

    plt.suptitle("Age-Structured Demographic Manifold (Expected Mean)", fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "analysis_demographic_baselines.png"), dpi=200)
    plt.close()

def plot_comprehensive_age_structure(Na_grid_mean, Nj_grid_mean, years, 
                                     Sa_grid_mean, Sj_grid_mean, Fmax_grid_mean, 
                                     land_mask, output_dir, recent_years_window=10):
    """
    Generates both the trajectory trend plot and the recent 2D spatial manifold map
    for the theoretical age structure at carrying capacity, processing full 3D tensors.
    """
    print("Processing Comprehensive Age Structure Diagnostics (Trajectories & Spatial Manifolds)...")
    
    # -------------------------------------------------------------------------
    # 1. COMPUTE TRAJECTORY METRICS (Realized vs Theoretical over time)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # PLOT 1: Trajectory Comparison Trendline
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2. COMPUTE SPATIAL MANIFOLD (Temporal average of recent years)
    # -------------------------------------------------------------------------
    window_clamped = min(recent_years_window, T_sliced)
    rho_K_recent_slices = all_rho_K_maps[-window_clamped:, :, :]
    
    # Take the temporal average across the 3D stack, preserving 2D spatial dimensions
    rho_K_map = np.nanmean(rho_K_recent_slices, axis=0)

    # -------------------------------------------------------------------------
    # PLOT 2: 2D Geographic Heatmap
    # -------------------------------------------------------------------------
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

def create_animation(density, obs_grid, years, output_dir, land_mask, filename="evolution_history.mp4", logscale=False):
    print(f"Generating Expected Density Animation: {filename}...")
    fig, (ax_sim, ax_obs) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(top=0.85)
    
    vmax_val = np.nanpercentile(density, 99)
    norm = mcolors.LogNorm(vmin=1e-6, vmax=vmax_val) if logscale else mcolors.Normalize(vmin=0, vmax=vmax_val)
    im_sim = ax_sim.imshow(density[0], cmap='magma', origin='upper', norm=norm)
    
    land_bg = np.zeros_like(land_mask, dtype=float)
    land_bg[land_mask == 1] = 0.2
    ax_obs.imshow(land_bg, cmap='gray_r', vmin=0, vmax=1, origin='upper', alpha=0.3)
    im_obs = ax_obs.imshow(obs_grid[0], cmap='magma', origin='upper', norm=norm, interpolation='none')
    
    title = fig.suptitle(f"Year: {years[0]}", fontsize=16, fontweight='bold')

    def update(frame):
        title.set_text(f"Year: {years[frame]}")
        sim_data = np.clip(density[frame], 1e-8, None) if logscale else density[frame]
        obs_data = np.clip(obs_grid[frame], 1e-8, None) if logscale else obs_grid[frame]
        im_sim.set_data(sim_data)
        im_obs.set_data(obs_data)
        return im_sim, im_obs, title

    ani = animation.FuncAnimation(fig, update, frames=len(years), interval=100, blit=False)
    save_path = os.path.join(output_dir, filename)
    try:
        writer = animation.FFMpegWriter(fps=10, bitrate=1800)
        ani.save(save_path, writer=writer)
    except Exception:
        print(f"FFMpegWriter failed. Saving as GIF instead: {save_path.replace('.mp4', '.gif')}")
        ani.save(save_path.replace(".mp4", ".gif"), writer='pillow', fps=10)
    plt.close()

# --- UTILITIES ---
def scatter_to_grid_robust(flat_array, rows, cols, shape):
    Ny, Nx = shape
    if flat_array is None: return None
    if flat_array.ndim == 1:
        grid = np.zeros((Ny, Nx)); grid[rows, cols] = flat_array; return grid
    elif flat_array.ndim == 2:
        if flat_array.shape[0] == len(rows):
            K = flat_array.shape[1]; grid = np.zeros((Ny, Nx, K)); grid[rows, cols, :] = flat_array
        else:
            T = flat_array.shape[0]; grid = np.zeros((T, Ny, Nx)); grid[:, rows, cols] = flat_array
        return grid
    elif flat_array.ndim == 3:
        T, _, K = flat_array.shape; grid = np.zeros((T, Ny, Nx, K)); grid[:, rows, cols, :] = flat_array
        return grid
    return None

def analyze_source_sink_mortality(data, Sa_grid, Sj_grid, Fmax_grid, source_prob_mean, output_dir):
    print("Generating Probabilistic Source-Sink & Mortality Maps...")
    land_mask = data['land_mask']
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the final timestep or calculate the time-averaged consensus across the run
    # If source_prob_mean is (time, N_land) or (time, Ny, Nx), adjust axis if needed.
    # Assuming shape is (time, Ny, Nx) matching your existing code:
    prob_source_avg = np.mean(source_prob_mean, axis=0) 
    prob_masked = np.ma.masked_where(land_mask == 0, prob_source_avg)
    
    # 1. Binary Consensus Source-Sink Map (Your original)
    plt.figure(figsize=(10, 8))
    consensus_map = (prob_masked > 0.5).astype(float)
    
    cmap_binary = mcolors.ListedColormap(['#d73027', '#4575b4']) # Red=Sink, Blue=Source
    plt.imshow(consensus_map, cmap=cmap_binary, origin='upper')
    
    legend_elements = [
        mpatches.Patch(color='#4575b4', label=r'Consensus Source ($P(R_0 > 1) > 0.5$)'),
        mpatches.Patch(color='#d73027', label=r'Consensus Sink ($P(R_0 > 1) \leq 0.5$)')
    ]
    plt.legend(handles=legend_elements, loc='lower right', frameon=True)
    plt.title("Probabilistic Source-Sink Landscape (Bayesian Consensus)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "analysis_source_sink_probabilistic.png"), dpi=200)
    plt.close()

    # 1B. Continuous Gradient Source-Sink Map (The full probability spectrum)
    plt.figure(figsize=(11, 8))
    # Using 'RdYlBu_r' reverses it so Red=Sink (0.0), Yellow=Edge (0.5), Blue=Source (1.0)
    im_continuous = plt.imshow(prob_masked, cmap='RdYlBu_r', origin='upper', vmin=0.0, vmax=1.0)
    
    cbar_cont = plt.colorbar(im_continuous, fraction=0.036, pad=0.04)
    cbar_cont.set_label(r"Probability of Functioning as a Demographic Source $P(R_0 > 1.0)$")
    
    plt.title("Probabilistic Source-Sink Landscape (Continuous Bayesian Gradient)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "analysis_source_sink_continuous.png"), dpi=200)
    plt.close()

    # 2. Uncertainty/Fuzziness Map (The "Range Edge")
    fuzziness = 1.0 - 2.0 * np.abs(prob_masked - 0.5)
    plt.figure(figsize=(11, 8))
    plt.imshow(fuzziness, cmap='magma', origin='upper', vmin=0, vmax=1)
    
    cbar_fuzz = plt.colorbar(im_continuous, fraction=0.036, pad=0.04)
    cbar_fuzz.set_label("Model Uncertainty (Distance from 0.5 Probability)")
    
    plt.title("Range Limit Fuzziness (Stochastic Edge)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "analysis_range_edge_fuzziness.png"), dpi=200)
    plt.close()

def scatter_observations_to_grid(obs, t_idx, rows, cols, shape, time_steps):
    grid = np.full((time_steps, shape[0], shape[1]), np.nan)
    grid[t_idx, rows, cols] = obs
    return grid

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
    PATH_INTEGRATION_DIR = "/home/breallis/processed_data/datasets/latent_avian_paths"
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
    plot_posterior_weights(raw_samples, M, env_labels, OUTPUT_PLOT_DIR)
    
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
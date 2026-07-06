import sys
import os
import pickle
import numpy as np
import gc
import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import lax
from numpyro.handlers import substitute, seed, trace
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import warnings
import glob
from tqdm import tqdm

# --- Setup Paths ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.age_run_map import load_data_to_gpu
from src.model.age_priors import build_model_2d
from src.model.age_forward import dispersal_step_age_structured, reproduction_age_structured

# --- CONFIGURATION ---
PRECISION = 'float32' 
INPUT_DIR = "/home/breallis/processed_data/model_inputs/numpyro_input"
# Pointing to the HMC trial output directory
RESULT_DIR = f"/home/breallis/processed_data/model_results/age_hmc_{PRECISION}_trial_3"
OUTPUT_PLOT_DIR = os.path.join(RESULT_DIR, "plots_analysis")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# --- 1. DIRECT POSTERIORS (RAW SAMPLES) ---
def plot_posterior_weights(raw_samples, M, z_names, output_dir):
    """Plots hmc estimates for beta_s and beta_r with 90% Credible Intervals."""
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
    ax.set_title("Environmental Profile: Survival vs. Reproduction (HMC Posterior)")
    ax.set_ylim(-0.5, M - 0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_posterior_weights_hmc.png"), dpi=300)
    plt.close()

def plot_demographic_response_curves(raw_samples, M, z_names, target_z_idx, output_dir):
    """Sweeps a target Z axis to show the non-linear biological response with 90% HDI."""
    z_sweep = np.linspace(-3, 3, 100)
    Z_matrix = np.zeros((100, M))
    Z_matrix[:, target_z_idx] = z_sweep
    
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

# --- 3. STATE VARIABLES AND TRAJECTORIES ---
def plot_age_structure_dynamics(Na_grid_mean, Nj_grid_mean, years, output_dir, land_mask):
    print("Visualizing Expected Age Structure Trajectories...")
    total_Na = np.sum(Na_grid_mean, axis=(1, 2))
    total_Nj = np.sum(Nj_grid_mean, axis=(1, 2))
    total_pop = total_Na + total_Nj + 1e-9
    global_rho = total_Nj / total_pop

    plt.figure(figsize=(10, 5))
    plt.plot(years, global_rho, color='forestgreen', linewidth=2)
    plt.axhline(0.5, color='black', linestyle='--', label="50/50 Theoretical Target")
    plt.title(r"Continental Expected Juvenile Fraction ($\mathbb{E}[N_j / N_{total}]$)")
    plt.xlabel("Year")
    plt.ylabel("Proportion of Population")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "trend_age_structure.png"))
    plt.close()

    snapshots = [1970, 1990, 2010, 2020]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, year in zip(axes.flatten(), snapshots):
        t_idx = np.where(years == year)[0][0]
        na = Na_grid_mean[t_idx]
        nj = Nj_grid_mean[t_idx]
        valid_pop = (na + nj) > 0.01
        rho_spatial = np.where(valid_pop, nj / (na + nj + 1e-9), np.nan)
        rho_masked = np.ma.masked_where((land_mask == 0) | ~valid_pop, rho_spatial)
        cmap = plt.cm.get_cmap('BrBG', 11)
        im = ax.imshow(rho_masked, cmap=cmap, origin='upper', vmin=0.2, vmax=0.8)
        ax.set_title(f"Juvenile Fraction - {year}")
        ax.axis('off')

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label="Juvenile Fraction (Brown = Adult heavy, Green = Juvenile heavy)")
    plt.suptitle("Spatiotemporal Expected Age Structure Dynamics", fontsize=18, y=0.95)
    plt.savefig(os.path.join(output_dir, "map_age_structure_snapshots.png"), dpi=200)
    plt.close()

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
    
    prob_source_avg = np.mean(source_prob_mean, axis=0) 
    prob_masked = np.ma.masked_where(land_mask == 0, prob_source_avg)
    
    # 1. Binary Consensus Source-Sink Map
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

    # 1B. Continuous Gradient Source-Sink Map
    plt.figure(figsize=(11, 8))
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

# --- MAIN CONTROLLER & HMC REBUILDER ---
def rebuild_spatial_grids(raw_samples, data_dict):
    """
    Streams the HMC parameter samples through the JAX forward simulator on the GPU 
    to rebuild the dense spatial grids without throwing an OOM error.
    """
    print("\n--- Rebuilding Spatial Grids from HMC Traces ---")
    
    @jax.jit
    def fast_forward_sim(single_sample):
        seeded_model = seed(build_model_2d, jax.random.PRNGKey(0))
        substituted = substitute(seeded_model, data=single_sample)
        model_trace = trace(substituted).get_trace(data=data_dict, anneal=1.0)
        return {
            'simulated_density': model_trace['simulated_density']['value'],
            'Sa_flat': model_trace['Sa_flat']['value'],
            'Sj_flat': model_trace['Sj_flat']['value'],
            'Fmax_flat': model_trace['Fmax_flat']['value'],
            'K_flat': model_trace['K_flat']['value'],
            'Q_flat': model_trace['Q_flat']['value'],
            'allee_gamma': model_trace['allee_gamma']['value'] 
        }

    @jax.jit
    def fast_rebuild_ages(sim_output, single_sample):
        time, Ny, Nx = data_dict['time'], data_dict['Ny'], data_dict['Nx']
        row, col = data_dict['inv_location']
        inv_pop = jnn.softplus(single_sample['inv_eta'])
        disp_log_int = single_sample['dispersal_logit_intercept']
        disp_log_slope = single_sample['dispersal_logit_slope']
        allee_gamma = sim_output['allee_gamma']

        def step(pools, t):
            N_a, N_j = pools
            k = t - data_dict['inv_timestep']
            val = jnp.where((k >= 0) & (k < inv_pop.shape[0]), inv_pop[jnp.clip(k, 0, inv_pop.shape[0]-1)], 0.0)
            N_a = N_a.at[row, col].add(val)
            Sa_g = jnp.zeros((Ny, Nx)).at[data_dict['land_rows'], data_dict['land_cols']].set(sim_output['Sa_flat'][t])
            Sj_g = jnp.zeros((Ny, Nx)).at[data_dict['land_rows'], data_dict['land_cols']].set(sim_output['Sj_flat'][t])
            Fmax_g = jnp.zeros((Ny, Nx)).at[data_dict['land_rows'], data_dict['land_cols']].set(sim_output['Fmax_flat'][t])
            K_g = jnp.zeros((Ny, Nx)).at[data_dict['land_rows'], data_dict['land_cols']].set(sim_output['K_flat'][t])
            Q_t = sim_output['Q_flat'][t]
            Q_g = jnp.zeros((Ny, Nx, Q_t.shape[-1])).at[data_dict['land_rows'], data_dict['land_cols'], :].set(Q_t).transpose(2, 0, 1)

            N_a_post, j_stayers, j_arriving = dispersal_step_age_structured(
                N_a, N_j, K_g, disp_log_int, disp_log_slope, 0.8,
                data_dict['adult_edge_correction'], data_dict['juvenile_edge_correction_stack'],
                data_dict['adult_fft_kernel'], data_dict['juvenile_fft_kernel_stack'], Q_g, 1e-6
            )
            N_a_new, N_j_new = reproduction_age_structured(
                N_a_post, j_stayers, j_arriving, Sa_g, Sj_g, Fmax_g, K_g, allee_gamma
            )
            return (jnp.maximum(N_a_new * data_dict['land_mask'], 0.0), 
                    jnp.maximum(N_j_new * data_dict['land_mask'], 0.0)), (N_a_new, N_j_new)

        _, (Na_grid, Nj_grid) = lax.scan(step, (data_dict['initpop_latent'] * 0.5, data_dict['initpop_latent'] * 0.5), jnp.arange(time))
        return {'Na_flat': Na_grid[:, data_dict['land_rows'], data_dict['land_cols']], 
                'Nj_flat': Nj_grid[:, data_dict['land_rows'], data_dict['land_cols']]}

    heavy_keys = ['simulated_density', 'Sa_flat', 'Sj_flat', 'Fmax_flat', 'K_flat', 'Q_flat', 'Na_flat', 'Nj_flat']
    welford_stats = {k: {'mean': None, 'M2': None} for k in heavy_keys}
    
    num_samples = raw_samples['w_env'].shape[0]
    source_counter = np.zeros((data_dict['time'], data_dict['N_land']), dtype=np.float32)

    for i in tqdm(range(num_samples), desc="Rebuilding HMC Spatial Trace"):
        # Isolate exactly one sample draw
        single_sample = {k: v[i] for k, v in raw_samples.items()}
        
        sim_output_gpu = fast_forward_sim(single_sample)
        age_output_gpu = fast_rebuild_ages(sim_output_gpu, single_sample)
        
        current_grids = {}
        for k, v in sim_output_gpu.items():
            current_grids[k] = np.array(v)
            v.delete()
        for k, v in age_output_gpu.items():
            current_grids[k] = np.array(v)
            v.delete()

        sample_R0 = (current_grids['Fmax_flat'] * current_grids['Sj_flat']) / (1.0 - current_grids['Sa_flat'] + 1e-6)
        source_counter += (sample_R0 > 1.0).astype(np.float32)
                
        count = i + 1
        for k in heavy_keys:
            x = current_grids[k]
            if welford_stats[k]['mean'] is None:
                welford_stats[k]['mean'] = np.zeros_like(x)
                welford_stats[k]['M2'] = np.zeros_like(x)
            delta = x - welford_stats[k]['mean']
            welford_stats[k]['mean'] += delta / count
            welford_stats[k]['M2'] += delta * (x - welford_stats[k]['mean'])
        
        del current_grids, sample_R0
        if i % 10 == 0: gc.collect()

    final_output = {f"{k}_mean": welford_stats[k]['mean'] for k in heavy_keys}
    final_output.update({f"{k}_var": welford_stats[k]['M2'] / max(1, num_samples - 1) for k in heavy_keys})
    final_output['source_probability_mean'] = source_counter / num_samples
    
    return final_output


def plot_results():
    data = load_data_to_gpu(INPUT_DIR, precision=PRECISION)

    hmc_path = os.path.join(RESULT_DIR, "hmc_trial_samples.pkl")
    if not os.path.exists(hmc_path):
        raise FileNotFoundError(f"HMC raw samples not found at {hmc_path}.")
        
    print(f"Loading raw HMC traces from {hmc_path}...")
    with open(hmc_path, 'rb') as f:
        raw_samples = pickle.load(f)

    # 1. Manage the Rebuilt Spatial Stats Cache
    sample_cache_path = os.path.join(OUTPUT_PLOT_DIR, "hmc_reconstructed_stats.npz")
    if os.path.exists(sample_cache_path):
        print(f"Loading cached spatial statistics from {sample_cache_path}...")
        with np.load(sample_cache_path) as loader:
            spatial_stats = {k: loader[k] for k in loader.files}
    else:
        # Rebuild the grids if they haven't been compiled yet
        spatial_stats = rebuild_spatial_grids(raw_samples, data)
        np.savez_compressed(sample_cache_path, **spatial_stats)
        print(f"Saved rebuilt spatial statistics to cache: {sample_cache_path}")

    # 2. Load Directional Dispersal Kernel Labels (K = 12)
    PATH_INTEGRATION_DIR = "/home/breallis/processed_data/datasets/latent_avian_paths"
    disp_files = glob.glob(os.path.join(PATH_INTEGRATION_DIR, "Z_disp_*.npz"))
    
    if not disp_files:
        print("Warning: No Z_disp files found. Labels will be generic.")
        kernel_labels = [f"Kernel_{i}" for i in range(data['juvenile_fft_kernel_stack'].shape[0])] 
    else:
        with np.load(disp_files[0]) as loader:
            kernel_labels = [str(lbl) for lbl in loader['labels']]
            
    # 3. Generate Environmental Feature Labels (M = 16)
    M = data['Z_gathered'].shape[-1]
    env_labels = [f"Env_PC_{i+1}" for i in range(M)]

    Ny, Nx = data['Ny'], data['Nx']
    rows, cols = data['land_rows'], data['land_cols']
    years = data['years']
    start_idx = np.where(years >= 1960)[0][0]
    
    # 4. Re-scatter pre-aggregated means
    Sa_grid_mean = scatter_to_grid_robust(spatial_stats['Sa_flat_mean'], rows, cols, (Ny, Nx))
    Sj_grid_mean = scatter_to_grid_robust(spatial_stats['Sj_flat_mean'], rows, cols, (Ny, Nx))
    Fmax_grid_mean = scatter_to_grid_robust(spatial_stats['Fmax_flat_mean'], rows, cols, (Ny, Nx))
    K_grid_mean = scatter_to_grid_robust(spatial_stats['K_flat_mean'], rows, cols, (Ny, Nx))
    
    # 5. Re-scatter exact pre-aggregated variances
    Sa_grid_var = scatter_to_grid_robust(spatial_stats['Sa_flat_var'], rows, cols, (Ny, Nx))
    Sj_grid_var = scatter_to_grid_robust(spatial_stats['Sj_flat_var'], rows, cols, (Ny, Nx))
    Fmax_grid_var = scatter_to_grid_robust(spatial_stats['Fmax_flat_var'], rows, cols, (Ny, Nx))
    K_grid_var = scatter_to_grid_robust(spatial_stats['K_flat_var'], rows, cols, (Ny, Nx))
    
    # 6. State variable means (No rebuilding needed - avoiding Jensen's Inequality)
    density_mean = spatial_stats['simulated_density_mean'] * data['pop_scalar']
    Na_grid_mean = scatter_to_grid_robust(spatial_stats['Na_flat_mean'], rows, cols, (Ny, Nx))
    Nj_grid_mean = scatter_to_grid_robust(spatial_stats['Nj_flat_mean'], rows, cols, (Ny, Nx))
    
    obs_grid = scatter_observations_to_grid(data['observed_results'], data['obs_time_indices'], data['obs_rows'], data['obs_cols'], (Ny, Nx), data['time'])
    source_prob_mean = scatter_to_grid_robust(spatial_stats['source_probability_mean'], rows, cols, (Ny, Nx))
    
    # --- Execute Diagnostic Suite ---
    plot_posterior_weights(raw_samples, M, env_labels, OUTPUT_PLOT_DIR)
    
    for target_idx in range(min(3, M)):
        plot_demographic_response_curves(raw_samples, M, env_labels, target_idx, OUTPUT_PLOT_DIR)
    
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
    plot_age_structure_dynamics(Na_grid_mean, Nj_grid_mean, years, OUTPUT_PLOT_DIR, data['land_mask'])
    create_animation(density_mean, obs_grid, years, OUTPUT_PLOT_DIR, data['land_mask'])

    print(f"All probabilistic diagnostics complete. Results at: {OUTPUT_PLOT_DIR}")

if __name__ == "__main__":
    plot_results()
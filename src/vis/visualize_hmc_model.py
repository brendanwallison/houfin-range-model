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

from src.model.data_loading import load_data_to_gpu
from src.model.age_priors import build_model_2d
from src.model.age_forward import dispersal_step_age_structured, reproduction_age_structured
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
# Pointing to the HMC trial output directory
RESULT_DIR = os.path.join(_cfg["results_dir"], _cfg["run_names"]["vis_hmc"].format(precision=PRECISION))
OUTPUT_PLOT_DIR = os.path.join(RESULT_DIR, "plots_analysis")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

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
    PATH_INTEGRATION_DIR = _cfg["path_features"]["output_dir"]
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
    plot_posterior_weights(raw_samples, M, env_labels, OUTPUT_PLOT_DIR, label="HMC", fname="1_posterior_weights_hmc.png")

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
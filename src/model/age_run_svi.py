import sys
import os
import pickle
import numpy as np
import gc
import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import lax
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
import optax
from tqdm import tqdm
from numpyro.handlers import substitute, seed, trace
import matplotlib.pyplot as plt

# --- SINGLE POINT OF CONTROL ---
PRECISION = 'float32' 

jax.config.update("jax_enable_x64", True if PRECISION == 'float64' else False)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.age_priors import build_model_2d
from src.model.age_forward import dispersal_step_age_structured, reproduction_age_structured
from src.model.data_loading import load_data
from src.config_utils import load_age_model_config

# --- CONFIGURATION ---
_cfg = load_age_model_config()
INPUT_DIR = _cfg["input_dir"]
OUTPUT_DIR = os.path.join(_cfg["results_dir"], _cfg["run_names"]["svi"].format(precision=PRECISION))

def run_vi():
    print(f"--- Starting {PRECISION.upper()} SVI Training (Age-Structured) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params_path = os.path.join(OUTPUT_DIR, "svi_params.pkl")
    
    # We will use the GPU for all math
    gpu_device = jax.devices("gpu")[0]
    data_dict = load_data(INPUT_DIR, gpu_device, precision=PRECISION)
    
    # --- RESUME DETECTOR ---
    if os.path.exists(params_path):
        print(f"-> Found existing parameters at {params_path}. Resuming to Welford Loop.")
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
    else:
        # --- TRAINING LOGIC (GPU) ---
        # rank=20 captures the top 20 principal axes of variation across the parameter space
        guide = AutoLowRankMultivariateNormal(build_model_2d, rank=20)
        total_steps = 600
        anneal_epochs = [0.1, 0.5, 1.0]
        steps_per_epoch = total_steps // len(anneal_epochs)
        scheduler = optax.cosine_decay_schedule(init_value=0.01, decay_steps=total_steps, alpha=0.1)
        optimizer = numpyro.optim.optax_to_numpyro(
            optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=scheduler, weight_decay=1e-4, eps=1e-7))
        )
        svi = SVI(build_model_2d, guide, optimizer, loss=Trace_ELBO())
        
        print("Compiling model & variational guide on GPU...")
        rng_key = jax.random.PRNGKey(41)
        svi_state = svi.init(rng_key, data=data_dict, anneal=anneal_epochs[0])

        for anneal_level in anneal_epochs:
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch (Anneal={anneal_level})")
            for i in pbar:
                svi_state, loss = svi.update(svi_state, data=data_dict, anneal=anneal_level)
                if i % 10 == 0: pbar.set_postfix({"loss": f"{float(loss):.4f}"})
        
        params = svi.get_params(svi_state)
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
        print("Training Complete. Params saved.")

    # --- WELFORD'S EXACT VARIANCE ENGINE (HYBRID) ---
    print("\n--- Initializing Welford's Exact Variance Engine ---")
    total_samples = 500
    sampling_key = jax.random.PRNGKey(42)
    
    # Must match the guide used for training (line above); the saved params are
    # a low-rank MVN parameter set, not a diagonal one.
    guide = AutoLowRankMultivariateNormal(build_model_2d, rank=20)

    # Initialize guide structure internally on the active device
    mock_svi = SVI(build_model_2d, guide, optax.adam(1e-3), loss=Trace_ELBO())
    _ = mock_svi.init(jax.random.PRNGKey(999), data=data_dict, anneal=1.0)
    
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
    source_counter = np.zeros((data_dict['time'], data_dict['N_land']), dtype=np.float32)
    global_samples_accumulator = {}

    print(f"Aggregating {total_samples} samples (Math on GPU, Welford on CPU)...")
    
    # Pre-compile JAX graph to avoid loop timing overhead
    sampling_key, dummy_key = jax.random.split(sampling_key)
    dummy_sample = guide.sample_posterior(dummy_key, params, sample_shape=())
    _ = fast_forward_sim(dummy_sample)
    for v in dummy_sample.values():
        v.delete()

    for i in tqdm(range(total_samples), desc="Welford Loop"):
        # 1. Stream exactly ONE sample parameter profile onto GPU
        sampling_key, subkey = jax.random.split(sampling_key)
        sample_gpu = guide.sample_posterior(subkey, params, sample_shape=())
        
        # 2. Run forward simulators on GPU
        sim_output_gpu = fast_forward_sim(sample_gpu)
        age_output_gpu = fast_rebuild_ages(sim_output_gpu, sample_gpu)
        
        # 3. Pull heavy arrays to host RAM as NumPy arrays, then shred GPU tracking pointers
        current_grids = {}
        for k, v in sim_output_gpu.items():
            current_grids[k] = np.array(v)
            v.delete()
        for k, v in age_output_gpu.items():
            current_grids[k] = np.array(v)
            v.delete()
        
        # 4. Extract lightweight global parameters to host lists before shredding sample
        for k, v in sample_gpu.items():
            if k not in heavy_keys:
                if k not in global_samples_accumulator:
                    global_samples_accumulator[k] = []
                global_samples_accumulator[k].append(np.array(v))
        
        for v in sample_gpu.values():
            v.delete()

        # 5. Pure NumPy Welford math and thresholding (Completely clear of JAX/VRAM)
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

    # --- PACKAGE AND SAVE ---
    print("\nFinalizing Variational Statistics...")
    final_output = {f"{k}_mean": welford_stats[k]['mean'] for k in heavy_keys}
    final_output.update({f"{k}_var": welford_stats[k]['M2'] / (total_samples - 1) for k in heavy_keys})
    final_output['source_probability_mean'] = source_counter / total_samples
    
    # Stack the tracked global parameter iterations into regular arrays
    for k, v in global_samples_accumulator.items():
        final_output[k] = np.stack(v, axis=0)

    analysis_dir = os.path.join(OUTPUT_DIR, "plots_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    sample_cache_path = os.path.join(analysis_dir, "reconstructed_samples_stats.npz")
    np.savez_compressed(sample_cache_path, **final_output)
    print(f"-> Saved exact statistical moments & global parameters to: {sample_cache_path}")
    print(f"Process Complete.")

if __name__ == "__main__":
    run_vi()
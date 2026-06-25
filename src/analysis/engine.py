import os
import pickle
import jax
import numpy as np
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoLowRankMultivariateNormal

def load_params(result_dir):
    """Detects and loads parameters from MAP or SVI runs."""
    # Priority: SVI Posterior > SVI Checkpoint > MAP Params
    possible_files = [
        "vi_posterior_params.pkl",
        "vi_params_step_5000.pkl", # Example checkpoint
        "map_params.pkl"
    ]
    
    for f in possible_files:
        path = os.path.join(result_dir, f)
        if os.path.exists(path):
            print(f"-> Loading parameters from: {path}")
            with open(path, 'rb') as src:
                return pickle.load(src), f
    
    raise FileNotFoundError(f"No parameter files found in {result_dir}")

def get_guide(model, filename, params):
    """Automatically selects the correct guide based on the file source."""
    if "map_params" in filename:
        return AutoDelta(model)
    
    # For SVI, check parameter keys to determine guide type
    if "auto_scale" in params:
        if "scale_tril" in params:
            # This would be for LowRank or FullRank guides
            return AutoLowRankMultivariateNormal(model) 
        return AutoDiagonalNormal(model)
    
    return AutoDelta(model)

def reconstruct_latents(model, data, params, filename, num_samples=100, chunk_size=1):
    """
    Reconstructs latents using a Python loop to prevent JAX compilation bloat.
    Forces the GPU to only think about one simulation at a time.
    """
    guide = get_guide(model, filename, params)
    
    return_sites = [
        "simulated_density", "Sa_flat", "Sj_flat", "Fmax_flat", 
        "K_flat", "Q_flat", "w_env", "n50_raw", "allee_gamma"
    ]
    
    actual_samples = 1 if "map_params" in filename else num_samples
    print(f"-> Drawing {actual_samples} samples from the posterior...")

    # Initialize predictive with a batch size of 1
    predictive = Predictive(
        model, 
        guide=guide, 
        params=params, 
        num_samples=1, # Draw one at a time
        return_sites=return_sites,
        batch_ndims=0
    )
    
    rng_key = jax.random.PRNGKey(42)
    
    all_samples = []
    
    print(f"-> Running Python-level sequential reconstruction...")
    for i in range(actual_samples):
        if i % 10 == 0:
            print(f"   Processing sample {i}/{actual_samples}...")
        
        # New key for every sample
        step_key = jax.random.fold_in(rng_key, i)
        
        # Execute a single simulation
        # Because this is a Python loop, JAX only sees 1 sim at a time
        sample = predictive(step_key, data=data)
        
        # Remove the leading unit dimension and move to CPU immediately to save VRAM
        sample_cpu = {k: jax.device_get(v[0]) for k, v in sample.items()}
        all_samples.append(sample_cpu)

    # Recombine the list of dicts into a single dict of arrays
    print("-> Finalizing and stacking results...")
    final_dict = {
        k: np.stack([s[k] for s in all_samples]) 
        for k in all_samples[0].keys()
    }
    
    # If MAP, drop the sample dimension
    if actual_samples == 1:
        return {k: v[0] for k, v in final_dict.items()}
        
    return final_dict
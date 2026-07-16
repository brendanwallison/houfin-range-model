"""Fit the age-structured model by HMC/NUTS sampling (full posterior).

The most expensive but most complete inference path: draws posterior samples
from ``build_model_2d`` with NUTS. Uses a NeuTra reparameterization and
memory-safe tricks (hiding/transforming large deterministic arrays in chunks) so
the long forward simulation is tractable to sample. GPU-oriented; usually
initialized from a MAP fit (see ``age_resume_hmc``).
"""
import sys
import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.infer.reparam import NeuTraReparam
from numpyro.handlers import reparam, block


# --- SINGLE POINT OF CONTROL ---
PRECISION = 'float32' 

jax.config.update("jax_enable_x64", True if PRECISION == 'float64' else False)

# Use the async allocator to completely eliminate VRAM fragmentation during tracing
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

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
OUTPUT_DIR = os.path.join(_cfg["results_dir"], _cfg["run_names"]["hmc"].format(precision=PRECISION))

def run_hmc_trial():
    print(f"--- Starting {PRECISION.upper()} HMC Trial (NeuTra Reparameterized) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    gpu_device = jax.devices("gpu")[0]
    data_dict = load_data(INPUT_DIR, gpu_device, precision=PRECISION)
    
    # 2. Load the SVI Parameters
    svi_params_path = os.path.join(
        _cfg["results_dir"],
        _cfg["run_names"]["hmc_svi_source"].format(precision=PRECISION),
        "svi_params.pkl",
    )
    if not os.path.exists(svi_params_path):
        raise FileNotFoundError(f"Could not find SVI params at {svi_params_path}. Run SVI script first.")
    
    print(f"Loading SVI Low-Rank guide from {svi_params_path}...")
    with open(svi_params_path, 'rb') as f:
        svi_params = pickle.load(f)

    # 3. Instantiate the exact same guide and INITIALIZE its internal structures
    guide = AutoLowRankMultivariateNormal(build_model_2d, rank=20)
    
    from numpyro.infer import SVI, Trace_ELBO
    import optax
    mock_svi = SVI(build_model_2d, guide, optax.adam(1e-3), loss=Trace_ELBO())
    _ = mock_svi.init(jax.random.PRNGKey(999), data=data_dict, anneal=1.0)

    # NOW wrap the initialized guide in NeuTraReparam
    neutra = NeuTraReparam(guide, svi_params)

    # 4. Warp the original model using the NeuTra configuration
    def neutra_config(site):
        if site["type"] == "sample" and not site.get("is_observed", False):
            if not site.get("infer", {}).get("is_auxiliary", False):
                if site["name"] in guide.prototype_trace:
                    return neutra
        return None

    # Base warped model
    base_neutra_model = reparam(build_model_2d, config=neutra_config)

    # Wrapper to hide massive deterministics from the MCMC memory pre-allocator
    def hide_massive_arrays(site):
        return site["type"] == "deterministic"

    def memory_safe_model(*args, **kwargs):
        with block(hide_fn=hide_massive_arrays):
            base_neutra_model(*args, **kwargs)

    # 5. Configure NUTS using the memory-safe wrapper
    nuts_kernel = NUTS(
        memory_safe_model,
        target_accept_prob=0.4,
        init_strategy=init_to_median,
        max_tree_depth=5
    )

    # Trial Run Parameters: 50 warmup steps, 50 samples, 1 chain
    num_warmup = 100
    num_samples = 200
    
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=True
    )

    # 6. Execute HMC (Explicitly passing anneal=1.0 to match your model signature)
    rng_key = jax.random.PRNGKey(84)
    print(f"\nStarting NUTS sampling: {num_warmup} warmup, {num_samples} samples...")
    mcmc.run(rng_key, data=data_dict, anneal=1.0)
    
    # 7. Transform the samples back to their original ecological space
    print("\nTransforming warped samples back to original parameter space...")
    zs_neutra = mcmc.get_samples()
    warped_latents = zs_neutra['auto_shared_latent']

    # Define the chunked transformation logic
    def transform_in_chunks(latents_array, chunk_size=10):
        num_samples = latents_array.shape[0]
        processed_chunks = []
        
        print(f"Unpacking {num_samples} samples in safe VRAM chunks of {chunk_size}...")
        for i in range(0, num_samples, chunk_size):
            # Slice a predictable, static block size
            chunk = latents_array[i:i+chunk_size]
            
            # XLA compiles the unrolled bijectors for this exact block size once, 
            # capping the intermediate memory overhead.
            chunk_unpacked = neutra.transform_sample(chunk)
            
            # Immediately pull the evaluated chunk to Host CPU memory (NumPy)
            chunk_cpu = jax.tree_util.tree_map(lambda x: np.array(x), chunk_unpacked)
            processed_chunks.append(chunk_cpu)
            
            # Force JAX to explicitly release the intermediate GPU buffers
            jax.tree_util.tree_map(lambda x: x.delete(), chunk_unpacked)
            
        # Merge the CPU chunks back into a single batched dictionary
        print("Concatenating unpacked chunks on CPU...")
        return {
            k: np.concatenate([c[k] for c in processed_chunks], axis=0)
            for k in processed_chunks[0].keys()
        }

    # Execute the chunked transformation
    original_samples = transform_in_chunks(warped_latents, chunk_size=10)

    # 8. Save the test traces
    trace_path = os.path.join(OUTPUT_DIR, "hmc_trial_samples.pkl")
    with open(trace_path, 'wb') as f:
        pickle.dump(original_samples, f)
        
    print(f"Trial complete. Transformed samples saved to {trace_path}.")
    mcmc.print_summary()

if __name__ == "__main__":
    run_hmc_trial()
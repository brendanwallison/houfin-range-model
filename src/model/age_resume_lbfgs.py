import os
import pickle
import sys
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.infer.initialization import init_to_value
import matplotlib.pyplot as plt
import optax

# --- Configuration ---
PRECISION = 'float32'
jax.config.update("jax_enable_x64", True if PRECISION == 'float64' else False)

# Output directory for the VI results
RESULT_DIR = f"/home/breallis/processed_data/model_results/age_vi_{PRECISION}_run_15"
# Source directory of your converged Adam/L-BFGS MAP run
PREVIOUS_MAP_DIR = f"/home/breallis/processed_data/model_results/age_map_{PRECISION}_run_13"
INPUT_DIR = "/home/breallis/processed_data/model_inputs/numpyro_input"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.age_priors import build_model_2d
from src.model.age_run_map import load_data_to_gpu

def run_vi_resume():
    print(f"--- Resuming MAP as Variational Inference Initializer ---")
    
    # 1. Load Data
    data_dict = load_data_to_gpu(INPUT_DIR, precision=PRECISION)
    
    # 2. Load the Previous MAP Parameters
    with open(os.path.join(PREVIOUS_MAP_DIR, "map_params.pkl"), 'rb') as f:
        map_params = pickle.load(f)

    # Convert MAP params to latent-site dict
    map_latents = {k.replace("_auto_loc", ""): v for k, v in map_params.items()}

    # 3. Setup Model & Guide
    model = build_model_2d
    # guide = AutoDiagonalNormal(
    #     model,
    #     init_loc_fn=init_to_value(values=map_latents),
    #     init_scale=0.01
    # )

    guide = AutoLowRankMultivariateNormal(
        model,
        init_loc_fn=init_to_value(values=map_latents),
        init_scale=0.01,
        rank=10  # Learned correlations for the top 10 'directions' of uncertainty
    )

    # 4. Setup Optimizer
    total_steps = 5000
    anneal_epochs = [1.0]
    steps_per_epoch = total_steps // len(anneal_epochs)

    scheduler = optax.cosine_decay_schedule(init_value=5e-4, decay_steps=total_steps, alpha=0.1)
    optimizer = numpyro.optim.optax_to_numpyro(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=scheduler))
    )

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=1))

    # --- Initialization ---
    print("Initializing VI from MAP coordinates...")
    rng_key = jax.random.PRNGKey(0)
    svi_state = svi.init(rng_key, data=data_dict, anneal=1.0)

    # 6. Run Training (Anneal fixed at 1.0)
    all_losses = []
    steps_per_block = 100 # Adjusted based on your last run
    num_blocks = total_steps // steps_per_block 
    
    # CRITICAL: Create the directory before the loop starts
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"--- Starting VI Training | Total Steps: {total_steps} | Anneal: 1.0 ---")

    for b in range(num_blocks):
        def body_fn(state, _):
            return svi.update(state, data=data_dict, anneal=1.0)

        svi_state, block_losses = jax.lax.scan(body_fn, svi_state, jnp.arange(steps_per_block))
        all_losses.append(block_losses)
        
        # --- PROGRESS REPORT ---
        current_step = (b + 1) * steps_per_block
        mean_elbo = jnp.mean(block_losses)
        
        current_params = svi.get_params(svi_state)
        avg_sigma = jnp.mean(jnp.exp(current_params['auto_scale']))
        
        print(f"Block {b+1:02d}/{num_blocks} | Step: {current_step:05d} | ELBO: {mean_elbo:.4f} | Avg Sigma: {avg_sigma:.4f}")

        # Save safety checkpoint every 500 steps
        if current_step % 500 == 0:
            ckpt_path = os.path.join(RESULT_DIR, f"vi_params_step_{current_step}.pkl")
            with open(ckpt_path, 'wb') as f:
                pickle.dump(current_params, f)

    # 7. Final Handoff and Save
    final_params = svi.get_params(svi_state)
    losses = jnp.concatenate(all_losses)
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, "vi_posterior_params.pkl"), 'wb') as f:
        pickle.dump(final_params, f)
    
    print(f"VI Resume complete. Final Loss: {losses[-1]:.4f}")

if __name__ == "__main__":
    run_vi_resume()
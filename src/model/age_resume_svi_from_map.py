"""Run SVI starting from a saved MAP initialization.

Loads a MAP checkpoint (``age_run_map``) and initializes the SVI guide from it,
so variational inference refines a posterior around the MAP mode instead of
optimizing from scratch. Writes the fitted guide/params. GPU-oriented.
"""
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
import numpy as np

# --- Configuration ---
PRECISION = 'float32'
jax.config.update("jax_enable_x64", True if PRECISION == 'float64' else False)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.age_priors import build_model_2d
from src.model.data_loading import load_data
from src.model.checkpoints import (
    auto_delta_params_to_latents, load_map_params, save_pickle_atomic,
)
from src.model.runtime_diagnostics import memory_snapshot, require_gpu
from src.config_utils import load_age_model_config

_cfg = load_age_model_config()
INPUT_DIR = _cfg["input_dir"]
# Output directory for the VI results
RESULT_DIR = os.path.join(_cfg["results_dir"], _cfg["run_names"]["resume_svi_out"].format(precision=PRECISION))
# Source directory of the converged MAP run this warm-starts from
PREVIOUS_MAP_DIR = os.path.join(_cfg["results_dir"], _cfg["run_names"]["resume_svi_from_map"].format(precision=PRECISION))

def run_vi_resume():
    print(f"--- Resuming MAP as Variational Inference Initializer ---")
    
    # 1. Load Data
    gpu_device = require_gpu("MAP-initialized SVI")
    data_dict = load_data(INPUT_DIR, gpu_device, precision=PRECISION)
    memory_snapshot("resume-svi-inputs-loaded", gpu_device)
    
    # 2. Load the Previous MAP Parameters
    map_params, map_checkpoint = load_map_params(PREVIOUS_MAP_DIR)
    print(f"Loaded verified MAP checkpoint at step {map_checkpoint['step']}")

    # Convert MAP params to latent-site dict
    map_latents = auto_delta_params_to_latents(map_params)

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
    steps_per_epoch = total_steps

    scheduler = optax.cosine_decay_schedule(init_value=5e-4, decay_steps=total_steps, alpha=0.1)
    optimizer = numpyro.optim.optax_to_numpyro(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=scheduler))
    )

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=1))

    # --- Initialization / exact optimizer resume ---
    os.makedirs(RESULT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(RESULT_DIR, "vi_checkpoint.pkl")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as fh:
            checkpoint = pickle.load(fh)
        if checkpoint.get("map_fingerprint") != map_checkpoint["fingerprint"]:
            raise RuntimeError("VI checkpoint was initialized from a different MAP run")
        if int(checkpoint.get("total_steps", -1)) != total_steps:
            raise RuntimeError("VI total_steps changed; start a fresh output run")
        svi_state = checkpoint["svi_state"]
        start_step = int(checkpoint["step"])
        loss_history = list(np.asarray(checkpoint["losses"]))
        print(f"Resuming exact VI optimizer state at {start_step}/{total_steps}")
    else:
        print("Initializing VI from MAP coordinates...")
        svi_state = svi.init(
            jax.random.PRNGKey(0), data=data_dict, prior_scale=1.0
        )
        start_step, loss_history = 0, []

    # 6. Run training under nominal priors.
    steps_per_block = 100 # Adjusted based on your last run
    if start_step % steps_per_block:
        raise RuntimeError("VI checkpoint step is not aligned to block size")
    print(f"--- Starting VI Training | Total Steps: {total_steps} | prior_scale=1.0 ---")

    for block_start in range(start_step, total_steps, steps_per_block):
        def body_fn(state, _):
            return svi.update(state, data=data_dict, prior_scale=1.0)

        block_size = min(steps_per_block, total_steps - block_start)
        svi_state, block_losses = jax.lax.scan(
            body_fn, svi_state, jnp.arange(block_size)
        )
        loss_history.extend(np.asarray(block_losses).tolist())
        
        # --- PROGRESS REPORT ---
        current_step = block_start + block_size
        mean_elbo = jnp.mean(block_losses)
        
        current_params = svi.get_params(svi_state)
        avg_sigma = jnp.mean(jnp.exp(current_params['auto_scale']))
        
        print(f"Step: {current_step:05d}/{total_steps} | ELBO: {mean_elbo:.4f} | Avg Sigma: {avg_sigma:.4f}")

        # Save the optimizer state, not merely a parameter snapshot.
        save_pickle_atomic(
            {"format_version": 2, "svi_state": svi_state,
             "params": current_params, "step": current_step,
             "losses": np.asarray(loss_history), "total_steps": total_steps,
             "map_fingerprint": map_checkpoint["fingerprint"]},
            checkpoint_path,
        )
        memory_snapshot(f"resume-svi-{current_step}", gpu_device)

    # 7. Final Handoff and Save
    final_params = svi.get_params(svi_state)
    losses = np.asarray(loss_history)
    save_pickle_atomic(
        {"format_version": 1, "params": final_params, "step": total_steps,
         "map_fingerprint": map_checkpoint["fingerprint"]},
        os.path.join(RESULT_DIR, "vi_posterior_params.pkl"),
    )
    
    print(f"VI Resume complete. Final Loss: {losses[-1]:.4f}")

if __name__ == "__main__":
    run_vi_resume()

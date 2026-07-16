"""Fit the age-structured model by MAP (maximum-a-posteriori) optimization.

Runs NumPyro's ``AutoDelta`` / gradient optimization over ``build_model_2d`` to
get a point estimate, typically as a fast first pass and as the initialization
for SVI (``age_resume_svi_from_map``) or HMC (``age_resume_hmc``). Writes the MAP
parameters to a pickle checkpoint. GPU-oriented; see the README for memory notes.
"""
import sys
import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

# --- SINGLE POINT OF CONTROL ---
PRECISION = 'float32' # Options: 'float32' or 'float64'

jax.config.update("jax_enable_x64", True if PRECISION == 'float64' else False)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.age_priors import build_model_2d
from src.model.data_loading import load_data, load_data_to_gpu
from src.config_utils import load_age_model_config

# --- CONFIGURATION ---
_cfg = load_age_model_config()
INPUT_DIR = _cfg["input_dir"]
OUTPUT_DIR = os.path.join(_cfg["results_dir"], _cfg["run_names"]["map"].format(precision=PRECISION))

def run_map():
    print(f"--- Starting {PRECISION.upper()} Optimization (Age-Structured) ---")
    data_dict = load_data_to_gpu(INPUT_DIR, precision=PRECISION)

    guide = AutoDelta(build_model_2d)

    total_steps = 900
    anneal_epochs = [0.1, 0.5, 1.0]
    steps_per_epoch = total_steps // len(anneal_epochs)

    scheduler = optax.cosine_decay_schedule(init_value=0.01, decay_steps=total_steps, alpha=0.1)

    optimizer = numpyro.optim.optax_to_numpyro(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=scheduler, weight_decay=1e-4, eps=1e-7)
        )
    )
    svi = SVI(build_model_2d, guide, optimizer, loss=Trace_ELBO())
    
    print("Compiling model...")
    rng_key = jax.random.PRNGKey(41)
    
    # Initialize state
    svi_state = svi.init(rng_key, data=data_dict, anneal=anneal_epochs[0])
    all_losses = []

    # --- THE EPOCH LOOP ---
    # Using a Python loop here avoids the XLA memory explosion 
    # that happens when scanning over thousands of high-res steps at once.
    # --- THE EPOCH LOOP ---
    for anneal_level in anneal_epochs:
        desc = f"Epoch (Anneal={anneal_level})"
        epoch_losses = []
        
        # Wrap the range in tqdm and assign to pbar
        pbar = tqdm(range(steps_per_epoch), desc=desc)
        for i in pbar:
            svi_state, loss = svi.update(svi_state, data=data_dict, anneal=anneal_level)
            
            # Convert loss to float for the progress bar
            loss_val = float(loss)
            epoch_losses.append(loss_val)
            
            # Update the progress bar every 10 steps to keep the display snappy
            if i % 10 == 0:
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        all_losses.append(np.array(epoch_losses))

    losses = np.concatenate(all_losses)
    params = svi.get_params(svi_state)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "map_params.pkl"), 'wb') as f:
        pickle.dump(params, f)
        
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.title(f"MAP Optimization Loss - Epoch-Based Annealing ({PRECISION})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve_log.png"))
    plt.close()
    
    print(f"Final Loss: {losses[-1]:.4f}")

if __name__ == "__main__":
    run_map()
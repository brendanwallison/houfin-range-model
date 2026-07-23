"""Fit the age-structured model by MAP (maximum-a-posteriori) optimization.

Runs NumPyro's ``AutoDelta`` / gradient optimization over ``build_model_2d`` to
get a point estimate, typically as a fast first pass and as the initialization
for SVI (``age_resume_svi_from_map``) or HMC (``age_resume_hmc``). Writes the MAP
parameters to a pickle checkpoint. GPU-oriented; see the README for memory notes.

**Checkpointing / resume.** The optimizer state is checkpointed every
``HOUFIN_MAP_CKPT_EVERY`` steps (atomic write) to ``map_checkpoint.pkl``, alongside
the current best-effort ``map_params.pkl`` + loss curve. On startup, if a checkpoint
exists it is restored and the run continues from the saved global step -- so a job
killed by the 2 h dev-queue wall clock just needs to be resubmitted to pick up where
it left off. Total steps + checkpoint cadence are env-overridable:
``HOUFIN_MAP_STEPS`` (default 900), ``HOUFIN_MAP_CKPT_EVERY`` (default 100).
Set ``HOUFIN_MAP_FRESH=1`` to ignore an existing checkpoint and start over.
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

TOTAL_STEPS = int(os.environ.get("HOUFIN_MAP_STEPS", 900))
CKPT_EVERY = int(os.environ.get("HOUFIN_MAP_CKPT_EVERY", 100))
ANNEAL_EPOCHS = [0.1, 0.5, 1.0]
CKPT_PATH = os.path.join(OUTPUT_DIR, "map_checkpoint.pkl")
PARAMS_PATH = os.path.join(OUTPUT_DIR, "map_params.pkl")


def _anneal_for_step(g, steps_per_epoch):
    """Global step -> annealing level (clamped to the last epoch)."""
    return ANNEAL_EPOCHS[min(g // steps_per_epoch, len(ANNEAL_EPOCHS) - 1)]


def _save_atomic(obj, path):
    """Pickle to a temp file then os.replace -> a kill mid-write can't corrupt the checkpoint."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _write_artifacts(svi, svi_state, losses):
    """Persist the resume checkpoint + the human-usable params/loss-curve (all atomic)."""
    step = len(losses)
    _save_atomic({"svi_state": svi_state, "step": step, "losses": np.asarray(losses)}, CKPT_PATH)
    _save_atomic(svi.get_params(svi_state), PARAMS_PATH)
    if losses:
        plt.figure(figsize=(10, 6))
        plt.plot(losses); plt.yscale('log')
        plt.title(f"MAP Optimization Loss ({PRECISION}) -- {step}/{TOTAL_STEPS} steps")
        plt.xlabel("Iteration"); plt.ylabel("Loss (Log Scale)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve_log.png")); plt.close()


def run_map():
    print(f"--- Starting {PRECISION.upper()} MAP (Age-Structured), "
          f"{TOTAL_STEPS} steps, ckpt every {CKPT_EVERY} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_dict = load_data_to_gpu(INPUT_DIR, precision=PRECISION)

    guide = AutoDelta(build_model_2d)
    steps_per_epoch = TOTAL_STEPS // len(ANNEAL_EPOCHS)
    scheduler = optax.cosine_decay_schedule(init_value=0.01, decay_steps=TOTAL_STEPS, alpha=0.1)
    optimizer = numpyro.optim.optax_to_numpyro(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=scheduler, weight_decay=1e-4, eps=1e-7)
        )
    )
    svi = SVI(build_model_2d, guide, optimizer, loss=Trace_ELBO())

    # Resume from checkpoint if present (unless HOUFIN_MAP_FRESH=1).
    fresh = os.environ.get("HOUFIN_MAP_FRESH", "0") == "1"
    if os.path.exists(CKPT_PATH) and not fresh:
        with open(CKPT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        svi_state = ckpt["svi_state"]
        start_step = int(ckpt["step"])
        losses = list(np.asarray(ckpt["losses"]))
        print(f"[resume] restored checkpoint at step {start_step}/{TOTAL_STEPS}")
    else:
        print("Compiling model (fresh init)...")
        rng_key = jax.random.PRNGKey(41)
        svi_state = svi.init(rng_key, data=data_dict, anneal=ANNEAL_EPOCHS[0])
        start_step, losses = 0, []

    if start_step >= TOTAL_STEPS:
        print("[resume] checkpoint already at TOTAL_STEPS; nothing to do.")
        _write_artifacts(svi, svi_state, losses)
        return

    pbar = tqdm(range(start_step, TOTAL_STEPS), initial=start_step, total=TOTAL_STEPS, desc="MAP")
    for g in pbar:
        anneal_level = _anneal_for_step(g, steps_per_epoch)
        svi_state, loss = svi.update(svi_state, data=data_dict, anneal=anneal_level)
        loss_val = float(loss)
        losses.append(loss_val)
        if g % 10 == 0:
            pbar.set_postfix({"anneal": anneal_level, "loss": f"{loss_val:.4f}"})
        # Checkpoint on cadence and on the final step.
        if (g + 1) % CKPT_EVERY == 0 or (g + 1) == TOTAL_STEPS:
            _write_artifacts(svi, svi_state, losses)

    print(f"Final Loss: {losses[-1]:.4f}  ->  {PARAMS_PATH}")


if __name__ == "__main__":
    run_map()

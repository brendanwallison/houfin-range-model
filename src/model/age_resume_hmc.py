import sys
import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
import numpyro

from numpyro.infer import MCMC, HMC
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_value

# --- SINGLE POINT OF CONTROL ---
PRECISION = 'float32'

jax.config.update(
    "jax_enable_x64",
    True if PRECISION == 'float64' else False
)

# --- FIXED: preserve your working import resolution exactly ---
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')
)

if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.age_priors import build_model_2d
from src.model.data_loading import load_data_to_gpu
from src.config_utils import load_age_model_config


# --- CONFIGURATION ---
_cfg = load_age_model_config()
INPUT_DIR = _cfg["input_dir"]
MAP_DIR = os.path.join(_cfg["results_dir"], _cfg["run_names"]["resume_hmc_from_map"].format(precision=PRECISION))
OUTPUT_DIR = os.path.join(_cfg["results_dir"], _cfg["run_names"]["resume_hmc_out"].format(precision=PRECISION))


# OPTIONAL DIAGNOSTICS (non-invasive)
class HMCDiagnostics:
    def __init__(self):
        self.t0 = time.time()
        self.last = self.t0
        self.i = 0

    def __call__(self, step_result):
        # step_result is not always provided by numpyro internals;
        # this is safe to ignore if unused.
        now = time.time()
        dt = now - self.last
        self.last = now
        self.i += 1

        print(f"[HMC] iter={self.i} dt={dt:.2f}s")

        return step_result


def build_map_initialization(noise_scale=0.01):

    map_path = os.path.join(MAP_DIR, "map_params.pkl")

    with open(map_path, "rb") as f:
        raw_map_params = pickle.load(f)

    rng = jax.random.PRNGKey(123)

    noisy_init = {}

    for i, (name, value) in enumerate(raw_map_params.items()):

        site_name = name.replace("auto_", "")
        key = jax.random.fold_in(rng, i)

        noise = noise_scale * jax.random.normal(
            key,
            shape=value.shape,
            dtype=value.dtype
        )

        noisy_init[site_name] = value + noise

    return noisy_init


# MAIN RUN (UNCHANGED SIGNATURE)
def run_hmc():

    print(
        f"--- Starting {PRECISION.upper()} HMC from MAP Initialization ---"
    )

    data_dict = load_data_to_gpu(
        INPUT_DIR,
        precision=PRECISION
    )

    init_values = build_map_initialization(
        noise_scale=0.01
    )

    init_strategy = init_to_value(values=init_values)

    kernel = HMC(
        build_model_2d,

        step_size=1e-3,
        trajectory_length=0.1,

        adapt_step_size=True,
        adapt_mass_matrix=False,
        dense_mass=False,

        init_strategy=init_strategy,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=20,
        num_samples=20,
        num_chains=1,
        progress_bar=True,
    )

    rng_key = jax.random.PRNGKey(41)

    print("Compiling + running HMC...")

    start = time.time()

    # IMPORTANT: no extra args, no diagnostics injection
    mcmc.run(
        rng_key,
        data=data_dict,
        anneal=1.0,
    )

    elapsed = time.time() - start

    print("\n--- TIMING ---")
    print(f"Total runtime: {elapsed:.2f} sec")
    print(f"Seconds/sample: {elapsed / 20:.2f}")

    print("\n--- SUMMARY ---")
    mcmc.print_summary()

    extra = mcmc.get_extra_fields()

    if "num_steps" in extra:
        avg_steps = np.mean(np.array(extra["num_steps"]))
        print(f"\nAverage leapfrog steps: {avg_steps:.2f}")

    samples = mcmc.get_samples()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(
        os.path.join(OUTPUT_DIR, "samples.pkl"),
        "wb"
    ) as f:
        pickle.dump(samples, f)

    print(f"\nSaved samples to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    run_hmc()
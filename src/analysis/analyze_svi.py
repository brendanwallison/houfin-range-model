import os
import sys
import glob
import numpy as np
import jax

# --- Modular Imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.age_priors import build_model_2d
from src.model.data_loading import load_data_to_gpu
from src.analysis.engine import load_params, reconstruct_latents
from src.analysis.plots import (
    plot_posterior_weights, 
    plot_demographic_response_curves,
    plot_temporal_epochs,
    plot_continental_violins,
    plot_keystone_r0,
    create_animation
)

# --- Configuration ---
PRECISION = 'float32'
jax.config.update("jax_enable_x64", False)

from src.config_utils import load_age_model_config
_cfg = load_age_model_config()
VI_RESULT_DIR = os.path.join(_cfg["results_dir"], _cfg["run_names"]["resume_svi_out"].format(precision=PRECISION))
INPUT_DIR = _cfg["input_dir"]
OUTPUT_DIR = os.path.join(VI_RESULT_DIR, "analysis_plots")
CACHE_FILE = os.path.join(OUTPUT_DIR, "reconstructed_samples.npz")

def run_full_svi_analysis():
    print(f"--- Starting SVI Analysis Suite ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Environmental Data
    data = load_data_to_gpu(INPUT_DIR, precision=PRECISION)
    
    # 2. Setup Labels safely matching the actual input data dimension
    actual_M = data['Z_gathered'].shape[-1]  # Configured model width (default 16 of source 64)
    
    PATH_INTEGRATION_DIR = _cfg["path_diagnostics_dir"]
    disp_files = glob.glob(os.path.join(PATH_INTEGRATION_DIR, "Z_disp_*.npz"))
    
    loaded_labels = []
    if disp_files:
        with np.load(disp_files[0]) as loader:
            if 'labels' in loader:
                loaded_labels = [str(lbl) for lbl in loader['labels']]

    # Align loaded labels with actual dimension size
    if len(loaded_labels) == actual_M:
        z_names = loaded_labels
    elif len(loaded_labels) < actual_M:
        print(f"--> Warning: Found {len(loaded_labels)} cached labels, but data has {actual_M} features.")
        print("--> Appending generic placeholders for missing labels.")
        # Keep the 12 names you have, fill the remaining 4 with generic tokens
        extra_needed = actual_M - len(loaded_labels)
        z_names = loaded_labels + [f"Feature_{i + len(loaded_labels)}" for i in range(extra_needed)]
    else:
        # If labels are somehow longer than current data, slice them down
        z_names = loaded_labels[:actual_M]

    # 3. Cache Logic: Load existing samples or generate new ones
    if os.path.exists(CACHE_FILE):
        print(f"-> Loading cached samples from {CACHE_FILE}...")
        with np.load(CACHE_FILE) as loader:
            samples = {k: loader[k] for k in loader.files}
    else:
        params, filename = load_params(VI_RESULT_DIR)
        samples = reconstruct_latents(
            build_model_2d, 
            data, 
            params, 
            filename, 
            num_samples=250
        )
        # Immediate save to prevent lost work
        np.savez_compressed(CACHE_FILE, **samples)
        print(f"-> Samples cached to disk.")

    # 4. Generate the Visualization Suite
    print("-> Plotting Weights and Demographic Responses...")
    plot_posterior_weights(samples, z_names, OUTPUT_DIR)
    
    for i in range(3):
        plot_demographic_response_curves(samples, data, z_names, i, OUTPUT_DIR)

    print("-> Generating Spatial Maps (Epochs & Keystones)...")
    plot_temporal_epochs(samples, data, z_names, OUTPUT_DIR)
    plot_keystone_r0(samples, data, z_names, OUTPUT_DIR)

    print("-> Generating Continental Distributions...")
    plot_continental_violins(samples, data, z_names, OUTPUT_DIR)

    print("-> Creating Population Evolution Animation...")
    # Consider num_frames=data['time'] if create_animation is slow
    create_animation(samples, data, OUTPUT_DIR)

    print(f"\nAnalysis Complete. Results at: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_full_svi_analysis()

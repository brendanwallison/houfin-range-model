"""Hash actual prepared inputs and quantify centering support; no model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from harness import Sandbox
from src.config_utils import load_age_model_config


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cfg = load_age_model_config()
    root = Path(cfg["input_dir"])
    with (root / "metadata.pkl").open("rb") as stream:
        meta = pickle.load(stream)
    print(json.dumps({"state": "hashing_input_files"}), flush=True)
    files = []
    for rel in ("metadata.pkl", meta["z_gathered_path"], meta["z_disp_gathered_path"]):
        path = root / rel
        files.append({"name": rel, "resolved_path": str(path.resolve()), "bytes": path.stat().st_size,
                      "sha256": digest_file(path)})
    print(json.dumps({"state": "checking_center_support"}), flush=True)
    sb = Sandbox(os.environ["SBX_REFINED"])
    data = sb.data
    obs_t, obs_r, obs_c = [np.asarray(data[key]) for key in ("obs_time_indices", "obs_rows", "obs_cols")]
    cells = sb.lookup[obs_r, obs_c]
    positives = (np.asarray(data["observed_results"]) > 0) & (cells >= 0)
    fit = np.asarray(data.get("obs_weight", np.ones(len(cells)))) > 0
    z = np.asarray(data["Z_gathered"])
    centers, counts = {}, {}
    for name, support in (("all_positive", positives), ("training_positive", positives & fit)):
        pairs = np.unique(np.stack([obs_t[support], cells[support]], axis=1), axis=0)
        centers[name] = np.asarray(z[pairs[:, 0], pairs[:, 1]], np.float64).mean(0)
        counts[name] = len(pairs)
    used = np.asarray(data["habitat_center_occupied"])
    result = {"files": files, "kernel_contract": meta.get("z_kernel_contract"),
              "dispersal_spec": meta.get("dispersal_spec"),
              "input_time": int(meta["time"]), "input_land_cells": int(meta["N_land"]),
              "input_features": int(meta["M"]), "input_kernel_count": int(meta["K"]),
              "initialization_year": int(sb.years[int(data["inv_timestep"])]),
              "centering_support_counts": counts,
              "used_center_error_vs_all_positive": float(np.max(np.abs(used - centers["all_positive"]))),
              "used_center_error_vs_training_positive": float(np.max(np.abs(used - centers["training_positive"]))),
              "center_difference_l2": float(np.linalg.norm(centers["all_positive"] - centers["training_positive"])),
              "seconds": time.time() - started}
    (args.out / "input_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("files", "dispersal_spec", "kernel_contract")}), flush=True)


if __name__ == "__main__":
    main()

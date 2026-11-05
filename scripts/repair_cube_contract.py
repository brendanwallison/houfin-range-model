"""Repair cube metadata after checking arrays and checkpoint identity; no encoding.

Defaults to dry run. --write preserves the old metadata as cube_meta.json.pre-contract-v2
and atomically installs the correction. Run in the same environment as the original export.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config_utils import load_config
from src.community_encoder.cube_contract import export_contract, add_basis_identities


def identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"name": path.name, "size": path.stat().st_size, "content_sha256": h.hexdigest()}


def repair(config, write=False):
    cfg = load_config(config)
    cube = cfg["latent_cube"]
    directory = Path(cube["output_dir"])
    meta_path = directory / "cube_meta.json"
    old = json.loads(meta_path.read_text())
    checkpoint = identity(cube["model_path"])
    dm_path = Path(cfg["paths"]["desk_output_dir"]) / "desk_meta.npz"
    if old.get("desk_checkpoint") != checkpoint or old.get("desk_meta") != identity(dm_path):
        raise ValueError("cube checkpoint/desk_meta identity differs from supplied config")
    with np.load(dm_path, allow_pickle=False) as dm:
        width = int(dm["latent_dim"])
        private = int(dm["private_dim"]) if cube.get("include_private") else 0
        if os.path.normpath(str(dm["esk_basis_dir"])) != os.path.normpath(cube["z_dir"]):
            raise ValueError("checkpoint basis differs from export basis")
    contract = export_contract(width, cube["z_dir"], private,
                               cfg.get("desk", {}).get("aux_metric_heads", []))
    shape = None
    files = sorted(directory.glob("Z_latent_*.npy"))
    years = [int(f.stem.removeprefix("Z_latent_")) for f in files]
    if not years or years != old["years"]:
        raise ValueError("cube file timeline differs from metadata")
    for f in files:
        a = np.load(f, mmap_mode="r")
        if a.ndim != 3 or a.shape[-1] != contract["latent_dim"]:
            raise ValueError(f"array shape disagrees with corrected contract: {f}: {a.shape}")
        shape = a.shape if shape is None else shape
        if a.shape != shape:
            raise ValueError("cube grids differ by year")
    add_basis_identities(contract)
    corrected = {**old, **contract, "contract_repair": {
        "previous_metadata_sha256": identity(meta_path)["content_sha256"],
        "config": str(config), "array_shape": list(shape), "arrays_checked": len(files)}}
    if all(old.get(k) == v for k, v in contract.items()):
        print(f"already correct: {directory}")
        return old
    if write:
        backup = meta_path.with_name("cube_meta.json.pre-contract-v2")
        if backup.exists():
            raise ValueError(f"backup already exists: {backup}; refusing to overwrite history")
        backup.write_bytes(meta_path.read_bytes())
        temp = meta_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(corrected, indent=2, allow_nan=False) + "\n")
        temp.replace(meta_path)
    print(json.dumps({"cube": str(directory), "write": write,
                      "kernel": contract["kernel"], "width": contract["latent_dim"],
                      "arrays_checked": len(files)}))
    return corrected


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--write", action="store_true")
    a = p.parse_args()
    repair(a.config, a.write)

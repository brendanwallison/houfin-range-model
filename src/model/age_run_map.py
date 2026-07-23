"""Checkpointed GPU MAP fit for the age-structured range model.

The fit uses prior continuation: tight priors stabilize early optimization,
then relax at fixed *absolute* optimizer steps. This is not simulated annealing.
Changing ``HOUFIN_MAP_STEPS`` therefore extends a run without changing either
its prior schedule or its learning-rate history.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import optax
from tqdm import tqdm

PRECISION = os.environ.get("HOUFIN_MODEL_PRECISION", "float32")
if PRECISION not in {"float32", "float64"}:
    raise ValueError("HOUFIN_MODEL_PRECISION must be float32 or float64")
jax.config.update("jax_enable_x64", PRECISION == "float64")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config_utils import load_age_model_config
from src.model.age_priors import build_model_2d
from src.model.data_loading import load_data
from src.model.runtime_diagnostics import memory_snapshot, require_gpu

_cfg = load_age_model_config()
_map_cfg = _cfg["map"]
MAP_PROFILE = os.environ.get("HOUFIN_MAP_PROFILE", "standard")
if MAP_PROFILE == "standard":
    _active_map_cfg = _map_cfg
else:
    try:
        _active_map_cfg = _map_cfg["profiles"][MAP_PROFILE]
    except KeyError as exc:
        choices = ", ".join(["standard", *_map_cfg.get("profiles", {}).keys()])
        raise ValueError(
            f"Unknown HOUFIN_MAP_PROFILE={MAP_PROFILE!r}; choose one of: {choices}"
        ) from exc
os.environ.setdefault(
    "HOUFIN_MODEL_INPUT_RESIDENCY",
    _cfg.get("runtime", {}).get("input_residency", "device"),
)
INPUT_DIR = _cfg["input_dir"]
_run_name = _cfg["run_names"]["map"].format(precision=PRECISION)
# Optimization profiles are separate experiments, never checkpoints that a
# standard production run could accidentally resume.
if MAP_PROFILE != "standard":
    _run_name = f"{_run_name}_{MAP_PROFILE}"
OUTPUT_DIR = os.path.join(_cfg["results_dir"], _run_name)
TOTAL_STEPS = int(
    os.environ.get("HOUFIN_MAP_STEPS", _active_map_cfg.get("target_steps", 900))
)
CKPT_EVERY = int(os.environ.get("HOUFIN_MAP_CKPT_EVERY", 100))
PRIOR_RELAXATION = tuple(
    (int(step), float(scale)) for step, scale in _active_map_cfg["prior_relaxation"]
)
LR_DECAY_STEPS = int(_active_map_cfg["lr_decay_steps"])
INIT_LR = float(_active_map_cfg.get("init_lr", _map_cfg["init_lr"]))
WEIGHT_DECAY = float(_active_map_cfg.get("weight_decay", _map_cfg.get("weight_decay", 0.0)))
CKPT_PATH = os.path.join(OUTPUT_DIR, "map_checkpoint.pkl")
PARAMS_PATH = os.path.join(OUTPUT_DIR, "map_params.pkl")


def _validate_schedule():
    starts = [x[0] for x in PRIOR_RELAXATION]
    scales = [x[1] for x in PRIOR_RELAXATION]
    if not starts or starts[0] != 0 or starts != sorted(set(starts)):
        raise ValueError("map.prior_relaxation must have unique ascending steps starting at 0")
    if any(x <= 0 for x in scales) or scales != sorted(scales):
        raise ValueError("map.prior_relaxation scales must be positive and nondecreasing")


def _prior_scale_for_step(global_step: int) -> float:
    """Return the continuation scale at an absolute optimizer step."""
    return next(
        scale
        for (start, scale), (end, _) in zip(
            PRIOR_RELAXATION, PRIOR_RELAXATION[1:] + ((10**30, 0.0),)
        )
        if start <= global_step < end
    )


def _file_identity(path: Path) -> dict:
    """Cheap but content-sensitive identity for large and small model inputs."""
    stat = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        if stat.st_size <= 64 * 1024 * 1024:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        else:
            head = f.read(1024 * 1024)
            h.update(head)
            f.seek(max(0, stat.st_size - 1024 * 1024))
            h.update(f.read(1024 * 1024))
    return {"name": path.name, "size": stat.st_size, "content_sha256": h.hexdigest()}


def _run_fingerprint() -> tuple[str, dict]:
    """Fingerprint state-defining code/config/data while allowing step extensions."""
    input_dir = Path(INPUT_DIR)
    metadata_path = input_dir / "metadata.pkl"
    with metadata_path.open("rb") as fh:
        metadata = pickle.load(fh)
    input_files = [
        metadata_path,
        input_dir / metadata["z_gathered_path"],
        input_dir / metadata["z_disp_gathered_path"],
    ]
    source_files = [
        Path(__file__),
        Path(__file__).with_name("age_priors.py"),
        Path(__file__).with_name("age_fields.py"),
        Path(__file__).with_name("age_forward.py"),
        Path(__file__).with_name("build_kernels.py"),
        Path(__file__).with_name("data_loading.py"),
    ]
    payload = {
        "map_profile": MAP_PROFILE,
        "precision": PRECISION,
        "age_model_config": _cfg,
        "prior_relaxation": PRIOR_RELAXATION,
        "lr_decay_steps": LR_DECAY_STEPS,
        "init_lr": INIT_LR,
        "weight_decay": WEIGHT_DECAY,
        "input_residency": os.environ.get("HOUFIN_MODEL_INPUT_RESIDENCY", "device"),
        "versions": {
            "jax": jax.__version__,
            "numpyro": numpyro.__version__,
            "optax": optax.__version__,
        },
        "inputs": [_file_identity(p) for p in input_files],
        "sources": [_file_identity(p) for p in source_files],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(canonical.encode()).hexdigest(), payload


def _save_atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_artifacts(svi, svi_state, losses, fingerprint, fingerprint_payload, device):
    step = len(losses)
    params = svi.get_params(svi_state)
    checkpoint = {
        "format_version": 2,
        "svi_state": svi_state,
        "step": step,
        "losses": np.asarray(losses),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "prior_relaxation": PRIOR_RELAXATION,
        "lr_decay_steps": LR_DECAY_STEPS,
        "precision": PRECISION,
        "memory": memory_snapshot(f"checkpoint-{step}", device),
        "params": params,
    }
    _save_atomic(checkpoint, CKPT_PATH)
    _save_atomic(params, PARAMS_PATH)
    if losses:
        fig_path = os.path.join(OUTPUT_DIR, "loss_curve_log.png")
        tmp_fig = fig_path + ".tmp"
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.yscale("log")
        plt.title(f"MAP Optimization Loss ({PRECISION}) — {step}/{TOTAL_STEPS} steps")
        plt.xlabel("Iteration")
        plt.ylabel("Loss (log scale)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(tmp_fig, format="png")
        plt.close()
        os.replace(tmp_fig, fig_path)


def run_map():
    _validate_schedule()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = require_gpu("age-model MAP")
    fingerprint, fingerprint_payload = _run_fingerprint()
    print(
        f"--- Starting {PRECISION.upper()} MAP ({MAP_PROFILE}): target={TOTAL_STEPS}, "
        f"checkpoint_every={CKPT_EVERY}, prior_relaxation={PRIOR_RELAXATION}, "
        f"lr_decay_steps={LR_DECAY_STEPS}, weight_decay={WEIGHT_DECAY} ---"
    )
    data_dict = load_data(INPUT_DIR, target_device=device, precision=PRECISION)
    memory_snapshot("inputs-loaded", device)

    guide = AutoDelta(build_model_2d)
    scheduler = optax.cosine_decay_schedule(
        init_value=INIT_LR, decay_steps=LR_DECAY_STEPS, alpha=0.1
    )
    base_optimizer = (
        optax.adamw(scheduler, weight_decay=WEIGHT_DECAY, eps=1e-7)
        if WEIGHT_DECAY
        else optax.adam(scheduler, eps=1e-7)
    )
    optimizer = numpyro.optim.optax_to_numpyro(
        optax.chain(optax.clip_by_global_norm(1.0), base_optimizer)
    )
    svi = SVI(build_model_2d, guide, optimizer, loss=Trace_ELBO())

    fresh = os.environ.get("HOUFIN_MAP_FRESH", "0") == "1"
    if fresh and os.path.exists(CKPT_PATH):
        stamp = time.strftime("%Y%m%dT%H%M%S")
        archived = f"{CKPT_PATH}.before_fresh_{stamp}"
        os.replace(CKPT_PATH, archived)
        if os.path.exists(PARAMS_PATH):
            os.replace(PARAMS_PATH, f"{PARAMS_PATH}.before_fresh_{stamp}")
        print(f"[fresh] archived prior checkpoint -> {archived}")
    if os.path.exists(CKPT_PATH) and not fresh:
        with open(CKPT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        saved = ckpt.get("fingerprint")
        if saved != fingerprint:
            reason = "legacy checkpoint has no fingerprint" if saved is None else "run fingerprint changed"
            raise RuntimeError(
                f"Refusing incompatible MAP resume: {reason}. "
                "Use HOUFIN_MAP_FRESH=1 to deliberately start a new optimizer state."
            )
        svi_state = ckpt["svi_state"]
        start_step = int(ckpt["step"])
        losses = list(np.asarray(ckpt["losses"]))
        if start_step != len(losses):
            raise RuntimeError("checkpoint step/loss history mismatch")
        print(f"[resume] compatible checkpoint at absolute step {start_step}/{TOTAL_STEPS}")
    else:
        print("Compiling model (fresh optimizer state)...")
        svi_state = svi.init(
            jax.random.PRNGKey(41),
            data=data_dict,
            prior_scale=_prior_scale_for_step(0),
        )
        start_step, losses = 0, []

    if start_step >= TOTAL_STEPS:
        print("[resume] checkpoint already meets target; no optimizer steps required.")
        _write_artifacts(
            svi, svi_state, losses, fingerprint, fingerprint_payload, device
        )
        return

    pbar = tqdm(
        range(start_step, TOTAL_STEPS),
        initial=start_step,
        total=TOTAL_STEPS,
        desc="MAP",
    )
    for global_step in pbar:
        prior_scale = _prior_scale_for_step(global_step)
        next_state, loss = svi.update(
            svi_state, data=data_dict, prior_scale=prior_scale
        )
        loss_val = float(loss)
        if not np.isfinite(loss_val):
            _write_artifacts(
                svi, svi_state, losses, fingerprint, fingerprint_payload, device
            )
            raise FloatingPointError(f"non-finite MAP loss at step {global_step}: {loss_val}")
        svi_state = next_state
        losses.append(loss_val)
        if global_step % 10 == 0:
            pbar.set_postfix(
                {"prior_scale": prior_scale, "loss": f"{loss_val:.4f}"}
            )
        if (global_step + 1) % CKPT_EVERY == 0 or global_step + 1 == TOTAL_STEPS:
            _write_artifacts(
                svi, svi_state, losses, fingerprint, fingerprint_payload, device
            )

    print(f"Final loss: {losses[-1]:.4f} -> {PARAMS_PATH}")


if __name__ == "__main__":
    run_map()

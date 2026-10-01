"""Stage 1 of the NeuTra arm: a low-rank MVN variational fit initialized at the MAP.

Loads the MAP checkpoint (always age_run_map's own OUTPUT_DIR, so it cannot point at a
stale run), places the guide's mean there with a small initial scale, and fits under
the NOMINAL priors (prior_scale=1). Stage 2 (``age_run_hmc``) uses the fitted guide as
a NeuTra reparameterization for NUTS.

Checkpointed per ``block`` steps (a ``lax.scan``); resumes automatically, and refuses a
checkpoint whose fingerprint (MAP, settings, code, versions) differs unless
``HOUFIN_VI_FRESH=1``, which archives it first.

Output (``results_dir/<run_names.vi>``): ``vi_checkpoint.pkl`` (optimizer state) and
``vi_posterior_params.pkl`` (format_version 2: constrained guide params + rank + the
MAP fingerprint and payload, which stage 2 checks).
"""
from __future__ import annotations

import os
import pickle
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import hmc_common as hc                            # noqa: E402  (sets x64)
import jax                                                         # noqa: E402
import jax.numpy as jnp                                            # noqa: E402
import numpy as np                                                 # noqa: E402
import numpyro                                                     # noqa: E402
import optax                                                       # noqa: E402
from numpyro.infer import SVI, Trace_ELBO                          # noqa: E402
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal  # noqa: E402
from numpyro.infer.initialization import init_to_value            # noqa: E402

from src.config_utils import load_age_model_config                # noqa: E402
from src.model.age_priors import build_model_2d                    # noqa: E402
from src.model.checkpoints import save_pickle_atomic               # noqa: E402
from src.model.data_loading import load_data                       # noqa: E402
from src.model.runtime_diagnostics import memory_snapshot, require_gpu  # noqa: E402


# The VI block is a jitted lax.scan, so the model must survive tracing (hc.jit_safe).
MODEL = hc.jit_safe(build_model_2d)


def vi_settings(pcfg: dict) -> dict:
    v = pcfg["vi"]
    return {
        "rank": hc.setting(v, "rank", "HOUFIN_VI_RANK", int),
        "steps": hc.setting(v, "steps", "HOUFIN_VI_STEPS", int),
        "block": hc.setting(v, "block", "HOUFIN_VI_BLOCK", int),
        "init_lr": hc.setting(v, "init_lr", "HOUFIN_VI_LR", float),
        "init_scale": hc.setting(v, "init_scale", "HOUFIN_VI_INIT_SCALE", float),
        "num_particles": hc.setting(v, "num_particles", "HOUFIN_VI_PARTICLES", int),
    }


def vi_dir(pcfg: dict, rank: int) -> str:
    return hc.posterior_dir(pcfg, "vi", rank=rank)


def make_guide(rank: int, init_values=None, init_scale: float = 0.01):
    kw = {} if init_values is None else {"init_loc_fn": init_to_value(values=init_values)}
    return AutoLowRankMultivariateNormal(MODEL, rank=rank, init_scale=init_scale, **kw)


def run_vi_resume():
    pcfg = hc.load_posterior_config()
    s = vi_settings(pcfg)
    if s["steps"] % s["block"]:
        raise ValueError("vi.steps must be a multiple of vi.block")
    out_dir = vi_dir(pcfg, s["rank"])
    hc.archive_if_fresh(out_dir, env="HOUFIN_VI_FRESH", files=("vi_checkpoint.pkl",))
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "vi_checkpoint.pkl")
    print(f"--- Low-rank VI from MAP | {hc.PRECISION} | {s} -> {out_dir} ---")

    device = require_gpu("MAP-initialized SVI")
    map_latents, map_ckpt = hc.load_map()
    hc.check_model_drift(map_ckpt["fingerprint_payload"])
    data = load_data(load_age_model_config()["input_dir"], target_device=device,
                     precision=hc.PRECISION)
    memory_snapshot("vi-inputs-loaded", device)

    guide = make_guide(s["rank"], map_latents, s["init_scale"])
    scheduler = optax.cosine_decay_schedule(init_value=s["init_lr"], decay_steps=s["steps"],
                                            alpha=0.1)
    optimizer = numpyro.optim.optax_to_numpyro(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(scheduler, eps=1e-7)))
    svi = SVI(MODEL, guide, optimizer, loss=Trace_ELBO(num_particles=s["num_particles"]))
    kw = hc.model_kwargs(data)

    payload = {"settings": s, "precision": hc.PRECISION, "map_fingerprint": map_ckpt["fingerprint"],
               "versions": hc.versions(), "sources": hc.source_identities([__file__])}
    run_fp = hc.fingerprint(payload)

    # init() is needed even on resume: numpyro keeps constrain_fn on the SVI object, not
    # in SVIState, so a restored state alone dies on the first update (see age_run_map
    # and tests/test_map_resume.py). The legacy version of this script lacked this.
    fresh_state = svi.init(jax.random.PRNGKey(0), **kw)
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as fh:
            ckpt = pickle.load(fh)
        if ckpt.get("fingerprint") != run_fp:
            raise RuntimeError("Refusing incompatible VI resume (fingerprint changed); "
                               "use HOUFIN_VI_FRESH=1 to archive it and start over.")
        svi_state, start, losses = ckpt["svi_state"], int(ckpt["step"]), list(ckpt["losses"])
        print(f"[resume] VI at {start}/{s['steps']}")
    else:
        svi_state, start, losses = fresh_state, 0, []

    def body(state, _):
        return svi.update(state, **kw)

    run_block = jax.jit(lambda st: jax.lax.scan(body, st, None, length=s["block"]))

    def save(state, step):
        save_pickle_atomic({"format_version": 3, "svi_state": state, "step": step,
                            "losses": np.asarray(losses), "fingerprint": run_fp,
                            "payload": payload}, ckpt_path)

    t_start = time.time()
    for block_start in range(start, s["steps"], s["block"]):
        t0 = time.time()
        new_state, block_losses = run_block(svi_state)
        block_losses = np.asarray(block_losses)
        if not np.isfinite(block_losses).all():
            save(svi_state, block_start)
            bad = int(np.argmin(np.isfinite(block_losses)))
            raise FloatingPointError(f"non-finite ELBO at step {block_start + bad}; "
                                     f"checkpoint kept at {block_start}")
        svi_state = new_state
        losses.extend(block_losses.tolist())
        step = block_start + s["block"]
        params = svi.get_params(svi_state)
        # auto_scale is already constrained (positive) in get_params; the legacy script
        # exp()'d it a second time.
        sd = np.asarray(params["auto_scale"])
        cf = np.asarray(params["auto_cov_factor"])
        print(f"[vi] step {step}/{s['steps']} -ELBO mean={block_losses.mean():.4f} "
              f"last={block_losses[-1]:.4f} diag sd median={np.median(sd):.3g} "
              f"max={sd.max():.3g} |cov_factor|={np.linalg.norm(cf):.3g} "
              f"{s['block'] / (time.time() - t0):.2f} step/s", flush=True)
        save(svi_state, step)
        if step == start + s["block"]:
            memory_snapshot(f"vi-{step}", device)

    params = jax.device_get(svi.get_params(svi_state))
    save_pickle_atomic(
        {"format_version": 2, "params": params, "rank": s["rank"], "step": s["steps"],
         "losses": np.asarray(losses), "fingerprint": run_fp,
         "map_fingerprint": map_ckpt["fingerprint"],
         "map_fingerprint_payload": map_ckpt["fingerprint_payload"]},
        os.path.join(out_dir, "vi_posterior_params.pkl"))
    print(f"VI complete in {(time.time() - t_start) / 60:.1f} min this job; final -ELBO "
          f"{losses[-1]:.4f} -> {out_dir}/vi_posterior_params.pkl")


if __name__ == "__main__":
    run_vi_resume()

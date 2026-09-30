"""Arm A of the HMC trial: NUTS started directly at the MAP point.

One chain per process. Run several via a SLURM array (``HOUFIN_HMC_CHAIN`` /
``SLURM_ARRAY_TASK_ID``); R-hat is computed across them afterwards
(``scripts/diagnostics/compare_hmc_trials.py``). Chain 0 starts exactly at MAP; the rest
start at MAP plus a Laplace-covariance draw (scaled by ``init_jitter``) in unconstrained
space, so disagreement between chains is informative.

``HOUFIN_HMC_METRIC`` picks the mass matrix -- the thing this arm is testing:

* ``diag``           diagonal, adapted from identity during warmup (the baseline);
* ``laplace_fixed``  dense Laplace inverse mass from the probe, never adapted --
                     warmup only tunes the step size;
* ``laplace_adapt``  dense Laplace as the INITIAL metric, then numpyro's windowed dense
                     adaptation replaces it. With d in the hundreds and a few hundred
                     warmup draws the sample covariance is poorly determined, so expect
                     this to be no better than laplace_fixed unless warmup is long.

Sampling is chunked and resumable (``hmc_common.run_chunked_mcmc``); chain it across
wall-clock windows with ``RESUBMITS`` in ``scripts/tacc/submit_hmc.sh``.
"""
from __future__ import annotations

import hashlib
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import hmc_common as hc                            # noqa: E402  (sets x64)
import jax                                                         # noqa: E402
import jax.numpy as jnp                                            # noqa: E402
import numpy as np                                                 # noqa: E402
from numpyro.infer import NUTS                                     # noqa: E402

from src.config_utils import load_age_model_config                # noqa: E402
from src.model.age_priors import build_model_2d                    # noqa: E402
from src.model.data_loading import load_data                       # noqa: E402
from src.model.runtime_diagnostics import memory_snapshot, require_gpu  # noqa: E402

METRICS = ("diag", "laplace_fixed", "laplace_adapt")


def hmc_settings(pcfg: dict, metric: str) -> dict:
    h = pcfg["hmc"]
    return {
        "num_warmup": hc.setting(h, "num_warmup", "HOUFIN_HMC_WARMUP", int),
        "num_samples": hc.setting(h, "num_samples", "HOUFIN_HMC_SAMPLES", int),
        "chunk": hc.setting(h, "chunk", "HOUFIN_HMC_CHUNK", int),
        "max_tree_depth": hc.setting(h, "max_tree_depth", "HOUFIN_HMC_MAX_TREE_DEPTH", int),
        "target_accept": hc.setting(h, "target_accept", "HOUFIN_HMC_TARGET_ACCEPT", float),
        "init_jitter": hc.setting(h, "init_jitter", "HOUFIN_HMC_INIT_JITTER", float),
        "step_size": float(os.environ.get("HOUFIN_HMC_STEP_SIZE", h["initial_step_size"][metric])),
    }


def run_hmc():
    metric = os.environ.get("HOUFIN_HMC_METRIC", "laplace_fixed")
    if metric not in METRICS:
        raise ValueError(f"HOUFIN_HMC_METRIC must be one of {METRICS}")
    chain = hc.chain_index()
    pcfg = hc.load_posterior_config()
    s = hmc_settings(pcfg, metric)
    variant = f"map_{metric}"
    out_dir = os.path.join(hc.posterior_dir(pcfg, "hmc", variant=variant), f"chain_{chain:02d}")
    hc.archive_if_fresh(out_dir)
    print(f"--- NUTS from MAP | metric={metric} chain={chain} {hc.PRECISION} | {s} ---")

    device = require_gpu("age-model NUTS from MAP")
    map_latents, map_ckpt = hc.load_map()
    hc.check_model_drift(map_ckpt["fingerprint_payload"])
    data = load_data(load_age_model_config()["input_dir"], target_device=device,
                     precision=hc.PRECISION)
    memory_snapshot("hmc-inputs-loaded", device)

    model = hc.hide_deterministics(build_model_2d)
    z_map, _, _ = hc.unconstrained_setup(model, data, map_latents)
    names, x_map, _ = hc.flatten_sorted(z_map)

    probe_dir = hc.posterior_dir(pcfg, "probe")
    laplace, laplace_id = None, None
    need_laplace = metric != "diag" or (chain > 0 and s["init_jitter"] > 0)
    if need_laplace:
        laplace = hc.load_laplace(probe_dir)
        if str(laplace["map_fingerprint"]) != map_ckpt["fingerprint"]:
            raise RuntimeError("probe laplace.npz was computed at a different MAP; rerun the probe")
        if tuple(laplace["site_names"].tolist()) != names:
            raise RuntimeError("probe site layout differs from the current model; rerun the probe")
        laplace_id = hashlib.sha256(np.ascontiguousarray(laplace["inv_mass"]).tobytes()).hexdigest()

    kernel_kw = dict(target_accept_prob=s["target_accept"], max_tree_depth=s["max_tree_depth"],
                     step_size=s["step_size"])
    if metric == "diag":
        kernel_kw.update(dense_mass=False, adapt_mass_matrix=True)
    else:
        inv_mass = jnp.asarray(laplace["inv_mass"], dtype=x_map.dtype)
        kernel_kw.update(dense_mass=True, inverse_mass_matrix={names: inv_mass},
                         adapt_mass_matrix=(metric == "laplace_adapt"))
    kernel = NUTS(model, **kernel_kw)

    rng = jax.random.PRNGKey(1000 + chain)
    rng_jitter, rng_run = jax.random.split(rng)
    init = z_map if chain == 0 else hc.jitter_unconstrained(
        z_map, rng_jitter, s["init_jitter"], None if laplace is None else laplace["inv_mass"])

    payload = {
        "arm": "map", "metric": metric, "chain": chain, "precision": hc.PRECISION,
        "settings": s, "map_fingerprint": map_ckpt["fingerprint"], "laplace": laplace_id,
        "versions": hc.versions(), "sources": hc.source_identities([__file__]),
    }
    hc.run_chunked_mcmc(
        kernel, out_dir=out_dir, run_fingerprint=hc.fingerprint(payload), payload=payload,
        num_warmup=s["num_warmup"], num_samples=s["num_samples"], chunk=s["chunk"],
        rng_key=rng_run, kwargs=hc.model_kwargs(data), init_params=init,
    )
    memory_snapshot("hmc-done", device)


if __name__ == "__main__":
    run_hmc()

"""Load the 54 km input with the FIT code that will run (GRID54_LADDER snapshot) and evaluate the model's
log density once at the init point. No optimization.

Exercises, in the snapshot: load_data (require_inputs_match_config on the dispersal spec, readout rank,
observation_mask incl. holdout_blocks.pinned_blocks, occupied habitat centre), build_model_2d's contract
checks, the forward simulation with the stored cell-integrated kernels, and the likelihood.

    PYTHONPATH=<snapshot>:<snapshot>/src AGE_MODEL_CONFIG=config/overlays/grid54_age.json \
        python research/exp/grid54_logdensity_check.py [--device cpu|gpu]
"""
import argparse
import json
import os
import time

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="/work/07980/bwa386/ls6/houfin/processed/grid54/grid54_logdensity_check.json")
a = ap.parse_args()
if a.device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import numpy as np  # noqa: E402
from numpyro.infer.util import initialize_model, log_density  # noqa: E402
from numpyro.infer import init_to_median, init_to_sample  # noqa: E402

import src.model.data_loading as dl  # noqa: E402
from src.config_utils import load_age_model_config  # noqa: E402
from src.model.age_priors import build_model_2d  # noqa: E402

cfg = load_age_model_config()
dev = jax.devices()[0]
t0 = time.time()
data = dl.load_data(cfg["input_dir"], target_device=dev, precision="float32")
res = {"snapshot": dl.__file__, "input_dir": cfg["input_dir"], "device": str(dev),
       "load_s": round(time.time() - t0, 1),
       "kernel_discretization_in_inputs": None, "holdout_withheld": None}
import pickle  # noqa: E402
with open(os.path.join(cfg["input_dir"], "metadata.pkl"), "rb") as fh:
    meta = pickle.load(fh)
res["kernel_discretization_in_inputs"] = meta.get("kernel_discretization")
res["dispersal_spec_check"] = "require_inputs_match_config passed (inside load_data)"
if "obs_holdout" in data:
    h = np.asarray(data["obs_holdout"])
    res["holdout_withheld"] = {"n": int(h.sum()), "share": float(h.mean())}
for strategy_name, strategy in (("median", init_to_median(num_samples=15)), ("sample", init_to_sample)):
    t0 = time.time()
    info = initialize_model(jax.random.PRNGKey(41), build_model_2d,
                            model_kwargs={"data": data, "prior_scale": 1.0},
                            init_strategy=strategy, validate_grad=False)
    params = info.param_info.z            # unconstrained
    pe = float(info.param_info.potential_energy)
    constrained = info.postprocess_fn(params)
    ld, trace = log_density(build_model_2d, (), {"data": data, "prior_scale": 1.0},
                            {k: v for k, v in constrained.items() if k in params})
    obs_sites = {k: float(np.sum(v["fn"].log_prob(v["value"]) * (1.0 if v.get("scale") is None else v["scale"])))
                 for k, v in trace.items() if v["type"] == "sample" and v.get("is_observed")}
    res[f"init_{strategy_name}"] = {
        "potential_energy": pe, "log_density": float(ld), "finite": bool(np.isfinite(pe) and np.isfinite(ld)),
        "n_latent_sites": len(params), "observed_site_logp": obs_sites, "seconds": round(time.time() - t0, 1)}
    print(strategy_name, json.dumps(res[f"init_{strategy_name}"], indent=1), flush=True)
with open(a.out, "w") as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps(res, indent=2, default=str))

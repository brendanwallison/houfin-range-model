"""Differentiate actual data likelihood at a stored fit; no optimization.

Separates data sensitivity from parameter-prior curvature, so a sampled but
inactive parameter cannot be mistaken for a fitted biological mechanism.
"""
import argparse
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density

from harness import Sandbox, ap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ["SBX_REFINED"])
    latents = {key: jnp.asarray(value) for key, value in sb.lat.items()}
    def score(params):
        lp, trace = log_density(ap.build_model_2d, (), {"data": sb.data, "prior_scale": 1.}, params)
        obs = trace["obs"]
        ll = jnp.sum(obs["fn"].log_prob(obs["value"]) * (obs.get("scale") if obs.get("scale") is not None else 1.))
        if "nest_loglik" in trace:
            nest = trace["nest_loglik"]
            ll = ll + jnp.sum(nest["fn"].log_prob(nest["value"]))
        return -ll, -lp
    started = time.time()
    print(json.dumps({"state": "differentiating", "n_sites": len(latents),
                      "n_parameters": sum(value.size for value in latents.values())}), flush=True)
    (nll, objective), grads = jax.jit(jax.value_and_grad(score, has_aux=True))(latents)
    jax.block_until_ready(grads)
    rows = []
    for key, value in grads.items():
        grad = np.asarray(value)
        rows.append({"site": key, "shape": list(grad.shape), "finite": bool(np.isfinite(grad).all()),
                     "l2": float(np.linalg.norm(grad)), "max_abs": float(np.max(np.abs(grad))) if grad.size else 0.})
    result = {"data_nll": float(nll), "objective": float(objective),
              "data_terms": ["obs", "nest_loglik"], "seconds": time.time() - started,
              "sites": rows, "devices": [str(device) for device in jax.devices()]}
    (args.out / "gradient_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not all(row["finite"] for row in rows):
        raise SystemExit("Nonfinite data-likelihood gradient")


if __name__ == "__main__":
    main()

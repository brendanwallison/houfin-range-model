"""Evaluate structural arms at explicitly mapped reference coefficients; no fit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density

from harness import Sandbox, ap
from src.config_utils import load_age_model_config
from local_model_trace_audit import trace_contributions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-reference-likelihood", action="store_true")
    parser.add_argument("--params", type=Path, help="Saved historical latent overrides; no optimizer is called")
    parser.add_argument("--expected-objective", type=float)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sb = Sandbox(os.environ["SBX_REFINED"])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    cfg = load_age_model_config()
    spec = cfg["population_model"]["manifold_prior"]
    if args.params:
        with np.load(args.params) as saved:
            lat.update({k: jnp.asarray(saved[k]) for k in saved.files})
    elif spec["form"] == "capacity_demography" and float(spec["fixed_capacity_correlation"]) < 1.:
        r = float(spec["fixed_capacity_correlation"])
        # Keep every effective coefficient exactly at the original fit.
        lat["capacity_factor"] = lat["manifold_factor"] * ((1. - r) / np.sqrt(1. - r*r))
    lp, tr = log_density(ap.build_model_2d, (), {"data": sb.data, "prior_scale": 1.}, lat)
    contributions = trace_contributions(tr)
    by_name = {row["name"]: row for row in contributions}
    data = {k: by_name[k]["log_density"] for k in ("obs", "nest_loglik")}
    reference_data = {"obs": -112057.684903781, "nest_loglik": -1965.800447046}
    expected_weights = np.asarray(lat["manifold_factor"]) * np.logaddexp(np.asarray(lat["w_scale_raw"]), 0.)
    w_error = float(np.max(np.abs(np.asarray(tr["w_env"]["value"]) - expected_weights)))
    active = [r["name"] for r in contributions if r["type"] == "sample" and not r["is_observed"]]
    fixed_onset = ap.DISEASE_ONSET_SPEC != {"form": "sigmoid"}
    checks = {"finite_objective": bool(np.isfinite(float(lp))),
              "onset_active_sites": ("disease_tau_raw" not in active) if fixed_onset else True}
    if not args.params:
        checks["coefficient_mapping"] = w_error < 1e-12
    if args.expected_objective is not None:
        checks["saved_objective"] = abs(-float(lp) - args.expected_objective) < 1e-5
    if args.expect_reference_likelihood:
        checks["reference_data_likelihood"] = all(abs(data[k] - reference_data[k]) < 1e-6 for k in data)
    result = {"state": "passed" if all(checks.values()) else "failed",
              "purpose": ("historical saved-point replay; no optimization" if args.params
                          else "mechanics validation at mapped reference; no optimization"),
              "objective": -float(lp), "data_log_density": data,
              "saved_params": ({"path": str(args.params), "sha256": hashlib.sha256(args.params.read_bytes()).hexdigest()}
                               if args.params else None),
              "effective_coefficients_max_error": w_error if not args.params else None,
              "active_sites": active, "onset_spec": ap.DISEASE_ONSET_SPEC,
              "manifold_spec": spec, "checks": checks,
              "source_hashes": {str(p.resolve().relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(ap.__file__), Path("src/model/age_fields.py"),
                                          Path("src/model/disease_onset.py"), Path("src/model/habitat_geometry.py")]},
              "seconds": time.time() - started, "devices": [str(d) for d in jax.devices()]}
    (args.out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

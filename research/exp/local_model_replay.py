"""Replay a known fit with its own imported model and record reproducibility anchors.

Run via invasion_sandbox/sbx_env.sh; this performs no fitting or interventions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import jax
import numpy as np
from numpyro.infer.util import log_density

from harness import Sandbox, ap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-objective", type=float, default=135590.8)
    parser.add_argument("--expected-bbs", type=float, default=110044.7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(json.dumps({"state": "loading", "devices": [str(device) for device in jax.devices()],
                      "model_source": ap.__file__}), flush=True)
    sb = Sandbox(os.environ["SBX_REFINED"])
    objective = -float(log_density(ap.build_model_2d, (), {"data": sb.data, "prior_scale": 1.}, sb.lat)[0])
    first = sb.run({"name": "baseline"}, keep_density=True)
    second = sb.run({"name": "canary"}, keep_density=True)
    first_density, second_density = first.pop("_density"), second.pop("_density")
    # The legacy harness reuses `h` for the held-out row mask after calculating
    # its hash. Compute from the actual density, never trust that field's type.
    hashes = [hashlib.sha256(np.ascontiguousarray(density[-1]).tobytes()).hexdigest()
              for density in (first_density, second_density)]
    bbs = -sum(item["loglik"] for item in first["abundance"].values())
    module_sources = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("src.") and filename and Path(filename).is_file():
            module_sources[name] = {"path": filename,
                                    "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest()}
    result = {"state": "passed", "objective": objective, "bbs_nll": bbs,
              "objective_error": objective - args.expected_objective,
              "bbs_error": bbs - args.expected_bbs,
              "final_state_max_nonfinite": first["nonfinite_state"],
              "exact_canary": hashes[0] == hashes[1],
              "final_land_density_sha256": hashes,
              "canary_max_abs_error": float(np.max(np.abs(first_density - second_density))),
              "numeric_canary": bool(np.allclose(first_density, second_density, rtol=1e-9, atol=1e-12)),
              "sources": module_sources, "devices": [str(device) for device in jax.devices()],
              "seconds": time.time() - started,
              "front": first["curves"]["rel_10pct"]["arrival_offset"],
              "heldout_nll": {key: -sum(value[field] for value in first["abundance"].values())
                              for key, field in (("blocks", "loglik_heldout_blocks"),
                                                 ("future", "loglik_heldout_2021"))}}
    checks = {"objective": abs(result["objective_error"]) < .2,
              "bbs": abs(result["bbs_error"]) < .2,
              "finite": first["nonfinite_state"] == 0,
              "canary": result["numeric_canary"]}
    result["checks"] = checks
    result["state"] = "passed" if all(checks.values()) else "failed"
    (args.out / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "sources"}), flush=True)
    if result["state"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

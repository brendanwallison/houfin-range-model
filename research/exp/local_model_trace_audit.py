"""Audit a real fit's full trace, penalties and exact field scale invariance.

No optimization and no new scientific intervention: alternative coordinates
preserve the fitted effective coefficients. Saves numeric contributions and
resolved configuration, rather than relying on model comments.
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
from src.config_utils import load_age_model_config


def trace_contributions(trace):
    rows = []
    for name, site in trace.items():
        value = site.get("value")
        if site["type"] not in ("sample", "deterministic"):
            continue
        row = {"name": name, "type": site["type"], "shape": list(np.shape(value)),
               "is_observed": bool(site.get("is_observed", False))}
        if site["type"] == "sample":
            lp = site["fn"].log_prob(value)
            if site.get("scale") is not None:
                lp = lp * site["scale"]
            row.update(distribution=type(site["fn"]).__name__, log_density=float(jnp.sum(lp)))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ["SBX_REFINED"])
    started = time.time()
    lp, trace = log_density(ap.build_model_2d, (), {"data": sb.data, "prior_scale": 1.}, sb.lat)
    sites = trace_contributions(trace)
    total = sum(row.get("log_density", 0.) for row in sites)
    if not np.isclose(total, float(lp), atol=1e-7, rtol=0.):
        raise ValueError(f"Trace accounting mismatch: {total} vs {lp}")
    base_w = np.asarray(trace["w_env"]["value"])
    raw_a = np.asarray(sb.lat["w_scale_raw"])
    a = np.logaddexp(raw_a, 0.)
    scales = []
    for c in (.5, 1., 2.):
        lat = dict(sb.lat)
        lat["manifold_factor"] = jnp.asarray(sb.lat["manifold_factor"]) * c
        adjusted = a / c
        lat["w_scale_raw"] = jnp.asarray(adjusted + np.log(-np.expm1(-adjusted)))
        lp_c, tr_c = log_density(ap.build_model_2d, (), {"data": sb.data, "prior_scale": 1.}, lat)
        contributions = trace_contributions(tr_c)
        by_name = {row["name"]: row for row in contributions}
        def part(name):
            return by_name.get(name, {}).get("log_density")
        scales.append({"factor_multiplier": c, "objective": -float(lp_c),
                       "effective_coefficient_max_error": float(np.max(np.abs(np.asarray(tr_c["w_env"]["value"]) - base_w))),
                       "obs_log_density": part("obs"),
                       "other_observed_or_factor_log_density": {
                           row["name"]: row["log_density"] for row in contributions
                           if row["is_observed"] and row["name"] != "obs"},
                       "factor_log_density": part("manifold_factor"),
                       "amplitude_log_density": part("w_scale_raw")})
    config = load_age_model_config()
    active_switches = {name: value for name, value in vars(ap).items()
                       if name.isupper() and isinstance(value, (bool, float, int, str, type(None)))}
    shapes = {key: list(np.shape(value)) for key, value in sb.data.items()
              if hasattr(value, "shape")}
    result = {"seconds": time.time() - started, "objective": -float(lp),
              "trace_accounting_error": total - float(lp), "sites": sites,
              "scale_invariance": scales, "active_switches": active_switches,
              "input_shapes": shapes, "resolved_config": config,
              "field_order": ["adult_survival", "reproduction", "capacity", "juvenile_survival"],
              "factor_norm": float(np.linalg.norm(np.asarray(sb.lat["manifold_factor"]))),
              "field_amplitudes": a.tolist()}
    (args.out / "trace_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("objective", "trace_accounting_error", "scale_invariance", "seconds")}), flush=True)


if __name__ == "__main__":
    main()

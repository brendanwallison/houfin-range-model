"""Explicit local disease-onset gates; geographic lag is handled by age_fields."""
import math

import jax.nn as jnn
import jax.numpy as jnp


def resolve_onset_spec(disease_prior):
    spec = dict(disease_prior.get("onset_gate", {}))
    form = spec.get("form", "sigmoid")
    if form not in {"sigmoid", "compact_start", "compact_midpoint"}:
        raise ValueError(f"unknown disease onset gate {form!r}")
    result = {"form": form}
    if form.startswith("compact"):
        duration = float(spec["duration_years"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("compact onset duration_years must be finite and positive")
        result["duration_years"] = duration
    elif spec.get("tau_years") is not None:
        tau = float(spec["tau_years"])
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("fixed sigmoid tau_years must be finite and positive")
        result["tau_years"] = tau
    return result


def onset_gate(age, tau, spec):
    """C2 finite-duration transition, or the historical unbounded sigmoid.

    compact_start treats reconstructed onset as the beginning of the gate;
    compact_midpoint keeps its historical sigmoid-midpoint interpretation.
    A saturated gate does not force constant mortality: recovery and current
    density continue to act separately in the forward model.
    """
    if spec["form"] == "sigmoid":
        return jnn.sigmoid(age / spec.get("tau_years", tau))
    x = age / spec["duration_years"]
    if spec["form"] == "compact_midpoint":
        x = x + .5
    x = jnp.clip(x, 0., 1.)
    return x ** 3 * (10. + x * (-15. + 6. * x))

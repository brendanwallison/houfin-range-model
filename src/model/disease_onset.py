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


def resolve_recovery_spec(disease_prior):
    """disease_prior.recovery_form: the shape of host recovery r(age) after local arrival.

    "exponential" (default; every existing config): r = rec * (1 - exp(-age / tau_rec)).
    "delayed": host ADAPTATION after a delay, r = rec * (S((age - d)/w) - S(-d/w)) / (1 - S(-d/w)), S the logistic:
      0 at arrival, near 0 for the first ~d - 2w years, saturating at rec. d is learned (disease_log_delay_rec,
      LogNormal(log delay_median_years, delay_log_scale)); w = width_years is fixed. Literature (RECOVERY-LIT,
      2026-11-05): eastern densities halved then stayed "stable or slowly declining" through the 2010s (Hochachka et al.
      2021 J Anim Ecol); tolerance evolved within ~20-25 yr (Henschen et al. 2023 PLoS Pathog); resistance within ~12 yr
      (Bonneaud et al. 2011 PNAS); virulence rose meanwhile (Hawley et al. 2013 PLoS Biol). An exponential from arrival
      cannot hold the plateau AND rebound ~20 yr later.
    """
    spec = dict(disease_prior.get("recovery", {}))
    form = str(spec.get("form", "exponential"))
    if form not in {"exponential", "delayed"}:
        raise ValueError(f"unknown disease recovery form {form!r}")
    out = {"form": form}
    if form == "delayed":
        out.update(delay_median_years=float(spec.get("delay_median_years", 22.0)),
                   delay_log_scale=float(spec.get("delay_log_scale", 0.2)),
                   width_years=float(spec.get("width_years", 3.0)))
        for k, v in out.items():
            if k != "form" and not (math.isfinite(v) and v > 0):
                raise ValueError(f"disease recovery {k} must be finite and positive: {v}")
    return out


def recovery_fraction(age, rec, tau_rec, spec, delay=None):
    """Host recovery r(age) in [0, rec], age >= 0 years since local onset (see resolve_recovery_spec)."""
    if spec["form"] == "exponential":
        return rec * (-jnp.expm1(-age / tau_rec))
    w = spec["width_years"]
    s0 = jnn.sigmoid(-delay / w)
    return rec * jnp.clip((jnn.sigmoid((age - delay) / w) - s0) / (1.0 - s0), 0.0, 1.0)


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

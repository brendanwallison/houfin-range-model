"""Diagnostic output constraints and exact likelihood-scale gauge removal."""
import jax.numpy as jnp
import numpy as np


def likelihood_gauge(latents, form):
    out = dict(latents)
    amplitudes = np.logaddexp(np.asarray(out["w_scale_raw"]), 0.)
    original = amplitudes.copy()
    out["manifold_factor"] = jnp.asarray(out["manifold_factor"]) * original[0]
    amplitudes /= original[0]
    fixed = [0]
    if form == "capacity_demography":
        out["capacity_factor"] = jnp.asarray(out["capacity_factor"]) * original[2]
        amplitudes[2] = 1.
        fixed.append(2)
    elif form != "exchangeable":
        raise ValueError("Profile gauge supports the registered F0/F2 constructions only")
    out["w_scale_raw"] = jnp.asarray(amplitudes + np.log(-np.expm1(-amplitudes)))
    return out, fixed


def late_growth_metric(sa, sj, fecundity, i0, i1, cells):
    """RMS per-cell change in log mate-unlimited potential lambda; no mean cancellation."""
    potential = sa + sj * fecundity
    delta = jnp.log(jnp.maximum(potential[i1,cells],1e-12)) - jnp.log(jnp.maximum(potential[i0,cells],1e-12))
    return jnp.sqrt(jnp.mean(delta*delta)+1e-16), delta


def constraint_cost(metric, target, rho, multiplier):
    if target is None:
        return jnp.asarray(0.)
    residual = metric-target
    return multiplier*residual + .5*rho*residual*residual

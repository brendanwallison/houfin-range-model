"""Support-respecting coordinates for a constrained-site MAP objective.

Transforms change optimization coordinates only. The caller's log_density
excludes their Jacobians; adding log_jacobian instead targets a different MAP
measure, such as the historical HMC-potential refinements.
"""
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from numpyro.distributions.transforms import AffineTransform, biject_to


class MapCoordinates:
    def __init__(self, trace, exclude=()):
        self.transforms, self.fixed, self.initial = {}, {}, {}
        for name, site in trace.items():
            if site["type"] != "sample" or site.get("is_observed", False):
                continue
            value = jnp.asarray(site["value"])
            if site["fn"].support.is_discrete:
                raise ValueError(f"Discrete site {name} needs a separate inference method")
            if name in exclude:
                self.fixed[name] = value
                continue
            fn = site["fn"]
            base_fn = fn
            while hasattr(base_fn, "base_dist"):
                base_fn = base_fn.base_dist
            transform = (AffineTransform(fn.mean, jnp.sqrt(fn.variance)) if type(base_fn).__name__ == "Normal"
                         else biject_to(fn.support))
            unconstrained = transform.inv(value)
            if not np.isfinite(np.asarray(unconstrained)).all():
                raise ValueError(f"Initial {name} lies on/outside its supported interior")
            self.transforms[name], self.initial[name] = transform, unconstrained
        unknown = set(exclude) - self.fixed.keys()
        if unknown:
            raise ValueError(f"Excluded sites were not free model sites: {sorted(unknown)}")
        self.names = tuple(sorted(self.initial))
        self.x0, self.unravel = ravel_pytree({k: self.initial[k] for k in self.names})

    def constrain(self, vector):
        values = self.unravel(vector)
        return {**self.fixed, **{k: self.transforms[k](values[k]) for k in self.names}}

    def log_jacobian(self, vector):
        values = self.unravel(vector)
        return sum(jnp.sum(self.transforms[k].log_abs_det_jacobian(
            values[k], self.transforms[k](values[k]))) for k in self.names)

    def describe(self):
        return [{"site": k, "shape": list(self.initial[k].shape),
                 "coordinates": type(self.transforms[k]).__name__} for k in self.names]

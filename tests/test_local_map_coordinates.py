"""Test actual objective measure and supports, not just transform round trips."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer.util import log_density
import pytest

from research.exp.local_map_coordinates import MapCoordinates


def test_gamma_mode_is_constrained_map_and_not_hmc_measure():
    def model():
        numpyro.sample("positive", dist.Gamma(2., 1.))
    tr = handlers.trace(handlers.seed(handlers.substitute(model, data={"positive": jnp.array(1.)}), 0)).get_trace()
    coords = MapCoordinates(tr)
    objective = lambda x: -log_density(model, (), {}, coords.constrain(x))[0]
    np.testing.assert_allclose(jax.grad(objective)(coords.x0), 0., atol=1e-7)
    hmc_objective = lambda x: objective(x) - coords.log_jacobian(x)
    np.testing.assert_allclose(jax.grad(hmc_objective)(coords.x0), -1., atol=1e-7)
    assert float(jax.hessian(objective)(coords.x0)[0, 0]) > 0.


def test_whitening_pins_bounded_and_positive_supports():
    def model():
        numpyro.sample("normal", dist.Normal(10., .02).expand([2]))
        numpyro.sample("bounded", dist.Beta(2., 3.))
        numpyro.sample("positive", dist.HalfNormal(2.))
        numpyro.sample("dead", dist.Normal(0., 1.))
        with handlers.condition(data={"pinned": jnp.array(4.)}):
            numpyro.sample("pinned", dist.Normal(0., 1.))
    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    coords = MapCoordinates(tr, exclude=["dead"])
    assert "pinned" not in coords.names and "pinned" not in coords.fixed
    assert coords.names == ("bounded", "normal", "positive")
    base = coords.constrain(coords.x0)
    for k in coords.names:
        np.testing.assert_allclose(base[k], tr[k]["value"], rtol=1e-5)
    shifted = coords.constrain(coords.x0 + .1)
    np.testing.assert_allclose(shifted["normal"] - base["normal"], .002, atol=1e-6)
    assert 0 < float(shifted["bounded"]) < 1 and float(shifted["positive"]) > 0
    np.testing.assert_array_equal(shifted["dead"], base["dead"])


def test_discrete_and_unknown_exclusions_are_errors():
    def model():
        numpyro.sample("state", dist.Bernoulli(.5))
    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    with pytest.raises(ValueError, match="Discrete"):
        MapCoordinates(tr)
    with pytest.raises(ValueError, match="Excluded"):
        MapCoordinates({}, exclude=["typo"])

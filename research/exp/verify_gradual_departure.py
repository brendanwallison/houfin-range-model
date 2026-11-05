"""Numerical checks of departure limits, age contrast and density derivatives."""
import json
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from gradual_departure import departure_probabilities

x = jnp.array([0., 1e-8, .1, .25, .5, 1., 2., 100.])
results = {}
for curve in ('veit', 'rational'):
    for rho in (.2, 1.):
        a, j = departure_probabilities(.6*x, .4*x, jnp.ones_like(x), curve, rho)
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(j))
        assert np.all(np.asarray(j) >= 0) and np.all(np.asarray(j) <= 1)
        np.testing.assert_allclose(a, rho*j)
        assert np.all(np.diff(np.asarray(j)) >= 0)
        # Splitting a stage pool between movers and stayers conserves that pool.
        np.testing.assert_allclose(.6*x*a+.6*x*(1-a), .6*x)
        np.testing.assert_allclose(.4*x*j+.4*x*(1-j), .4*x)
        for density in (.1, .5, 1., 2.):
            f = lambda z: departure_probabilities(z, 0., 1., curve, rho)[1]
            deriv = float(jax.grad(f)(density))
            fd = float((f(density+1e-5)-f(density-1e-5))/2e-5)
            np.testing.assert_allclose(deriv, fd, rtol=1e-6, atol=1e-8)
        results[f'{curve}_{rho}'] = np.asarray(j).tolist()
np.testing.assert_allclose(departure_probabilities(1e-8, 0., 1., 'rational', 1.)[1]/1e-8, 1., rtol=1e-7)
np.testing.assert_array_equal(departure_probabilities(x, 0., 1., 'veit', 1.)[1],
    jnp.clip(x-jax.nn.softplus(10*(x-1))/10, 1e-6, 1.))
print(json.dumps(dict(passed=True, density=x.tolist(), curves=results)))

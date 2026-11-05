import importlib.util
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from scipy.stats import nbinom

s=importlib.util.spec_from_file_location('hr',Path(__file__).with_name('historical_range.py'))
hr=importlib.util.module_from_spec(s);s.loader.exec_module(hr)

def test_nb_censoring_and_gradients():
    for k in [.1,.5,2.]:
        for mu in [0.,1e-10,.01,1.,100.]:
            np.testing.assert_allclose(hr.zero_loss(mu,k),-nbinom.logpmf(0,k,k/(k+mu)),atol=1e-12)
            expected=-np.log(-np.expm1(-float(hr.zero_loss(mu+1e-12,k))))
            np.testing.assert_allclose(hr.positive_loss(mu,k),expected)
            assert np.isfinite(jax.grad(hr.positive_loss)(jnp.array(mu),k))
            assert np.isfinite(jax.grad(jax.grad(hr.positive_loss))(jnp.array(mu),k))

def test_quadrature_weight_invariance_and_saturation():
    n=jnp.array([.01,.1,1.]);w=jnp.array([1.,2.,3.])
    first=jnp.sum(w*hr.zero_loss(n,.3))
    second=jnp.sum(jnp.repeat(w/2,2)*hr.zero_loss(jnp.repeat(n,2),.3))
    np.testing.assert_allclose(first,second)
    assert hr.positive_loss(100.,.3)<hr.positive_loss(1.,.3)<hr.positive_loss(.01,.3)

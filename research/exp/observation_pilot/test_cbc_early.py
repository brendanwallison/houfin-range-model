import importlib.util,sys
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
load('src.model.cbc_joint','cbc_joint.py')
early=load('cbc_early','cbc_early.py')

def test_detection_matches_poisson_probability_and_finite_derivatives():
    for present in [0.,1.]:
        for x in [-25.,-5.,0.,5.,24.]:
            got=float(early.detection_loglik(jnp.array(x),present))
            expected=np.log(-np.expm1(-np.exp(x))) if present else -np.exp(x)
            np.testing.assert_allclose(got,expected,rtol=1e-12)
            assert np.isfinite(jax.grad(early.detection_loglik)(jnp.array(x),present))
            assert np.isfinite(jax.grad(jax.grad(early.detection_loglik))(jnp.array(x),present))

def test_detection_not_count_magnitude():
    x=jnp.array([-3.,0.,2.]);counts=jnp.array([0.,1.,1000.])
    a=early.detection_loglik(x,counts>0)
    b=early.detection_loglik(x,jnp.minimum(counts,1))
    np.testing.assert_allclose(a,b)

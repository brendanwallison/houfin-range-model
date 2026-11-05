import importlib.util
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpyro
from numpyro import handlers
from scipy.stats import nbinom

spec=importlib.util.spec_from_file_location('cbc_joint',Path(__file__).with_name('cbc_joint.py'))
cbc=importlib.util.module_from_spec(spec);spec.loader.exec_module(cbc)

def test_effort_power_limit_and_gradient():
    x=jnp.array([-2.,0.,2.])
    np.testing.assert_allclose(cbc.boxcox_effort(x,0.),x)
    np.testing.assert_allclose(jax.jacfwd(lambda p:cbc.boxcox_effort(x,p))(0.),.5*np.asarray(x)**2)
    for p in [-.3,.2]:np.testing.assert_allclose(cbc.boxcox_effort(x,p),np.expm1(p*np.asarray(x))/p)

def toy():
    return {'land_rows':jnp.array([0,0]),'land_cols':jnp.array([0,1]),'cbc_circle':jnp.array([0,1,0]),
      'cbc_source_idx':jnp.array([[0,1],[0,1]]),'cbc_w_local':jnp.eye(2),
      'cbc_w_seasonal':jnp.array([[.5,0.],[.5,1.]]),'cbc_t':jnp.array([0,0,1]),
      'cbc_log_effort':jnp.zeros(3),'cbc_feeder':jnp.zeros(3),'cbc_feeder_missing':jnp.zeros(3),
      'cbc_count':jnp.array([4.,2.,100.]),'cbc_train':jnp.array([1.,1.,0.]),'cbc_n_circles':2}

def test_seasonal_readout_against_explicit_matrix():
    d=toy();a=jnp.array([[10.,20.],[30.,40.]]);j=a*.5
    for fraction in [0.,.3,1.]:
        got=cbc.winter_readout(a,j,d,fraction,.7)
        expected=[]
        W=(1-fraction)*np.eye(2)+fraction*np.array([[.5,0],[.5,1.]])
        for t,i in zip(d['cbc_t'],d['cbc_circle']):expected.append((W@(np.asarray(a[t])+.7*np.asarray(j[t])))[i])
        np.testing.assert_allclose(got,expected)
        assert np.isclose((W@np.asarray(a[0])).sum(),a[0].sum())
    grad=jax.grad(lambda q:cbc.winter_readout(a,j,d,q,.7).sum())(.3)
    fd=(cbc.winter_readout(a,j,d,.300001,.7).sum()-cbc.winter_readout(a,j,d,.299999,.7).sum())/2e-6
    np.testing.assert_allclose(grad,fd,rtol=1e-8)

def test_nb2_reference_and_no_holdout_leakage():
    d=toy();a=jnp.array([[[2.,3.]],[[4.,5.]]]);j=a*.4
    values={'cbc_migrant_fraction':.3,'cbc_young_availability':.7,'cbc_log_scale':.2,
      'cbc_circle_raw':jnp.array([.1,-.1]),'cbc_effort_slope':.6,'cbc_effort_power':.1,
      'cbc_feeder_effect':0.,'cbc_feeder_missing_effect':0.,'cbc_concentration':.8}
    def trace(data):return handlers.trace(handlers.substitute(cbc.cbc_observe,data=values)).get_trace(data,a,j,20.,{'circle_sd':1.})
    t=trace(d);mu=np.asarray(t['cbc_expected']['value']);ll=np.sum(nbinom.logpmf(np.asarray(d['cbc_count'])[:2],.8,.8/(.8+mu[:2])))
    np.testing.assert_allclose(t['cbc_loglik_train']['value'],ll,rtol=1e-12)
    e=dict(d);e['cbc_count']=d['cbc_count'].at[2].set(10000.)
    np.testing.assert_equal(trace(e)['cbc_loglik_train']['value'],t['cbc_loglik_train']['value'])
    assert trace(e)['cbc_loglik_test']['value']!=t['cbc_loglik_test']['value']

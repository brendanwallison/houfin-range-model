import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.exp.local_profile_metrics import likelihood_gauge, late_growth_metric, constraint_cost


@pytest.mark.parametrize("form",["exchangeable","capacity_demography"])
def test_gauge_preserves_every_effective_coefficient(form):
    random = np.random.default_rng(6)
    lat = {"manifold_factor":random.normal(size=(64,1)),"capacity_factor":random.normal(size=(64,1)),
           "w_scale_raw":np.array([2.,3.,4.,-2.])}
    def effective(p):
        amplitudes = np.logaddexp(np.asarray(p["w_scale_raw"]),0.)
        beta = np.asarray(p["manifold_factor"])*amplitudes
        if form == "capacity_demography":
            beta[:,2] = np.asarray(p["capacity_factor"])[:,0]*amplitudes[2]
        return beta
    changed, fixed = likelihood_gauge(lat,form)
    np.testing.assert_allclose(effective(changed),effective(lat),rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(np.logaddexp(np.asarray(changed["w_scale_raw"])[fixed],0.),1.,rtol=1e-6)


def test_rms_detects_equal_opposite_changes_that_escape_mean_constraint():
    delta = np.array([.2,-.2])
    sa = jnp.full((2,2),.4)
    sj = jnp.full((2,2),.2)
    fecundity = jnp.stack([jnp.full(2,3.),jnp.asarray((np.exp(delta)-.4)/.2)])
    rms, paired = late_growth_metric(sa,sj,fecundity,0,1,jnp.array([0,1]))
    np.testing.assert_allclose(rms,.2,atol=1e-6)
    assert abs(float(paired.mean())) < 1e-6


def test_default_constraint_is_exactly_zero_and_has_no_gradient():
    fn = lambda x: constraint_cost(x,None,10000.,3.)
    assert float(fn(.3)) == 0. and float(jax.grad(fn)(.3)) == 0.


def test_penalty_and_multiplier_have_expected_constraint_derivative():
    fn = lambda x: constraint_cost(x,.2,100.,-3.)
    np.testing.assert_allclose(jax.grad(fn)(.25),2.,atol=1e-6)

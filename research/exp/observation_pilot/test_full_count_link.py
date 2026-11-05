import unittest
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from scipy.stats import nbinom
from observation_pilot.full_count_link import observation_loglik


class FullCountTests(unittest.TestCase):
    def test_numeric_zeros_counts_and_x_match_independent_distribution(self):
        x=np.ones((4,1));a=np.array([.2,1.,3.,2.]);q=.7;k=1.3
        y=np.array([0.,2.,5.,0.]);unknown=np.array([False,False,False,True])
        theta=np.array([np.log(q),np.log(k)])
        actual=np.asarray(observation_loglik(jnp.asarray(theta),jnp.asarray(x),jnp.log(a),jnp.asarray(y),jnp.asarray(unknown)))
        p=k/(k+q*a);expected=nbinom.logpmf(y,k,p);expected[-1]=nbinom.logsf(0,k,p[-1])
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        def objective(t):return observation_loglik(t,jnp.asarray(x),jnp.log(a),jnp.asarray(y),jnp.asarray(unknown)).sum()
        gradient=np.asarray(jax.grad(objective)(jnp.asarray(theta)))
        h=1e-5;fd=np.array([(float(objective(theta+np.eye(2)[i]*h))-float(objective(theta-np.eye(2)[i]*h)))/(2*h) for i in range(2)])
        np.testing.assert_allclose(gradient,fd,rtol=1e-6,atol=1e-6)


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from scipy.special import expit,logit
from bbl_survival import negative_loglik,likelihood_counts,recovery_probabilities
from observation_pilot.bbl_block import adult_region_loglik


class AdultBlockTests(unittest.TestCase):
    def test_matches_existing_multinomial_with_censoring_and_gradients(self):
        rng=np.random.default_rng(41);era=np.repeat([0,1],6)
        p=recovery_probabilities(.55,np.array([.01,.02])[era],12)
        counts=np.array([rng.multinomial(5000,row) for row in p])
        row,col,recovered,never=likelihood_counts(counts)
        def objective(theta):
            return -adult_region_loglik(jax.nn.sigmoid(theta[0]),jax.nn.sigmoid(theta[1:]),
                jnp.asarray(era),jnp.asarray(row),jnp.asarray(col),jnp.asarray(recovered),jnp.asarray(never))
        theta=np.array([logit(.55),logit(.01),logit(.02)])
        self.assertAlmostEqual(float(objective(theta)),negative_loglik(theta,counts,era),places=8)
        gradient=np.asarray(jax.grad(objective)(jnp.asarray(theta)))
        epsilon=1e-5
        reference=np.array([(negative_loglik(theta+np.eye(3)[i]*epsilon,counts,era)-negative_loglik(theta-np.eye(3)[i]*epsilon,counts,era))/(2*epsilon) for i in range(3)])
        np.testing.assert_allclose(gradient,reference,rtol=1e-5,atol=1e-5)


if __name__=='__main__':unittest.main()

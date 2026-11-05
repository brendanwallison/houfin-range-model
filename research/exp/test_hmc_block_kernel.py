"""API/shape check for the actual mixed dense/diagonal NUTS configuration."""
import unittest
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC,NUTS


class BlockKernelTests(unittest.TestCase):
    def test_core_dense_and_zero_sum_observer_diagonal(self):
        def model():
            a=numpyro.sample('a',dist.Normal(0.,1.).expand([2]).to_event(1))
            b=numpyro.sample('b',dist.Normal(0.,1.))
            e=numpyro.sample('observer_route_eps',dist.ZeroSumNormal(1.,event_shape=(4,)))
            numpyro.sample('y',dist.Normal(a[0]+b+e,1.).to_event(1),obs=jnp.zeros(4))
        core=('a','b');obs=('observer_route_eps',)
        kernel=NUTS(model,dense_mass=[core],inverse_mass_matrix={core:jnp.eye(3),obs:jnp.ones(3)},max_tree_depth=2)
        mc=MCMC(kernel,num_warmup=8,num_samples=8,progress_bar=False)
        mc.run(jax.random.PRNGKey(13),init_params={'a':jnp.zeros(2),'b':jnp.array(0.),'observer_route_eps':jnp.zeros(3)},
            extra_fields=('num_steps','accept_prob','energy','potential_energy','diverging','adapt_state.step_size'))
        self.assertEqual(mc.get_samples()['observer_route_eps'].shape,(8,4))
        self.assertEqual(mc.get_extra_fields()['num_steps'].shape,(8,))


if __name__=='__main__':unittest.main()

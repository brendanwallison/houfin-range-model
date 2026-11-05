import tempfile,unittest
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import biject_to
from numpyro.infer import NUTS
from checkpointed_nuts import run_checkpointed


class BlockResumeTests(unittest.TestCase):
    def test_dense_diagonal_resume_with_dynamic_data_and_zero_sum_postprocess(self):
        def model(arrays):
            a=numpyro.sample('a',dist.Normal(0.,1.))
            e=numpyro.sample('observer_route_eps',dist.ZeroSumNormal(1.,event_shape=(4,)))
            numpyro.sample('y',dist.Normal(a+e,1.).to_event(1),obs=arrays['y'])
        def kernel():return NUTS(model,dense_mass=[('a',)],inverse_mass_matrix={('a',):jnp.eye(1),('observer_route_eps',):jnp.ones(3)},max_tree_depth=2)
        transform=biject_to(dist.ZeroSumNormal(1.,event_shape=(4,)).support)
        def post(z,kw):return {'a':z['a'],'observer_route_eps':transform(z['observer_route_eps'])}
        initial={'a':jnp.array(0.),'observer_route_eps':jnp.zeros(3)}
        kw={'arrays':{'y':jnp.arange(4.)/4}}
        with tempfile.TemporaryDirectory() as tmp:
            args=(initial,kw,jax.random.PRNGKey(42),8,8);p=Path(tmp)
            full=run_checkpointed(kernel(),*args,p/'full','block',post,chunk=2)
            run_checkpointed(kernel(),*args,p/'resume','block',post,chunk=2,session_steps=4)
            restored=run_checkpointed(kernel(),*args,p/'resume','block',post,chunk=2)
            for name in full[0]:np.testing.assert_array_equal(full[0][name],restored[0][name])
            self.assertEqual(restored[0]['observer_route_eps'].shape,(8,4))


if __name__=='__main__':unittest.main()

import tempfile,unittest
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS
from checkpointed_nuts import run_checkpointed


class CheckpointTests(unittest.TestCase):
    def test_resume_during_warmup_matches_uninterrupted_chain(self):
        def model():numpyro.sample('x',dist.Normal(0.,1.))
        def kernel():return NUTS(model,max_tree_depth=2)
        post=lambda z,kw:z
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);args=(dict(x=jnp.array(0.)),{},jax.random.PRNGKey(71),8,8)
            full=run_checkpointed(kernel(),*args,p/'full','toy',post,chunk=2)
            partial=run_checkpointed(kernel(),*args,p/'resumed','toy',post,chunk=2,session_steps=4)
            self.assertIsNone(partial)
            resumed=run_checkpointed(kernel(),*args,p/'resumed','toy',post,chunk=2)
            np.testing.assert_array_equal(full[0]['x'],resumed[0]['x'])
            for k in full[1]:np.testing.assert_array_equal(full[1][k],resumed[1][k])


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
import jax.numpy as jnp
from run_movement_followup import moment_origins


class MovementMomentTests(unittest.TestCase):
    def test_distance_transpose_counts_only_successful_offcell_moves(self):
        distance=np.array([[0.,125.],[125.,0.]])
        adult=np.array([[.1,.2],[.3,.4]]);juvenile=np.array([[.2,.1],[.4,.3]])
        stay=np.array([.3,.4]);sa=np.array([.5,.8]);dm=np.array([.7,.9])
        a=np.array([3.,7.]);j=np.array([5.,11.])
        def move(aa,jj):
            return jnp.array(stay)*aa+jnp.array(adult*distance)@aa,jnp.zeros_like(jj),jnp.array(juvenile*distance)@jj
        ma,mj=moment_origins(move,jnp.array(stay),jnp.array(sa),jnp.array(dm),jnp.array(a),jnp.array(j))
        np.testing.assert_allclose(ma,a*((sa[:,None]*adult*distance).sum(0)))
        np.testing.assert_allclose(mj,j*((dm[:,None]*juvenile*distance).sum(0)))


if __name__=='__main__':unittest.main()

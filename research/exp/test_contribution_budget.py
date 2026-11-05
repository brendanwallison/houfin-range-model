import unittest
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from contribution_budget import attributed_step


class ContributionTests(unittest.TestCase):
    def test_all_origin_flows_against_independent_matrix_enumeration(self):
        # Column = origin. Juvenile movers already include journey survival.
        A=np.array([[.6,.2],[.3,.7]])
        S=np.diag([.2,.4]);J=np.array([[.1,.2],[.3,.1]])
        sa=np.array([.5,.8]);sj=np.array([.3,.6]);a=np.array([7.,3.]);j=np.array([11.,5.]);f=np.array([2.,4.])
        def move(aa,jj):return jnp.asarray(A)@aa,jnp.asarray(S)@jj,jnp.asarray(J)@jj
        b=attributed_step(move,jnp.array(a),jnp.array(j),jnp.array(sa),jnp.array(sj),jnp.array(f),
                         tuple(jnp.array(np.diag(v)) for v in [A,S,J]))
        breeder=np.diag(sa)@A@np.diag(a)+(np.diag(sj)@S+J)@np.diag(j)
        mass=A@np.diag(a)+(S+J)@np.diag(j)
        np.testing.assert_allclose(b['breeder_contribution'],breeder.sum(0))
        np.testing.assert_allclose(b['next_breeders'],breeder.sum(1))
        np.testing.assert_allclose(b['retained_breeders'],np.diag(breeder))
        np.testing.assert_allclose(b['successful_breeder_export'],breeder.sum(0)-np.diag(breeder))
        np.testing.assert_allclose(b['immigrant_breeders'],breeder.sum(1)-np.diag(breeder))
        np.testing.assert_allclose(b['successful_emigration'],mass.sum(0)-np.diag(mass))
        np.testing.assert_allclose(b['immigration'],mass.sum(1)-np.diag(mass))
        np.testing.assert_allclose(b['total_descendants_origin'],((1+f)[:,None]*breeder).sum(0))
        self.assertAlmostEqual(float(b['breeder_contribution'].sum()),float(b['next_breeders'].sum()))

    def test_exporting_source_is_not_an_apparent_sink(self):
        # Origin 0 exports most of its successful contribution. Incoming animals
        # to origin 1 never get credited back to origin 1.
        A=jnp.array([[.1,0.],[.8,.1]]);J=jnp.array([[.05,0.],[.5,.1]])
        def move(a,j):return A@a,jnp.zeros_like(j),J@j
        b=attributed_step(move,jnp.array([1.,1.]),jnp.array([2.,0.]),jnp.ones(2),jnp.ones(2),jnp.zeros(2),
             (jnp.diag(A),jnp.zeros(2),jnp.diag(J)))
        self.assertGreater(float(b['breeder_contribution'][0]),1.)
        self.assertLess(float(b['retained_breeders'][0]),1.)
        self.assertLess(float(b['breeder_contribution'][1]),1.)
        self.assertGreater(float(b['next_breeders'][1]),1.)


if __name__=='__main__':unittest.main()

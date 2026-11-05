import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from priority_overflow import priority_overflow
from priority_overflow import departure_probabilities


def test_limits_conservation_and_adult_priority():
    for a,j,c in [(0.,0.,100.),(70.,50.,100.),(120.,50.,100.),(5.,4.,100.),(2.,4.,0.)]:
        ra,rj,ea,ej=map(float,priority_overflow(a,j,c,.1))
        assert min(ra,rj,ea,ej)>-1e-12
        assert ra<=a+1e-12 and rj<=j+1e-12 and ra+rj<=c+1e-12
        np.testing.assert_allclose([ra+ea,rj+ej],[a,j],atol=1e-12)
        np.testing.assert_allclose([ra,rj],[min(a,c),min(j,max(c-a,0.))],atol=.1)


def test_jacobian_finite_on_empty_and_capacity_boundaries():
    f=lambda x:jnp.stack(priority_overflow(x[0],x[1],x[2],5.))
    for v in [(0.,0.,100.),(100.,0.,100.),(100.,50.,100.),(70.,30.,100.),(0.,5.,0.)]:
        x=jnp.asarray(v);jac=np.asarray(jax.jacfwd(f)(x));hess=np.asarray(jax.jacfwd(jax.jacrev(f))(x))
        assert np.isfinite(jac).all() and np.isfinite(hess).all()
    x=jnp.array([70.,50.,100.]);v=jnp.array([.3,-.2,.1]);eps=1e-5
    np.testing.assert_allclose(jax.jacfwd(f)(x)@v,(f(x+eps*v)-f(x-eps*v))/(2*eps),rtol=1e-7,atol=1e-8)


def test_below_capacity_leakage_and_soft_boundary():
    ra,rj,ea,ej=priority_overflow(50.,0.,100.,5.)
    assert float(ea/50)<5e-6
    ra,rj,ea,ej=priority_overflow(100.,0.,100.,5.)
    np.testing.assert_allclose(float(ea/100),.05*np.log(2),rtol=1e-12)


def test_probabilities_match_pool_allocation_and_empty_derivatives():
    for a,j,c in [(0.,0.,100.),(70.,50.,100.),(120.,50.,100.),(1e-9,1e-8,100.)]:
        pa,pj=departure_probabilities(a,j,c,'adult_priority')
        ra,rj,ea,ej=priority_overflow(a,j,c,.05*c)
        np.testing.assert_allclose([a*pa,j*pj],[ea,ej],atol=1e-12)
        for mode in ['shared','adult_priority']:
            f=lambda v:jnp.stack(departure_probabilities(v[0],v[1],v[2],mode))
            x=jnp.array([a,j,c])
            assert np.isfinite(jax.jacfwd(f)(x)).all()
            assert np.isfinite(jax.jacfwd(jax.jacrev(f))(x)).all()
    pa,pj=departure_probabilities(70.,50.,100.,'adult_priority')
    assert float(pa)<float(pj)

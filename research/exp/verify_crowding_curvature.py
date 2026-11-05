"""Independent fecundity, derivative, equilibrium and viability checks."""
import json
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from src.model import age_forward as af,age_priors as ap
from src.model.density_regulation import CROWDING_POWER,crowding_argument
from src.vis.age_model_math import allee_viability

ratio=np.array([0.,1e-8,.1,.5,1.,2.,10.])
np.testing.assert_array_equal(crowding_argument(ratio,1),ratio)
np.testing.assert_array_equal(crowding_argument(ratio,2),ratio*ratio)
sa,sj,f,k=.55,.25,5.4,1.
c=float(ap.equilibrium_age_quantities(sa,sj,f,k,1.)[0])
n=jnp.asarray(ratio)
out=af.reproduction_age_structured(.6*n,.4*n,jnp.zeros_like(n),sa,sj,f,k,c,1.,
 allee_off=jnp.ones_like(n),yearling_breeding=True,return_F=True)
expected=f/(1+c*ratio**CROWDING_POWER)
np.testing.assert_allclose(out[-1],expected,rtol=1e-12,atol=1e-12)
assert np.isfinite(np.asarray(out[0])).all() and np.isfinite(np.asarray(out[1])).all()
def objective(p):
 a,j,ff,cap,density=p
 cc=ap.equilibrium_age_quantities(a,j,ff,cap,1.)[0]
 return af.reproduction_age_structured(.6*density,.4*density,0.,a,j,ff,cap,cc,1.,
   allee_off=1.,yearling_breeding=True,return_F=True)[-1]
p=jnp.array([sa,sj,f,k,.5]);grad=np.asarray(jax.grad(objective)(p));fd=[]
for i in range(5):
 d=np.eye(5)[i]*1e-5;fd.append(float((objective(p+d)-objective(p-d))/2e-5))
np.testing.assert_allclose(grad,fd,rtol=2e-5,atol=1e-8)
# Density feedback uses the same two-age census as the actual simulator.
f_at_K=f/(1+c);a0=1/(1+f_at_K);j0=1-a0
actual=af.reproduction_age_structured(a0,j0,0.,sa,sj,f,k,c,1.,allee_off=1.,yearling_breeding=True)
np.testing.assert_allclose(actual,[a0,j0],atol=2e-6)
def two_age(v):return jnp.stack(af.reproduction_age_structured(v[0],v[1],0.,sa,sj,f,k,c,1.,allee_off=1.,yearling_breeding=True))
feedback=np.linalg.eigvals(np.asarray(jax.jacfwd(two_age)(jnp.array([a0,j0]))))
assert np.max(np.abs(feedback))<1
# Compare diagnostic peak search directly with the forward reproductive function.
K=np.array([1.,.001,1.]);F=np.array([5.4,5.4,.5]);Sa=np.full(3,sa);Sj=np.full(3,sj)
v=allee_viability(Sa,Sj,F,K,10.,n_grid=512)
u=np.logspace(-5.,1.,512)[:,None];N=u*K
C=np.maximum(F*Sj/(1-Sa+1e-6)-1,0)
fwd=af.reproduction_age_structured(.6*N,.4*N,jnp.zeros_like(N),Sa,Sj,F,K,C,10.,yearling_breeding=True,return_F=True)[-1]
np.testing.assert_allclose(v['F_peak'],np.max(np.asarray(fwd),axis=0),rtol=1e-12,atol=1e-12)
assert not v['viable'][-1]
print(json.dumps(dict(passed=True,power=CROWDING_POWER,gradient=grad.tolist(),finite_difference=fd,
 feedback_eigenvalues=feedback.tolist(),viability=v['viable'].tolist())))

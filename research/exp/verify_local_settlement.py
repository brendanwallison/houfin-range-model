"""Compare settlement conservation, diagonals and endpoint moments to dense enumeration."""
import json
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from local_arrival_settlement import redistribute,composite_diagonal,offset_fractions,shift
ny,nx=4,5;n=ny*nx;hy,hx=2*ny,2*nx
capacity=np.linspace(.1,2.,n).reshape(ny,nx);capacity[1,2]=0.;capacity[3,0]=0.
q=np.linspace(.2,.8,n).reshape(ny,nx);alpha=.25
coords=np.stack(np.unravel_index(np.arange(n),(ny,nx)),axis=1)
rng=np.random.default_rng(43);kernel=rng.uniform(.1,1.,(hy,hx));kernel/=kernel.sum()
C=np.array([[kernel[(d[0]-o[0])%hy,(d[1]-o[1])%hx] for o in coords] for d in coords])
L=np.column_stack([np.asarray(redistribute(jnp.eye(n)[i].reshape(ny,nx),jnp.asarray(capacity),alpha)).ravel() for i in range(n)])
np.testing.assert_allclose(L.sum(0),(capacity.ravel()>0).astype(float),atol=1e-14)
assert L.min()>=0
M=L@(q.ravel()[:,None]*C)
diagonal=np.asarray(composite_diagonal(jnp.fft.fft2(jnp.asarray(kernel)),jnp.asarray(q),jnp.asarray(capacity),alpha)).ravel()
# Ocean origins have no biological tags; helper intentionally masks them.
land=capacity.ravel()>0
np.testing.assert_allclose(diagonal[land],np.diag(M)[land],rtol=1e-12,atol=1e-14)
parts=offset_fractions(jnp.asarray(capacity),alpha)
reconstructed=sum(np.asarray(shift(jnp.asarray(capacity)*part,dy,dx)) for dy,dx,part in parts)
np.testing.assert_allclose(reconstructed,redistribute(jnp.asarray(capacity),jnp.asarray(capacity),alpha),atol=1e-14)
sy=np.where(np.arange(hy)<=hy//2,np.arange(hy),np.arange(hy)-hy)
sx=np.where(np.arange(hx)<=hx//2,np.arange(hx),np.arange(hx)-hx)
distance=np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=-1)
moment_errors={}
for name,threshold in [('mean',None),('tail',1.5)]:
 expected=(M*(distance if threshold is None else distance>threshold)).sum(0)
 actual=np.zeros(n)
 for dy,dx,part in parts:
  dd=np.sqrt((sy[:,None]+dy)**2+(sx[None,:]+dx)**2)
  filt=dd if threshold is None else dd>threshold
  weighted=kernel*filt
  CK=np.array([[weighted[(d[0]-o[0])%hy,(d[1]-o[1])%hx] for o in coords] for d in coords])
  for origin in range(n):
   field=q.reshape(-1)*CK[:,origin]
   actual[origin]+=np.asarray(shift(jnp.asarray(field.reshape(ny,nx))*part,dy,dx)).sum()
 np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)
 moment_errors[name]=float(np.max(np.abs(actual-expected)))
arr=jnp.asarray(capacity*.3)
np.testing.assert_array_equal(redistribute(arr,jnp.asarray(capacity),0.),arr)
value,grad=jax.value_and_grad(lambda a:jnp.sum(redistribute(a,jnp.asarray(capacity),alpha)**2))(arr)
assert np.isfinite(value) and np.isfinite(grad).all()
print(json.dumps(dict(passed=True,mass_error=float(np.max(np.abs(L.sum(0)-land))),diagonal_error=float(np.max(np.abs(diagonal[land]-np.diag(M)[land]))),moment_errors=moment_errors)))

"""Illustrative density-shape calculation; no House Finch fit or parameter estimate."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('/mnt/c/Dev/houfin-range-model/research/experiments/lambda_history_review_20261102')
sa,sj,xmatch,r=.55,.25,.5,1.5
replacement=(1-sa)/sj
for Rcheck in (1.2,3.,12.):
 for density in (.1,.5,.9,1.,1.2):
  fcheck=replacement*Rcheck
  def response(fmax):return fmax/(1+(fmax/replacement-1)*density)
  h=1e-5*fcheck
  measured=(np.log(response(fcheck+h))-np.log(response(fcheck-h)))/(np.log(fcheck+h)-np.log(fcheck-h))
  expected=(1-density)/(1+(Rcheck-1)*density)
  np.testing.assert_allclose(measured,expected,rtol=1e-6,atol=1e-9)
x=np.linspace(0,1.3,500);records=[]
fig,ax=plt.subplots(figsize=(7.2,4.3))
for theta,color in [(1,'#b24b32'),(2,'#216b9a')]:
 z=xmatch**theta
 R=r*(1-z)/(1-r*z)
 fmax=replacement*R;c=R-1
 f=lambda n:fmax/(1+c*n**theta)
 g=sa+sj*f(x)
 Aeq=1/(1+replacement);Jeq=1-Aeq
 def step(v):
  a,j=v;adult=sa*a+sj*j
  return np.array([adult,adult*f(a+j)])
 eps=1e-6;v=np.array([Aeq,Jeq])
 jac=np.column_stack([(step(v+eps*np.eye(2)[i])-step(v-eps*np.eye(2)[i]))/(2*eps) for i in range(2)])
 eig=np.linalg.eigvals(jac)
 np.testing.assert_allclose(step(v),v,atol=1e-12)
 np.testing.assert_allclose(f(xmatch),replacement*r,atol=1e-12)
 assert np.max(np.abs(eig))<1
 record=dict(theta=theta,Sa=sa,Sj=sj,replacement_output=replacement,R=R,Fmax=fmax,
   potential_growth=sa+sj*fmax,matched_density=xmatch,matched_output=f(xmatch),
   fixed_density_projection_at_match=sa+sj*f(xmatch),equilibrium_feedback_eigenvalues=eig.tolist())
 records.append(record)
 ax.plot(x,g,color=color,lw=2.5,label=f'Crowding power {theta}; maximum growth {sa+sj*fmax:.2f}')
ax.axhline(1,color='.45',ls='--',lw=1,label='Replacement')
ax.scatter([xmatch],[sa+sj*replacement*r],s=60,color='black',zorder=4)
ax.annotate('Same reproduction at half of K',xy=(xmatch,sa+sj*replacement*r),xytext=(.55,1.65),
 arrowprops=dict(arrowstyle='->',color='.3'),fontsize=10)
ax.set(xlabel='Population as a fraction of K',ylabel='Fixed-density annual growth multiplier',
 title='Matching reproduction can require much less maximum growth')
ax.set_ylim(.8,2.02);ax.legend(loc='upper right',fontsize=9,frameon=False)
ax.spines[['right','top']].set_visible(False)
fig.text(.01,.015,'Illustrative algebra, not fitted values. Yearling breeding; no movement, mate limitation or disease.',fontsize=8)
fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'DENSITY_SHAPE_DERIVATION.png',dpi=170)
(out/'DENSITY_SHAPE_DERIVATION.json').write_text(json.dumps(dict(scope='Illustrative, not fitted',records=records),indent=2)+'\n')
print(json.dumps(records))

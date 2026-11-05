"""Registered full-count NB2 control, with exact presence-only censoring.

Uses NumPyro's existing count distribution and SciPy's standard optimizer.
Demographic points remain fixed. Zero/positive counts share one source scale
and concentration; X contributes probability of a positive count.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from numpyro.distributions import NegativeBinomialLogits
from scipy.optimize import minimize
try:
    from .population_link import REFERENCES,align,paired_blocks
    from .calibrate import subsets,PACKAGES
    from .census import breeding_grid
    from .prepare import write_json
except ImportError:
    from population_link import REFERENCES,align,paired_blocks
    from calibrate import subsets,PACKAGES
    from census import breeding_grid
    from prepare import write_json


def observation_loglik(theta,design,offset,count,x_only):
    log_k=theta[-1]
    eta=design@theta[:-1]+offset
    distribution=NegativeBinomialLogits(total_count=jnp.exp(log_k),logits=eta-log_k)
    numeric=distribution.log_prob(count)
    log_p0=distribution.log_prob(jnp.zeros_like(count))
    return jnp.where(x_only,jnp.log(-jnp.expm1(log_p0)),numeric)


@jax.jit
def objective(theta,design,offset,count,x_only):
    return -jnp.mean(observation_loglik(theta,design,offset,count,x_only))


value_gradient=jax.jit(jax.value_and_grad(objective))


def design_matrix(frame,train,geography):
    import patsy
    terms='log_duration + I(log_duration**2) + log_distance + n_observers + hour_sin + hour_cos'
    if frame.loc[train,'protocol'].nunique()==2:terms+=" + C(protocol, levels=['P21','P22'])"
    if geography:terms+=' + x + y + I(x*x) + I(y*y) + x:y + t'
    raw=patsy.dmatrix(terms,frame,return_type='dataframe')
    center=raw.loc[train].mean();scale=raw.loc[train].std(ddof=0)
    keep=(scale>1e-10)|(raw.columns=='Intercept')
    raw=raw.loc[:,keep];center=center[keep];scale=scale[keep]
    center.loc['Intercept']=0.;scale.loc['Intercept']=1.
    return ((raw-center)/scale).to_numpy(),{'columns':list(raw.columns),'center':center.to_dict(),'scale':scale.to_dict()}


def fit(frame,offset,geo=False):
    split=subsets(frame);train=np.asarray(split['train']);x,contract=design_matrix(frame,train,geo)
    unknown=frame['count'].isna().to_numpy()
    if np.any(unknown&((frame.detected.to_numpy()!=1)|(frame.x_only.to_numpy()!=1))):raise ValueError('unclassified unknown count')
    y=frame['count'].fillna(0).to_numpy(float)
    args=tuple(jnp.asarray(a) for a in (x[train],offset[train],y[train],unknown[train]))
    theta=np.zeros(x.shape[1]+1)
    numeric=train&~unknown
    theta[0]=np.log(max(float(y[numeric].mean()),1e-8)/float(np.exp(offset[numeric]).mean()))
    def vg(t):
        v,g=value_gradient(jnp.asarray(t),*args)
        return float(v),np.asarray(g,dtype=float)
    result=minimize(vg,theta,jac=True,method='L-BFGS-B',bounds=[(None,None)]*(len(theta)-1)+[(-20.,20.)],
                    options={'maxiter':1000,'ftol':1e-12,'gtol':1e-7,'maxls':40})
    gradient=float(np.max(np.abs(result.jac)));boundary=bool(abs(result.x[-1])>=19.999)
    accepted=bool(result.success and np.isfinite(result.fun) and gradient<1e-3 and not boundary)
    record={'numerically_valid':accepted,'optimizer_success':bool(result.success),'message':str(result.message),
      'mean_nll':float(result.fun),'max_abs_mean_gradient':gradient,'log_concentration_boundary':boundary,
      'concentration':float(np.exp(result.x[-1])),'parameters':result.x.tolist(),'design':contract,'scores':{}}
    scores={}
    if not accepted:return record,scores
    for label,mask in split.items():
        mask=np.asarray(mask)
        if not mask.any():continue
        xx,oo,yy,uu=(jnp.asarray(a) for a in (x[mask],offset[mask],y[mask],unknown[mask]))
        ll=np.asarray(observation_loglik(jnp.asarray(result.x),xx,oo,yy,uu))
        if not np.isfinite(ll).all():raise ValueError('nonfinite predictive score')
        eta=x[mask]@result.x[:-1]+offset[mask];k=np.exp(result.x[-1]);mu=np.exp(eta)
        p=-np.expm1(-k*np.logaddexp(0,eta-np.log(k)))
        record['scores'][label]={'n':int(mask.sum()),'mean_log_score':float(ll.mean()),
          'observed_detection_rate':float(frame.loc[mask,'detected'].mean()),'predicted_detection_rate':float(p.mean()),
          'numeric_count_mean_observed':float(y[mask&~unknown].mean()),
          'numeric_count_mean_predicted':float(mu[~unknown[mask]].mean()),
          'encounter_brier':float(np.mean((frame.loc[mask,'detected'].to_numpy()-p)**2))}
        scores[label]=ll
    return record,scores


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--cohort',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,PACKAGES);jax.config.update('jax_enable_x64',True)
    if a.out.exists():raise FileExistsError('existing full-count comparison retained')
    frame=pd.read_parquet(a.cohort);split=subsets(frame)
    states={}
    for name in REFERENCES:
        with np.load(a.root/'census'/name/'census.npz') as z:states[name]={k:z[k] for k in z}
    cells,years,keep=align(frame,states[REFERENCES[0]])
    if not keep.all():raise ValueError('cohort/state alignment mismatch')
    configurations=[('effort_only',None,None,False),('geographic',None,None,True)]
    for name in REFERENCES:
        for label,phi in [('no_young',0.),('bbs_weight',float(states[name]['fledgling_weight'])),('all_young',1.)]:
            configurations.append((name+'__'+label,name,phi,False))
    result={'status':'fixed-point full-count comparison','scope':'No demographic fitting; zero/positive counts and X censoring share NB2 parameters',
      'cohort':str(a.cohort),'n':len(frame),'registration':'POPULATION_EBIRD_LINK_PILOT_20261101.md','arms':{}}
    a.out.mkdir(parents=True);row_scores={}
    for label,name,phi,geo in configurations:
        offset=np.zeros(len(frame))
        if name:
            s=states[name];grid=breeding_grid(s['total'],s['young'],phi,float(s['pop_scalar']),initial=s['initial'])
            offset=np.log(grid[years,cells])
        r,scores=fit(frame,offset,geo)
        r['reference']=name;r['young_weight']=phi;result['arms'][label]=r;row_scores[label]=scores
        print(label,'valid',r['numerically_valid'],'k',r['concentration'],'gradient',r['max_abs_mean_gradient'],flush=True)
        write_json(a.out/'results.json',result)
    for label,scores in row_scores.items():
        result['arms'][label]['paired_vs_geographic']={}
        for section in ['spatial_test','temporal_test','both_test']:
            if section in scores and section in row_scores['geographic']:
                result['arms'][label]['paired_vs_geographic'][section]=paired_blocks(
                    scores[section]-row_scores['geographic'][section],frame.loc[split[section],'block'].to_numpy())
    write_json(a.out/'results.json',result)
    lines=['# Full-count population-linked eBird comparison','',
      f"Same June-only Canadian cohort: {len(frame):,} checklists. Fixed demographic points; source observation parameters fitted on training data only.",'',
      '| Arm | Numeric checks | Concentration | Spatial gain | Temporal gain | Temporal observed / predicted detection |','|---|---|---:|---:|---:|---|']
    for name,r in result['arms'].items():
        t=r['scores'].get('temporal_test',{});paired=r.get('paired_vs_geographic',{})
        def value(k):return 'not reported' if k not in paired else f"{paired[k]['mean']:+.4f}"
        rate='not reported' if not t else f"{t['observed_detection_rate']:.4f} / {t['predicted_detection_rate']:.4f}"
        lines.append(f"| {name} | {r['numerically_valid']} | {r['concentration']:.4g} | {value('spatial_test')} | {value('temporal_test')} | {rate} |")
    lines+=['','Paired gains use the same NB2/censoring model, cohort and held-out rows. Bootstrap intervals and block counts are in JSON. This is not a joint population fit or parameter-identification result. No observation spatial field or year trend is allowed in population-linked nuisance parameters.']
    (a.out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

"""Registered fixed-population versus geographic eBird observation comparison.

No demographic fitting. Use the same data/holdouts across every arm, with
explicit young-visibility scenarios and distinct source observation parameters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
try:
    from .calibrate import spatial,thinning,subsets,describe_fit,PACKAGES
    from .census import breeding_grid
    from .prepare import write_json
except ImportError:
    from calibrate import spatial,thinning,subsets,describe_fit,PACKAGES
    from census import breeding_grid
    from prepare import write_json

REFERENCES=['step33_nestcbc_s41','step33_nestcbc_abs_s41','step34_abs_natdep_s41']


def encounter_intercept(offset,rate):
    if not 0<rate<1:raise ValueError('both encounter outcomes are required')
    # Marginal-rate initialization only. This does not alter the GLM likelihood,
    # add a prior, or use any held-out outcome.
    return brentq(lambda b:float(np.mean(-np.expm1(-np.exp(np.clip(b+offset,-700,700)))))-rate,-100,100)


def align(frame,state):
    index=pd.MultiIndex.from_arrays([state['land_rows'],state['land_cols']])
    cells=index.get_indexer(pd.MultiIndex.from_arrays([frame.row,frame.col]))
    years=pd.Index(state['years']).get_indexer(frame.year)
    return cells,years,(cells>=0)&(years>=0)


def paired_blocks(delta,blocks,seed=41):
    groups=pd.DataFrame({'delta':delta,'block':blocks}).groupby('block').delta.agg(['sum','size'])
    if len(groups)<2:return {'mean':float(np.mean(delta)),'blocks':len(groups),'interval90':None}
    rng=np.random.default_rng(seed);draw=rng.integers(0,len(groups),(2000,len(groups)))
    samples=groups['sum'].to_numpy()[draw].sum(1)/groups['size'].to_numpy()[draw].sum(1)
    return {'mean':float(np.mean(delta)),'blocks':len(groups),'interval90':np.quantile(samples,[.05,.95]).tolist(),
            'scope':'Exploratory spatial-block bootstrap conditional on fitted models; not demographic parameter uncertainty'}


def fit_arm(frame,offset,geography=False):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP
    split=subsets(frame);train=frame[split['train']];train_offset=offset[np.asarray(split['train'])]
    effort='log_duration + I(log_duration**2) + log_distance + n_observers + hour_sin + hour_cos'
    if train.protocol.nunique()==2:effort+=" + C(protocol, levels=['P21','P22'])"
    formula='detected ~ '+effort
    if geography:formula+=' + x + y + I(x*x) + I(y*y) + x:y + t'
    positive=(train['count']>0)&train['count'].notna()
    count_formula=formula.replace('detected ~','count ~')
    with warnings.catch_warnings(record=True) as seen:
        encounter_model=smf.glm(formula,train,family=sm.families.Binomial(sm.families.links.CLogLog()),offset=train_offset)
        start=np.zeros(encounter_model.exog.shape[1]);start[0]=encounter_intercept(train_offset,float(train.detected.mean()))
        er=encounter_model.fit(start_params=start,maxiter=300)
        count_model=TruncatedLFNegativeBinomialP.from_formula(count_formula,train[positive],
            offset=train_offset[np.asarray(positive)],truncation=0,p=2)
        count_start=np.zeros(count_model.exog.shape[1]+1)
        # At alpha=1, a zero-truncated NB2 has mean 1+mu. Match its
        # training mean with an intercept before optimizing all nuisance terms.
        q=max((float(train.loc[positive,'count'].mean())-1)/float(np.exp(train_offset[np.asarray(positive)]).mean()),1e-8)
        count_start[0]=np.log(q);count_start[-1]=1.
        count=count_model.fit(start_params=count_start,method='bfgs',maxiter=400,disp=False)
    result={'encounter':describe_fit(er),'positive_count':describe_fit(count),'formula':formula,
            'warnings':list({str(w.message) for w in seen}),'scores':{},
            'initialization':{'encounter_intercept':float(start[0]),'positive_count_intercept':float(count_start[0]),'alpha':1.}}
    if not result['encounter']['converged'] or not result['positive_count']['converged']:
        result['status']='numerical nonconvergence; no predictive comparison reported'
        return result,{},{}
    row_scores={};predictions={}
    for label,mask in split.items():
        f=frame[mask];o=offset[np.asarray(mask)]
        if not len(f):continue
        prob=np.asarray(er.predict(f,offset=o));prob=np.clip(prob,1e-12,1-1e-12)
        scores=stats.bernoulli.logpmf(f.detected.to_numpy(),prob)
        numeric=(f['count']>0)&f['count'].notna()
        if numeric.any():
            model=TruncatedLFNegativeBinomialP.from_formula(count_formula,f[numeric],offset=o[np.asarray(numeric)],truncation=0,p=2)
            if model.exog_names!=count.model.exog_names:raise ValueError('validation design mismatch')
            scores[np.flatnonzero(numeric)]+=model.loglikeobs(np.asarray(count.params))
        if not np.isfinite(scores).all():raise ValueError('nonfinite held-out likelihood')
        row_scores[label]=scores;predictions[label]=prob
        result['scores'][label]={'n':len(f),'mean_hurdle_log_score':float(scores.mean()),
          'observed_detection_rate':float(f.detected.mean()),'predicted_detection_rate':float(prob.mean()),
          'brier':float(np.mean((f.detected.to_numpy()-prob)**2)),
          'positive_numeric_records':int(numeric.sum()),'x_only_records':int(f.x_only.sum())}
    return result,row_scores,predictions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--country',default='CA',choices=['CA','US']);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,PACKAGES)
    if a.out.exists():raise FileExistsError('existing comparison retained')
    directory=a.root/f'ebird_{a.country}'
    if not (directory/'READY.json').exists():raise ValueError('full focal extraction is not complete')
    states={};receipts={}
    for step in REFERENCES:
        folder=a.root/'census'/step
        receipt=json.load(open(folder/'READY.json'))
        if receipt['bbs_before_observer_parity_max_abs']>1e-10:raise ValueError('census parity not accepted')
        with np.load(folder/'census.npz') as z:states[step]={k:z[k] for k in z}
        receipts[step]=receipt
    reference=states[REFERENCES[0]]
    for s in states.values():
        for k in ['land_rows','land_cols','years']:
            if not np.array_equal(reference[k],s[k]):raise ValueError('population export grid/time mismatch')
    frame=pd.read_parquet(directory/'breeding.parquet')
    frame=frame[pd.to_datetime(frame.date).dt.month==6].copy()
    n_june=len(frame);frame=spatial(frame)
    _,_,keep=align(frame,reference)
    excluded=int((~keep).sum());excluded_positive=int(frame.loc[~keep,'detected'].sum())
    frame=thinning(frame[keep],'event_key').reset_index(drop=True)
    cells,years,keep=align(frame,reference)
    if not keep.all():raise AssertionError('unexpected cohort change')
    frame['log_duration']=np.log(frame.duration_min);frame['log_distance']=np.log1p(frame.distance_km)
    frame['hour_sin']=np.sin(frame.hour*2*np.pi/24);frame['hour_cos']=np.cos(frame.hour*2*np.pi/24)
    split=subsets(frame)
    result={'country':a.country,'status':'fixed-point observation comparison','registration':'POPULATION_EBIRD_LINK_PILOT_20261101.md',
      'n_june_before_grid_filter':n_june,'n_excluded_off_model_land_or_year':excluded,
      'positive_events_excluded_off_model_land_or_year':excluded_positive,'n_after_outcome_independent_thinning':len(frame),
      'splits':{k:int(v.sum()) for k,v in split.items()},'references':receipts,'arms':{},
      'cohort_sha256':hashlib.sha256('\n'.join(frame.event_key).encode()).hexdigest(),
      'limits':['Source nuisance fits only; no demographic parameters updated.',
        'References use BBS/NestWatch training and previously examined exploratory holdouts.',
        'No eBird observer parameter is shared with BBS.',
        'Source-relative scale is not absolute detection or abundance.']}
    a.out.mkdir(parents=True)
    frame.to_parquet(a.out/'cohort.parquet',index=False)
    configs=[('effort_only',None,None,False),('geographic',None,None,True)]
    for step in REFERENCES:
        s=states[step]
        for label,phi in [('no_young',0.),('bbs_weight',float(s['fledgling_weight'])),('all_young',1.)]:
            configs.append((step+'__'+label,step,phi,False))
    row_scores={}
    for name,step,phi,geo in configs:
        offset=np.zeros(len(frame))
        if step:
            s=states[step]
            grid=breeding_grid(s['total'],s['young'],phi,float(s['pop_scalar']),initial=s['initial'])
            abundance=grid[years,cells]
            if np.any(~np.isfinite(abundance)) or np.any(abundance<=0):raise ValueError('invalid linked readout')
            offset=np.log(abundance)
        try:
            record,scores,predictions=fit_arm(frame,offset,geo)
            record['reference']=step;record['young_weight']=phi
            if step:record['positive_records_at_readout_floor']=int(((abundance<=1e-6)&(frame.detected==1)).sum())
            result['arms'][name]=record;row_scores[name]=scores
            # Retain paired row evidence without writing observer identifiers again.
            np.savez_compressed(a.out/(name+'.scores.npz'),**{k:v for k,v in scores.items()})
        except Exception as exc:
            result['arms'][name]={'error':str(exc),'reference':step,'young_weight':phi}
        write_json(a.out/'results.json',result)
        print(name,'finished',result['arms'][name].get('error',''),flush=True)
    baseline=row_scores.get('geographic',{})
    for name,scores in row_scores.items():
        result['arms'][name]['paired_vs_geographic']={}
        for label in ['spatial_test','temporal_test','both_test']:
            if label in baseline and label in scores:
                result['arms'][name]['paired_vs_geographic'][label]=paired_blocks(scores[label]-baseline[label],frame.loc[split[label],'block'].to_numpy())
    write_json(a.out/'results.json',result)
    lines=['# Population-linked breeding eBird pilot','',
      f"Country: {a.country}; June-only; {len(frame):,} retained checklists. Population points are fixed; this is not a demographic refit.",'',
      f"Excluded off model land/year: {excluded:,} events, including {excluded_positive:,} detections. Census source receipts, sample hashes, convergence/warnings and all split scores are in results.json.",'',
      '| Arm | Encounter / count converged | Spatial score gain | Temporal score gain | Temporal observed / predicted detection |',
      '|---|---|---:|---:|---|']
    for name,r in result['arms'].items():
        if 'error' in r:lines.append(f"| {name} | ERROR: {r['error']} | | | |");continue
        paired=r.get('paired_vs_geographic',{});t=r['scores'].get('temporal_test',{})
        def text(key):
            v=paired.get(key,{})
            return 'missing' if not v else f"{v['mean']:+.4f}"
        rate='not reported' if not t else f"{t['observed_detection_rate']:.4f} / {t['predicted_detection_rate']:.4f}"
        lines.append(f"| {name} | {r['encounter']['converged']} / {r['positive_count']['converged']} | {text('spatial_test')} | {text('temporal_test')} | {rate} |")
    lines+=['','Gains are paired mean hurdle log-score differences against the same-row geographic model. Block-bootstrap intervals and block counts are in JSON; they are exploratory, conditional on fitted points, and not demographic uncertainty. Convergence and predictive gain alone do not establish a valid census interpretation.']
    (a.out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

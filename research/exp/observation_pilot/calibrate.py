"""Standard source-observation pilots using statsmodels, not a demographic refit.

Spatial split is the project's pinned 216-km BBS holdout. Temporal test is
2021-25. All selection/thinning is independent of House Finch outcomes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
try:
    from .prepare import write_json
except ImportError:
    from prepare import write_json

ROOT=Path('/work/07980/bwa386/ls6/houfin/processed/grid54')
PACKAGES='/mnt/c/Dev/houfin-range-model/.research_runtime/observation_packages'


def spatial(frame):
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(ROOT/'ref_grid_54km.tif') as ref:
        x,y=transform('EPSG:4326',ref.crs,frame.lon.to_numpy(),frame.lat.to_numpy())
        row,col=rasterio.transform.rowcol(ref.transform,x,y)
        frame=frame.assign(row=np.asarray(row),col=np.asarray(col))
        frame=frame[(frame.row>=0)&(frame.row<ref.height)&(frame.col>=0)&(frame.col<ref.width)].copy()
    held={tuple(x) for x in json.load(open(ROOT/'holdout_blocks_216km_s64.json'))['blocks']}
    frame['block']=(frame.row//4).astype(str)+'_'+(frame.col//4).astype(str)
    frame['spatial_test']=[(r//4,c//4) in held for r,c in zip(frame.row,frame.col)]
    frame['temporal_test']=frame.year>=2021
    frame['x']=(frame.lon+95)/20;frame['y']=(frame.lat-38)/10;frame['t']=(frame.year-2015)/5
    return frame


def thinning(frame,key,max_rows=200000):
    # One event per cell/year/week. Hash ties do not depend on detections/counts.
    frame=frame.copy()
    frame['tie']=pd.util.hash_pandas_object(frame[key].astype(str),index=False).to_numpy()
    dates=pd.to_datetime(frame['date'])
    frame['week']=dates.dt.isocalendar().week.astype(int)
    frame=frame.sort_values('tie').drop_duplicates(['row','col','year','week'])
    return frame.head(max_rows).copy()


def subsets(frame):
    return {'train':~frame.spatial_test & ~frame.temporal_test,
            'spatial_test':frame.spatial_test & ~frame.temporal_test,
            'temporal_test':~frame.spatial_test & frame.temporal_test,
            'both_test':frame.spatial_test & frame.temporal_test}


def describe_fit(fit):
    params=fit.params
    return {'converged':bool(getattr(fit,'mle_retvals',{}).get('converged',getattr(fit,'converged',False))),
            'params':dict(zip(fit.model.exog_names,np.asarray(params).tolist())),
            'covariance_available':bool(getattr(fit,'normalized_cov_params',None) is not None)}


def positive_logscore(fit,formula,frame):
    # Use the public formula constructor for new observations. The internal
    # formula metadata differs across statsmodels versions (including 0.15).
    from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP
    model=TruncatedLFNegativeBinomialP.from_formula(formula,frame,truncation=0,p=2)
    if model.exog_names!=fit.model.exog_names:
        raise ValueError('validation design columns differ from training')
    return model.loglikeobs(np.asarray(fit.params))


def cbc(root):
    sys.path.insert(0,PACKAGES)
    import statsmodels
    import statsmodels.formula.api as smf
    frame=pd.read_parquet(root/'cbc/circle_years.parquet').rename(columns={'latitude':'lat','longitude':'lon','year_end':'year'})
    frame=frame[(frame.year>=2010)&(frame.year<=2025)&frame.party_hours.notna()&frame['count'].notna()&~frame.unknown_count.astype(bool)&frame.date_year_mismatch.eq(0)]
    frame=spatial(frame)
    frame['log_effort']=np.log(frame.party_hours)
    frame['log_feeder']=np.log1p(frame.feeder_hrs.fillna(0));frame['feeder_missing']=frame.feeder_hrs.isna().astype(int)
    split=subsets(frame);train=frame[split['train']]
    ecology='x + y + I(x*x) + I(y*y) + x:y + t'
    formulas={'linear_effort':'count ~ '+ecology+' + log_effort + log_feeder + feeder_missing',
              'curved_effort':'count ~ '+ecology+' + log_effort + I(log_effort**2) + log_feeder + feeder_missing'}
    result={'source':'CBC','scope':'Winter relative-count observation calibration; no seasonal demographic inference',
            'statsmodels_version':statsmodels.__version__,'n':len(frame),'splits':{k:int(v.sum()) for k,v in split.items()},'models':{}}
    for name,formula in formulas.items():
        # Constant missing indicators are omitted rather than fitted as zero columns.
        if train.feeder_missing.nunique()==1:formula=formula.replace(' + feeder_missing','')
        with warnings.catch_warnings(record=True) as seen:
            fit=smf.negativebinomial(formula,train).fit(method='lbfgs',maxiter=300,disp=False)
        record=describe_fit(fit);record['formula']=formula;record['warnings']=list({str(w.message) for w in seen})
        alpha=float(fit.params['alpha']);record['scores']={}
        if alpha<=0 or not np.isfinite(alpha):raise ValueError('invalid NB2 dispersion')
        for label,mask in split.items():
            f=frame[mask]
            if len(f)==0:continue
            mu=np.asarray(fit.predict(f));y=f['count'].to_numpy();n=1/alpha
            record['scores'][label]={'n':len(f),'mean_log_score':float(stats.nbinom.logpmf(y,n,n/(n+mu)).mean()),
              'log1p_rmse':float(np.sqrt(np.mean((np.log1p(y)-np.log1p(mu))**2))),
              'mean_observed':float(y.mean()),'mean_predicted':float(mu.mean())}
        effort=np.geomspace(train.party_hours.quantile(.01),train.party_hours.quantile(.99),30)
        standard=train.iloc[[0]*len(effort)].copy();standard['log_effort']=np.log(effort)
        record['effort_curve']={'hours':effort.tolist(),'mean_count':np.asarray(fit.predict(standard)).tolist()}
        result['models'][name]=record
    # Conditional circle-panel contrast: separates stable site heterogeneity from
    # effort without pretending to predict spatially withheld, unseen circles.
    panel=frame[((frame.lon<-110)&(frame.lat<49))|((frame.lon>-83)&(frame.lat<48))]
    available=panel[panel.year<=2020].groupby('abbrev').year.nunique()
    circles=available[available>=8].index
    circles=sorted(circles,key=lambda x:hashlib.sha256(x.encode()).hexdigest())[:80]
    panel=panel[panel.abbrev.isin(circles)]
    ptrain=panel[panel.year<=2020];ptest=panel[panel.year>=2021]
    with warnings.catch_warnings(record=True) as seen:
        pf=smf.negativebinomial('count ~ C(abbrev) + t + log_effort + I(log_effort**2) + log_feeder',ptrain).fit(method='lbfgs',maxiter=400,disp=False)
    record=describe_fit(pf);record['circles']=len(circles);record['warnings']=list({str(w.message) for w in seen})
    record['scope']='Within-circle temporal prediction only; circles selected on geography and survey availability, not focal outcomes.'
    record['scores']={}
    alpha=float(pf.params['alpha'])
    for label,f in [('train',ptrain),('temporal_test',ptest)]:
        if not len(f):continue
        mu=np.asarray(pf.predict(f));n=1/alpha
        record['scores'][label]={'n':len(f),'mean_log_score':float(stats.nbinom.logpmf(f['count'],n,n/(n+mu)).mean()),
          'mean_observed':float(f['count'].mean()),'mean_predicted':float(mu.mean())}
    result['models']['circle_panel']=record
    old=root/'cbc/calibration.json'
    if old.exists() and not (root/'cbc/calibration_initial.json').exists():old.rename(root/'cbc/calibration_initial.json')
    write_json(root/'cbc/calibration.json',result)
    print('CBC CALIBRATION',result['splits'],{k:v['converged'] for k,v in result['models'].items()},flush=True)


def ebird(root,country):
    sys.path.insert(0,PACKAGES)
    import statsmodels
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP
    directory=root/f'ebird_{country}'
    results={'source':'eBird','country':country,'statsmodels_version':statsmodels.__version__,
             'scope':'Effort-adjusted encounter/conditional-count pilot, not absolute abundance or a demographic refit','seasons':{}}
    for s in ['breeding','winter']:
        path=directory/f'{s}.parquet'
        if not path.exists():continue
        frame=pd.read_parquet(path);frame=thinning(spatial(frame),'event_key')
        frame['log_duration']=np.log(frame.duration_min)
        frame['log_distance']=np.log1p(frame.distance_km)
        frame['hour_sin']=np.sin(frame.hour*2*np.pi/24);frame['hour_cos']=np.cos(frame.hour*2*np.pi/24)
        split=subsets(frame);train=frame[split['train']]
        if len(train)<500 or train.detected.nunique()<2:
            results['seasons'][s]={'status':'insufficient calibration sample','n':len(frame)};continue
        ecology='x + y + I(x*x) + I(y*y) + x:y + t'
        effort='log_duration + I(log_duration**2) + log_distance + n_observers + hour_sin + hour_cos'
        # Protocol may have only one level; a fixed effect is estimable only if both occur.
        if train.protocol.nunique()==2:effort+=" + C(protocol, levels=['P21','P22'])"
        formula='detected ~ '+ecology+' + '+effort
        er=smf.glm(formula,train,family=sm.families.Binomial()).fit(maxiter=150)
        positive=train[(train['count']>0)&train['count'].notna()]
        count_formula=formula.replace('detected ~','count ~')
        count=TruncatedLFNegativeBinomialP.from_formula(count_formula,positive,truncation=0,p=2).fit(method='bfgs',maxiter=200,disp=False)
        result={'n':len(frame),'positive_numeric_train':len(positive),'splits':{k:int(v.sum()) for k,v in split.items()},
                'encounter':describe_fit(er),'conditional_count':describe_fit(count),'formula':formula,'scores':{}}
        for label,mask in split.items():
            f=frame[mask]
            if not len(f):continue
            prob=np.clip(np.asarray(er.predict(f)),1e-9,1-1e-9)
            detect=f.detected.to_numpy()
            log_score=stats.bernoulli.logpmf(detect,prob)
            numeric=(f['count']>0)&f['count'].notna()
            if numeric.any():
                log_score[np.flatnonzero(numeric)]+=positive_logscore(count,count_formula,f[numeric])
            result['scores'][label]={'n':len(f),'mean_hurdle_log_score':float(np.mean(log_score)),
              'encounter_brier':float(np.mean((detect-prob)**2)),
              'observed_detection_rate':float(detect.mean()),'predicted_detection_rate':float(prob.mean()),
              'x_only_records':int(f.x_only.sum())}
        results['seasons'][s]=result
    write_json(directory/'calibration.json',results)
    print(country,'CALIBRATION DONE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',choices=['cbc','CA','US','watch']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    if a.source=='cbc':cbc(a.root)
    elif a.source in ('CA','US'):ebird(a.root,a.source)
    else:
        # One bounded attempt per ready source, with durable receipts and no silent retries.
        import fcntl
        a.root.mkdir(parents=True,exist_ok=True)
        with (a.root/'calibration_watch.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            (a.root/'calibration_watch.pid').write_text(str(__import__('os').getpid()))
            while not (a.root/'STOP_CALIBRATION').exists():
                all_done=True
                for label,folder in [('cbc','cbc'),('CA','ebird_CA'),('US','ebird_US')]:
                    target=a.root/folder
                    if (target/'CALIBRATION_RECEIPT.json').exists():continue
                    all_done=False
                    if not (target/'READY.json').exists():continue
                    start=time.time()
                    with (target/'calibration.log').open('w') as stream:
                        run=subprocess.run(['timeout','--kill-after=30','1800',sys.executable,__file__,label,'--root',str(a.root)],stdout=stream,stderr=subprocess.STDOUT)
                    write_json(target/'CALIBRATION_RECEIPT.json',{'exit_code':run.returncode,'elapsed_seconds':time.time()-start,'started':start})
                write_json(a.root/'calibration_watch.json',{'epoch':time.time(),'all_sources_attempted':all_done})
                if all_done:break
                time.sleep(30)


if __name__=='__main__':main()

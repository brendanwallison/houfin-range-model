"""Compare emergent process maps, separately from component parameter stability.

These are robustness comparisons of fitted processes, not a posterior proof of
identifiability. Fixed-kernel/shared-field assumptions can enforce agreement.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def ratio(n,d):return np.divide(n,d,out=np.full_like(n,np.nan,dtype=float),where=d>1e-20)
def weighted_corr(x,y,w):
    if w.sum()==0:return None
    w=w/w.sum();xx=x-np.dot(w,x);yy=y-np.dot(w,y);den=np.sqrt(np.dot(w,xx*xx)*np.dot(w,yy*yy))
    return float(np.dot(w,xx*yy)/den) if den>0 else None


def compare(x,y,weights,threshold=False):
    keep=np.isfinite(x)&np.isfinite(y)&(weights>0)
    w=weights[keep];a,b=x[keep],y[keep]
    if not len(a):return {'coverage':0.}
    out={'coverage':float(w.sum()/weights.sum()),'cells':int(keep.sum()),
         'weighted_pearson':weighted_corr(a,b,w),
         'spearman':float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None,
         'mean_abs_change':float(np.dot(w,np.abs(a-b))/w.sum()),
         'rms_change':float(np.sqrt(np.dot(w,(a-b)**2)/w.sum())),
         'q50_abs_change':float(np.median(np.abs(a-b)))}
    if threshold:
        out['categories']={}
        for eps in [0.,.01,.05]:
            ca=np.where(a<1-eps,-1,np.where(a>1+eps,1,0));cb=np.where(b<1-eps,-1,np.where(b>1+eps,1,0))
            union=(ca==-1)|(cb==-1);both=(ca==-1)&(cb==-1)
            out['categories'][str(eps)]={'agreement':float(w[ca==cb].sum()/w.sum()),
                'source_sink_reversal_cells':int(np.count_nonzero(ca*cb==-1)),
                'source_sink_reversal':float(w[ca*cb==-1].sum()/w.sum()),
                'sink_jaccard':float(w[both].sum()/w[union].sum()) if union.any() else None}
    return out


def load_maps(root,step):
    with np.load(root/step/'annual_budget.npz') as z:
        years=z['years'];rows=z['land_rows'];cols=z['land_cols'];out={}
        for era,lo,hi in [('bbs_early',1966,1985),('modern',2010,2025)]:
            tt=(years>=lo)&(years<=hi)
            a=z['input_adult'][tt];j=z['input_juvenile'][tt];adult=a.sum(0);state=(a+j).sum(0)
            out[era]={'adult_exposure':adult,'state_exposure':state,
              'potential_lambda':ratio((z['potential_lambda'][tt]*a).sum(0),adult),
              'C_breeding':ratio(z['breeder_contribution'][tt].sum(0),adult),
              'C_state':ratio(z['total_descendants_origin'][tt].sum(0),state),
              'local_breeder_replacement':ratio((np.nan_to_num(z['local_breeder_replacement'][tt])*a).sum(0),adult),
              'local_total_replacement':ratio((np.nan_to_num(z['local_total_replacement'][tt])*(a+j)).sum(0),state),
              'retention':ratio(z['retained_breeders'][tt].sum(0),adult),
              'export':ratio(z['successful_breeder_export'][tt].sum(0),adult),
              'net_cell_flux':ratio((z['immigration'][tt]-z['successful_emigration'][tt]).sum(0),state),
              'gross_successful_emigration':ratio(z['successful_emigration'][tt].sum(0),state),
              'mean_density':(z['next_adult'][tt]+z['next_juvenile'][tt]).mean(0)}
        trajectories=z['next_adult']+z['next_juvenile']
    return out,years,rows,cols,trajectories


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();repo=a.repo
    root=Path('/mnt/e/Datasets/houfin/contribution_20261101_v3');data=json.load(open(repo/'research/experiments/FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json'))
    inventory={r['step']:r for r in data['runs']}
    pairs=[]
    def add(x,y,label):
        if (root/x/'READY.json').exists() and (root/y/'READY.json').exists():pairs.append((x,y,label))
    for prefix in ['ctl','nat']:
        base=f'step35_est1951_{prefix}_mid_s41'
        for bound in ['lo','hi']:add(base,f'step35_est1951_{prefix}_{bound}_s41','initial abundance')
        add(base,f'step35_est1951_{prefix}_mid_s42','later-state seed')
        add(base,f'step36_est1951_nonest_{prefix}_s41','later-state NestWatch')
    add('step33_nestcbc_abs_s41','step34_abs_s42','original absolute seed')
    add('step34_abs_natdep_s41','step34_abs_natdep_s42','original native seed')
    add('step33_nestcbc_abs_s41','step35_est1951_ctl_mid_s41','initialization formulation')
    add('step34_abs_natdep_s41','step35_est1951_nat_mid_s41','initialization formulation')
    add('step33_nestcbc_abs_s41','step34_abs_natdep_s41','native departure')
    add('step35_est1951_ctl_mid_s41','step35_est1951_nat_mid_s41','later-state native departure')
    add('step34_abs_natdep_s41','step34_abs_v2_natdep_s41','DESK basis')
    add('step33_nestcbc_abs_s41','step36_nonest_ctl_s41','NestWatch')
    add('step34_abs_natdep_s41','step36_nonest_nat_s41','NestWatch')
    add('step33_nestcbc_s41','step33_nestcbc_abs_s41','Allee law')
    cache={}
    for s in sorted({s for x,y,_ in pairs for s in [x,y]}):cache[s]=load_maps(root,s)
    ref=cache['step33_nestcbc_abs_s41'];lr,lc=ref[2],ref[3];index={int(r)*10000+int(c):i for i,(r,c) in enumerate(zip(lr,lc))}
    import pickle,rasterio
    from rasterio.warp import transform
    source=inventory['step33_nestcbc_abs_s41'];cfg=source['config'];metadata=pickle.load(open(Path(cfg['input_dir'])/'metadata.pkl','rb'))
    oy=np.asarray(metadata['years'])[np.asarray(metadata['obs_time_indices'],int)]
    obsr,obsc=np.asarray(metadata['obs_rows']),np.asarray(metadata['obs_cols']);positive=np.asarray(metadata['observed_results'])>0
    oi=np.array([index.get(int(r)*10000+int(c),-1) for r,c in zip(obsr,obsc)])
    with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
        xx,yy=rasterio.transform.xy(src.transform,lr,lc);lon,lat=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
    regions={'all':np.ones(len(lr),bool),'west':lon<-100,'plains':(lon>=-100)&(lon<-95),'east':(lon>=-80)&(lat>=25)&(lat<49)}
    obsweights={}
    for era,lo,hi in [('bbs_early',1966,1985),('modern',2010,2025)]:
        use=(oy>=lo)&(oy<=hi)&positive&(oi>=0);obsweights[era]=np.bincount(oi[use],minlength=len(lr)).astype(float)
    records=[]
    for x,y,label in pairs:
        left,right=cache[x],cache[y]
        if not(np.array_equal(left[1],right[1]) and np.array_equal(left[2],right[2]) and np.array_equal(left[3],right[3])):raise ValueError('grid/time mismatch')
        row={'left':x,'right':y,'contrast':label,'eras':{}}
        for era in ['bbs_early','modern']:
            aa,bb=left[0][era],right[0][era];row['eras'][era]={}
            for reg,mask in regions.items():
                row['eras'][era][reg]={}
                for support,w in [('land',mask.astype(float)),('observed_positive',mask*obsweights[era]),
                    ('pooled_breeders',mask*(aa['adult_exposure']+bb['adult_exposure'])/2)]:
                    row['eras'][era][reg][support]={}
                    for metric in aa:
                        if metric.endswith('exposure'):continue
                        row['eras'][era][reg][support][metric]=compare(aa[metric],bb[metric],w,
                            metric in ['C_breeding','C_state','potential_lambda','local_breeder_replacement','local_total_replacement'])
        records.append(row)
    out=repo/'research/experiments/FUNCTIONAL_STABILITY_20261101.json';out.write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    lines=['# Functional process stability across fitted experiments','',
      'Compares realized source/sink structure, local replacement and movement directly. These are cross-fit robustness measurements, not proof of statistical identifiability or posterior uncertainty. Component non-identifiability does not imply process non-identifiability. Agreement imposed by fixed kernels/shared fields must also be distinguished from information supplied by data.','',
      'All comparisons use common grid/time support. The primary table weights cells by the same actual positive BBS record counts, independently of either fitted population. Equal-land and symmetric pooled-breeder weights, four regions and early/modern eras are retained in JSON. Coverage measures common finite support; absent support is not classified as agreement. Replacement/contribution classes use sink <.95, source >1.05, and a neutral band; .01 thresholds are also saved. This avoids treating tiny fluctuations around replacement as large structural changes.','',
      '| Contrast | Left to right | C map correlation | Mean absolute C change | C class agreement | Source/sink reversal | Local replacement correlation | Net flux correlation | Mean absolute export change |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        q=r['eras']['modern']['all']['observed_positive'];c=q['C_breeding'];g=q['local_total_replacement'];net=q['net_cell_flux'];ex=q['export']
        fmt=lambda v:'unavailable' if v is None else f'{v:.3f}'
        cats=c.get('categories',{}).get('0.05',{})
        lines.append(f"| {r['contrast']} | {r['left']} → {r['right']} | {fmt(c.get('weighted_pearson'))} | {fmt(c.get('mean_abs_change'))} | {fmt(cats.get('agreement'))} | {fmt(cats.get('source_sink_reversal'))} | {fmt(g.get('weighted_pearson'))} | {fmt(net.get('weighted_pearson'))} | {fmt(ex.get('mean_abs_change'))} |")
    lines+=['','Local total replacement is the actual-state no-movement next-year/previous-year postbirth population ratio; C breeding follows breeder-origin survivors/recruits across all destinations. They use different censuses. Net cell flux is gross immigration minus successful emigration per starting individual; it includes exchanges within regions. Gross export above is successful breeder export per source breeder.','',
      'The fitted-state comparison does not establish that growth responses agree away from fitted densities, that complete origin/destination transport matrices agree, or that stochastic establishment/forecasts are identified. Those require common-state perturbations, prediction tests and posterior/simulation-based evaluation.']
    lines+=['','## Interpretation','',
      'Founder abundance is not making every realized process fluctuate equally. Across the 280/560/1120 scenarios, modern C map correlations against the middle scenario are .997-.999 for the control and .977-.994 with native departure. Mean absolute C changes are .006-.009 and .010-.021, respectively. There are no direct source-to-sink reversals on modern positive-record support at the .05 threshold; disagreements are with the neutral band. The control retains 94.9-95.5% class agreement, native departure 89.2-94.0%. These are substantially more stable than the native fits\' age-specific survival allocation (adult .389-.445, juvenile/adult .801-.962).','',
      'Changing the movement mechanism is a different contrast. Adding native departure in the later-state middle scenario changes the western C map appreciably: modern positive-record-weighted correlation .769, mean absolute change .121, class agreement 54.6%, and direct source/sink reversal 10.7%. National values dilute this regional change. Early western reversal is 14.0%. Thus stable regional average C does not establish stable spatial contribution structure.','',
      'Equal-land results also preserve the initial-abundance C-map similarity (.993-.999 correlations). However, net-flux correlation for native low versus middle abundance falls to .794 on equal land, versus .973 on positive-record support. Stability is weaker away from observed habitat and cannot be generalized to every movement quantity.','',
      'Independent historical/abundance and demographic observations remain valuable to estimate initial conditions and separate survival/reproduction. The present comparisons do not justify saying that the entire experiment is unconstrained, or that all realized growth and movement outputs are unidentified. These scenarios fix abundance rather than estimating a freely floating parameter; they are a sensitivity bracket, not a profile likelihood.','',
      '## Regional movement-mechanism comparison','',
      'Later-state middle abundance, control versus native departure; common positive-record weights. Fractions below are record-weighted shares, not independent-cell percentages.','',
      '| Era | Region | C correlation | Mean absolute C change | Class agreement | Direct source/sink reversal | Net flux correlation |',
      '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        if r['contrast']!='later-state native departure':continue
        for era in ['bbs_early','modern']:
            for reg in regions:
                q=r['eras'][era][reg]['observed_positive'];c=q['C_breeding'];cats=c['categories']['0.05']
                lines.append(f"| {era} | {reg} | {fmt(c['weighted_pearson'])} | {fmt(c['mean_abs_change'])} | {fmt(cats['agreement'])} | {fmt(cats['source_sink_reversal'])} | {fmt(q['net_cell_flux']['weighted_pearson'])} |")
    lines+=['','## Contribution versus potential lambda stability','',
      'Potential lambda here is Sa + Sj * Fmax before Allee/density suppression, density-dependent disease and dispersal losses. In this yearling-breeding life cycle it is a potential local growth diagnostic, not actual growth at zero density under absolute Allee. Annual lambda values are weighted by starting breeders, matching the within-cell temporal exposure weighting of breeding contribution. The same positive-record cell weights are used for both. Actual-state local total replacement is included separately; it is a one-step population ratio, not an eigenvalue.','',
      '| Contrast | Right fit (relative to left in main table) | C correlation | Potential lambda correlation | C mean absolute change | Lambda mean absolute change | C=1 crossing cells | Lambda=1 crossing cells | Local replacement correlation |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        q=r['eras']['modern']['all']['observed_positive'];c=q['C_breeding'];l=q['potential_lambda']
        lines.append(f"| {r['contrast']} | {r['right']} | {fmt(c['weighted_pearson'])} | {fmt(l['weighted_pearson'])} | {fmt(c['mean_abs_change'])} | {fmt(l['mean_abs_change'])} | {c['categories']['0.0']['source_sink_reversal_cells']} | {l['categories']['0.0']['source_sink_reversal_cells']} | {fmt(q['local_total_replacement']['weighted_pearson'])} |")
    (repo/'research/experiments/FUNCTIONAL_STABILITY_20261101.md').write_text('\n'.join(lines)+'\n')
    print('FUNCTIONAL COMPARISONS',len(records))


if __name__=='__main__':main()

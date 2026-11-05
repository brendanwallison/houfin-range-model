"""Compare vital-rate and aggregate process stability across supported model versions."""
import json,pickle
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform
from functional_stability_audit import compare,ratio

repo=Path('/mnt/c/Dev/houfin-range-model');exp=repo/'research/experiments'
root=Path('/mnt/e/Datasets/houfin/contribution_20261101_v3')
inv={r['step']:r for r in json.loads((exp/'FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json').read_text())['runs']}
steps=sorted(p.name for p in root.iterdir() if (p/'READY.json').exists() and p.name in inv and
 inv[p.name]['config']['population_model'].get('reproduction',{}).get('yearling_breeding') is True)
ref='step33_nestcbc_abs_s41'
eras={'1951-1965':(1951,1965),'1966-1985':(1966,1985),'2010-2025':(2010,2025)}
cache={};grids=None
for step in steps:
    with np.load(root/step/'annual_budget.npz') as z:
        years=z['years'];rr,cc=z['land_rows'],z['land_cols']
        if grids is None:grids=(years.copy(),rr.copy(),cc.copy())
        elif not all(np.array_equal(x,y) for x,y in zip(grids,(years,rr,cc))):raise ValueError('grid/time mismatch')
        a=z['input_adult'];n=a+z['input_juvenile']
        fields={k:z[k] for k in ['potential_lambda','local_adult_survival','local_juvenile_stayer_survival',
            'realized_offspring_per_breeder','breeder_contribution','local_total_replacement','immigration','successful_emigration']}
        fields['full_process_lambda']=z['potential_lambda_with_process'] if 'potential_lambda_with_process' in z else fields['potential_lambda']
        fields['biological_C']=z['biological_breeder_contribution'] if 'biological_breeder_contribution' in z else fields['breeder_contribution']
    cache[step]={}
    for era,(lo,hi) in eras.items():
        tt=(years>=lo)&(years<=hi);aa=a[tt].sum(0);nn=n[tt].sum(0)
        cache[step][era]={k:fields[k][tt].mean(0) for k in ['potential_lambda','local_adult_survival',
            'local_juvenile_stayer_survival','realized_offspring_per_breeder','full_process_lambda']}
        cache[step][era].update(C=ratio(fields['breeder_contribution'][tt].sum(0),aa),
            biological_C=ratio(fields['biological_C'][tt].sum(0),aa),
            local_replacement=ratio((np.nan_to_num(fields['local_total_replacement'][tt])*n[tt]).sum(0),nn),
            net_flux=ratio((fields['immigration'][tt]-fields['successful_emigration'][tt]).sum(0),nn))
with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
    xx,yy=rasterio.transform.xy(src.transform,rr,cc)
    lon,lat=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
regions={'all':np.ones(len(rr),bool),'west':lon<-100,'plains':(lon>=-100)&(lon<-95),
    'east':(lon>=-80)&(lat>=25)&(lat<49)}
belt=(lat>=25)&(lat<49)
regions.update(east_belt=(lon>=-95)&belt,plains_belt=(lon>=-100)&(lon<-95)&belt,west_belt=(lon<-100)&belt)
m=pickle.load(open(Path(inv[ref]['config']['input_dir'])/'metadata.pkl','rb'))
oy=np.asarray(m['years'])[np.asarray(m['obs_time_indices'],int)]
index={(int(r),int(c)):i for i,(r,c) in enumerate(zip(rr,cc))}
oi=np.array([index.get((int(r),int(c)),-1) for r,c in zip(m['obs_rows'],m['obs_cols'])])
ok=(oy>=2010)&(oy<=2025)&(np.asarray(m['observed_results'])>0)&(oi>=0)
observed=np.bincount(oi[ok],minlength=len(rr))>0
contrasts=[(ref,s,'common reference') for s in steps if s!=ref]
for x,y,label in [
 ('step33_nestcbc_s41',ref,'Allee law'),
 (ref,'step34_abs_natdep_s41','native departure'),
 ('step34_abs_natdep_s41','step34_abs_v2_natdep_s41','DESK basis with native departure'),
 (ref,'step34_abs_v2_s41','DESK basis control'),
 (ref,'step35_est1951_ctl_mid_s41','initialization formulation control'),
 ('step34_abs_natdep_s41','step35_est1951_nat_mid_s41','initialization formulation native'),
 ('step35_est1951_ctl_mid_s41','step35_est1951_nat_mid_s41','later-state native departure'),
 (ref,'step36_nonest_ctl_s41','NestWatch control'),
 ('step34_abs_natdep_s41','step36_nonest_nat_s41','NestWatch native'),
 (ref,'step34_abs_broodlo','annual nest conversion low'),
 (ref,'step34_abs_broodhi','annual nest conversion high'),
 ('step29_yb_native',ref,'yearling baseline to absolute/nest/anchor'),
 ('step38_cbc1951_noncbc_free_s41','step38_cbc1951_free_s41','CBC likelihood free abundance seed41'),
 ('step38_cbc1951_noncbc_free_s42','step38_cbc1951_free_s42','CBC likelihood free abundance seed42'),
 ('step38_cbc1951_free_s41','step38_cbc1951_mid_s41','CBC initial abundance profile seed41'),
 ('step38_cbc1951_free_s42','step38_cbc1951_mid_s42','CBC initial abundance profile seed42')]:
    if x in cache and y in cache:contrasts.append((x,y,label))
for seed in [41,42]:
    base=f'step38_cbc1951_noncbc_free_s{seed}';historical=f'step40_rangev2_historical_s{seed}';invasion=f'step40_rangev2_invasion_s{seed}'
    for a,b,label in [(base,historical,'historical range replacement'),(historical,invasion,'dated1951-1965 range supervision')]:
        if a in cache and b in cache:contrasts.append((a,b,f'{label} seed{seed}'))
    control=f'step39_veit_fixed_s{seed}'
    floored=f'step39_kfloor4p0_s{seed}'
    floor_factor={
        'factor_p1_rational_adult20':f'step41_rational_adult20_s{seed}',
        'factor_p2_veit':f'step42_power2_veit_s{seed}',
        'factor_p2_rational_adult20':f'step42_power2_rational_adult20_s{seed}'}
    for label,parent in floor_factor.items():
        child=f'step43_{label}_s{seed}'
        if parent in cache and child in cache:contrasts.append((parent,child,f'floor with {label} seed{seed}'))
    f_p1r=f'step43_factor_p1_rational_adult20_s{seed}'
    f_p2v=f'step43_factor_p2_veit_s{seed}'
    f_p2r=f'step43_factor_p2_rational_adult20_s{seed}'
    for a,b,label in [(floored,f_p1r,'floored departure power1'),(floored,f_p2v,'floored curvature Veit'),
                       (f_p1r,f_p2r,'floored curvature rational'),(f_p2v,f_p2r,'floored departure power2')]:
        if a in cache and b in cache:contrasts.append((a,b,f'{label} seed{seed}'))
    for family in ['reprovar','settlement','native']:
        off=f'step43_{family}_floor0_s{seed}';on=f'step43_{family}_floor4_s{seed}'
        for a,b,label in [(control,off,'mechanism without floor'),(floored,on,'mechanism with floor'),(off,on,'floor within mechanism')]:
            if a in cache and b in cache:contrasts.append((a,b,f'{family}: {label} seed{seed}'))
    rational=f'step41_rational_adult20_s{seed}'
    for mode in ['veit_adult20','rational_shared','rational_adult20']:
        step=f'step41_{mode}_s{seed}'
        if control in cache and step in cache:contrasts.append((control,step,f'gradual departure {mode} seed{seed}'))
    for movement,parent in [('veit',control),('rational_adult20',rational)]:
        step=f'step42_power2_{movement}_s{seed}'
        if parent in cache and step in cache:contrasts.append((parent,step,f'crowding power2 {movement} seed{seed}'))
    power_veit=f'step42_power2_veit_s{seed}';power_rational=f'step42_power2_rational_adult20_s{seed}'
    if power_veit in cache and power_rational in cache:
        contrasts.append((power_veit,power_rational,f'departure under power2 crowding seed{seed}'))
    for mode in ['shared','adult_priority']:
        step=f'step39_{mode}_fixed_s{seed}'
        if control in cache and step in cache:contrasts.append((control,step,f'{mode} overflow seed{seed}'))
    for arm in ['kfloor1p25','kfloor4p0','priority_floor4']:
        step=f'step39_{arm}_s{seed}'
        if control in cache and step in cache:contrasts.append((control,step,f'{arm} seed{seed}'))
    for arm in ['survey','envelope']:
        step=f'step39_early_{arm}_s{seed}';base=f'step38_cbc1951_noncbc_free_s{seed}'
        if base in cache and step in cache:contrasts.append((base,step,f'early CBC {arm} seed{seed}'))
    x=f'step39_early_survey_s{seed}';y=f'step39_early_envelope_s{seed}'
    if x in cache and y in cache:contrasts.append((x,y,f'buffered envelope addition seed{seed}'))
    x=f'step39_adult_priority_fixed_s{seed}';y=f'step39_priority_floor4_s{seed}'
    if x in cache and y in cache:contrasts.append((x,y,f'floor with adult priority seed{seed}'))
for registry in ['MINOR_FLEXIBILITY_RUNS_20261103.json','OUTSTANDING_TRANSFER_RUNS_20261103.json']:
    new_registry=exp/registry
    if not new_registry.exists():continue
    specs=json.loads(new_registry.read_text())
    for s in specs:
        if s['reference'] in cache and s['step'] in cache:
            contrasts.append((s['reference'],s['step'],f"registered {s['family']} {s.get('value','')} seed{s['seed']}"))
    for seed in [41,42,43]:
        for sigma in ['s005','s010','s020']:
            x=f'step45_regional_{sigma}_s{seed}';y=f'step45_cell_iid_{sigma}_s{seed}'
            if x in cache and y in cache:contrasts.append((x,y,f'matched marginal noise {sigma} seed{seed}'))
contrasts=list(dict.fromkeys(contrasts))
records=[]
for x,y,label in contrasts:
    for era in eras:
        for reg,mask in regions.items():
            for support,w in [('all_land',mask.astype(float)),('fixed_observed_cells',(mask&observed).astype(float))]:
                metrics={}
                for metric in cache[x][era]:
                    a,b=cache[x][era][metric],cache[y][era][metric]
                    q=compare(a,b,w,metric in ['potential_lambda','full_process_lambda','biological_C','C','local_replacement'])
                    keep=np.isfinite(a)&np.isfinite(b)&(w>0)
                    if keep.any():
                        ww=w[keep];aa,bb=a[keep],b[keep]
                        q.update(left_mean=float(np.average(aa,weights=ww)),right_mean=float(np.average(bb,weights=ww)),
                            signed_mean_change=float(np.average(bb-aa,weights=ww)),
                            q90_abs_change=float(np.quantile(np.abs(bb-aa),.9)))
                        q['symmetric_relative_abs_change']=float(np.average(2*np.abs(bb-aa)/(np.abs(aa)+np.abs(bb)+1e-12),weights=ww))
                    metrics[metric]=q
                records.append({'left':x,'right':y,'contrast':label,'era':era,'region':reg,'support':support,'metrics':metrics})
out={'reference':ref,'fits':steps,'records':records}
(exp/'CROSS_MODEL_PROCESS_AUDIT_20261101.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
lines=['# Cross-model vital rates and aggregate growth processes','',
 f'Includes all {len(steps)} locally READY contribution audits with the same standard yearling-breeding life cycle. This includes Allee laws, DESK bases, native departure, journey penalties, nesting/annual conversion, founder anchors and initialization formulations, plus seeds. It does not include older unsupported/different-life-cycle versions. Every fit is compared against the same reference ({ref}); selected matched structural contrasts are also shown.','',
 'The question is whether different model versions trade vital rates while preserving aggregate growth and movement at matched locations and years. Within-era cross-model comparisons below answer that question; historical change within one fit is a separate question. Correlation is accompanied by absolute differences and exact crossings of 1.','',
 'All-land comparisons give each finite land cell equal weight. Fixed-observed support gives equal weight to the same cells with a positive modern BBS record, regardless of era or fitted model abundance. Potential lambda is annual-mean Sa+Sj*Fmax before density/Allee suppression, density-dependent disease and movement losses. Sa/Sj are the saved local survivals after density-dependent disease; reproduction is realized offspring per breeder, not Fmax. C and no-movement replacement use their respective fitted-stage exposure weights within each era; they therefore compare realized fitted-state processes, not responses at a common imposed density. Net flux is immigration minus successful emigration per starting individual. Cross-fit robustness does not establish statistical identification or agree on responses away from fitted states.','',
 '## Uniform metrics for the three illustrative structural contrasts','',
 'All rows below use 2010-2025, identical fixed-observed cells within each region, and equal cell weights (East 206; Great Plains 189; West 704). The primary Great Plains band is 100W-95W. Mean absolute changes are calculated cell by cell before averaging; they are not changes in regional means. These three diagnostics have different ranges and meanings, so their raw magnitudes are not a calibrated measure of relative identifiability.','',
 '| Contrast | Region | Mean absolute C change | Mean absolute potential lambda change | Mean absolute local replacement change |',
 '|---|---|---:|---:|---:|']
selected=['annual nest conversion low','Allee law','later-state native departure']
for label in selected:
    for region in ['east','plains','west']:
        r=next(r for r in records if r['contrast']==label and r['region']==region and r['era']=='2010-2025' and r['support']=='fixed_observed_cells')
        q=r['metrics']
        name='Great Plains' if region=='plains' else region.title()
        lines.append(f"| {label} | {name} | {q['C']['mean_abs_change']:.3f} | {q['potential_lambda']['mean_abs_change']:.3f} | {q['local_replacement']['mean_abs_change']:.3f} |")
lines+=['','For a consistent comparison between components and aggregates, the next table applies the SAME mean symmetric relative difference to every quantity: 100 * mean(2*abs(right-left)/(abs(left)+abs(right))). It is not conventional baseline-relative percent change and is not posterior uncertainty.','',
 '| Contrast | Region | Adult survival | Juvenile survival | Realized reproduction | Potential lambda | C | Local replacement |',
 '|---|---|---:|---:|---:|---:|---:|---:|']
for label in selected:
    for region in ['east','plains','west']:
        r=next(r for r in records if r['contrast']==label and r['region']==region and r['era']=='2010-2025' and r['support']=='fixed_observed_cells')
        vals=[100*r['metrics'][k]['symmetric_relative_abs_change'] for k in ['local_adult_survival','local_juvenile_stayer_survival','realized_offspring_per_breeder','potential_lambda','C','local_replacement']]
        name='Great Plains' if region=='plains' else region.title()
        lines.append('| '+label+' | '+name+' | '+' | '.join(f'{v:.1f}%' for v in vals)+' |')
lines+=['','## Interpretation of cross-model compensation','',
 'There is evidence for partial conservation of realized growth, but not a single invariant growth process across all structures. Three matched modern contrasts illustrate the distinction, using equal weights on fixed positive-BBS cells (206 eastern cells or 704 western cells). These supports differ from the earlier record-weighted summaries.','',
 '- Lowering annual nest conversion in the absolute control changes eastern juvenile survival and realized reproduction by mean symmetric relative differences of 23.2% and 24.6%, versus 3.5% for C. Mean potential lambda is 2.222 versus 2.249; mean absolute map changes are .069 for lambda, .026 for C and .044 for actual-state local replacement. This supports demographic compensation with relatively conserved realized growth in that contrast. Symmetric relative difference is 2*abs(right-left)/(abs(left)+abs(right)), averaged equally over the fixed cells; it is not an uncertainty interval.','',
 '- Changing relative to absolute Allee changes eastern mean potential lambda from 3.246 to 2.222 (mean absolute map change 1.032). Yet mean absolute changes in C and actual-state local replacement are .050 and .050. Thus fitted-state replacement is appreciably more stable than potential growth capacity: this is not conservation of the whole growth law.','',
 '- Adding native departure to the later-state middle-abundance fit changes western mean potential lambda from 1.778 to 2.253; mean absolute changes are .133 in C and .152 in local replacement, with 50/704 cells reversing between C<.95 and C>1.05. This is a movement-structure change that is not fully absorbed by vital-rate compensation.','',
 'These statements compare model versions at the same places and eras; none relies on historical changes within a model. Stable fitted-state replacement can reflect the count likelihood constraining trajectories while allowing different density dependence, maximum growth or transport mechanisms. Establishing an identifiable growth response requires testing common-state perturbations and uncertainty, not just fitted-state agreement.','',
 '## Matched structural comparisons in the East (all land)','',
 '| Contrast | Era | Adult survival means | Juvenile survival means | Realized reproduction means | Potential lambda means | Lambda mean absolute change | Lambda crossing cells | C mean absolute change | C crossing cells | Local replacement mean absolute change |',
 '|---|---|---|---|---|---|---:|---:|---:|---:|---:|']
f=lambda q:f"{q['left_mean']:.3f} to {q['right_mean']:.3f}"
for r in records:
    if r['contrast']=='common reference' or r['region']!='east' or r['support']!='all_land':continue
    q=r['metrics'];l=q['potential_lambda'];c=q['C']
    lines.append(f"| {r['contrast']} | {r['era']} | {f(q['local_adult_survival'])} | {f(q['local_juvenile_stayer_survival'])} | {f(q['realized_offspring_per_breeder'])} | {f(l)} | {l['mean_abs_change']:.3f} | {l['categories']['0.0']['source_sink_reversal_cells']} | {c['mean_abs_change']:.3f} | {c['categories']['0.0']['source_sink_reversal_cells']} | {q['local_replacement']['mean_abs_change']:.3f} |")
lines+=['','## All model versions versus common reference: modern East, all land','',
 '| Fit | Potential lambda means | Lambda mean absolute change | Lambda crossing cells | C mean absolute change | C crossing cells | Local replacement mean absolute change |',
 '|---|---|---:|---:|---:|---:|---:|']
for r in records:
    if r['contrast']!='common reference' or r['era']!='2010-2025' or r['region']!='east' or r['support']!='all_land':continue
    q=r['metrics'];l=q['potential_lambda'];c=q['C']
    lines.append(f"| {r['right']} | {f(l)} | {l['mean_abs_change']:.3f} | {l['categories']['0.0']['source_sink_reversal_cells']} | {c['mean_abs_change']:.3f} | {c['categories']['0.0']['source_sink_reversal_cells']} | {q['local_replacement']['mean_abs_change']:.3f} |")
lines+=['','## Registered new comparisons across the complete regional belt','',
 'The belt exhaustively partitions land at 25–49N: East east of 95W, Plains 100–95W, West west of 100W. These rows use fixed observed cells and report the same measures in every region. Biological C excludes abundance innovations; fitted C includes them. Legacy regional summaries above retain their original masks.', '',
 '| Contrast | Era | Region | Cells | Mean absolute biological lambda change | Mean absolute full-process lambda change | Mean absolute biological C change | Mean absolute fitted C change | Fitted C reversals |',
 '|---|---|---|---:|---:|---:|---:|---:|---:|']
for r in records:
    if not r['contrast'].startswith(('registered ','matched marginal noise')) or not r['region'].endswith('_belt') or r['support']!='fixed_observed_cells':continue
    q=r['metrics']
    if not q['C'].get('cells'):continue
    lines.append(f"| {r['contrast']} | {r['era']} | {r['region']} | {q['C']['cells']} | {q['potential_lambda']['mean_abs_change']:.3f} | {q['full_process_lambda']['mean_abs_change']:.3f} | {q['biological_C']['mean_abs_change']:.3f} | {q['C']['mean_abs_change']:.3f} | {q['C']['categories']['0.0']['source_sink_reversal_cells']} |")
lines+=['','Full JSON includes every contrast, three eras, seven regional supports and both observation supports, including correlations, continuous differences, crossing counts with exact/.01/.05 thresholds and finite-support coverage. No pair is declared stable solely because a correlation is high.']
(exp/'CROSS_MODEL_PROCESS_AUDIT_20261101.md').write_text('\n'.join(lines)+'\n')
print('CROSS MODEL FITS',len(steps),'RECORDS',len(records),flush=True)

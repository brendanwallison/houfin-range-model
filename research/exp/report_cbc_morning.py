"""Summarize recovered overnight joint-CBC fits without selecting on a few metrics."""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform
from functional_stability_audit import compare

repo=Path('/mnt/c/Dev/houfin-range-model');exp=repo/'research/experiments';runs=Path('/mnt/e/Datasets/houfin/grid54/runs')
points=json.loads((repo/'.research_runtime/CBC_MORNING_POINTS_20261102.json').read_text())
records=[];maps={}
for point in points:
    s=point['step'];out=runs/s/'out';paths={n:out/f'{n}.json' for n in ['pathology','front_speed','cbc_score']}
    if not all(p.exists() for p in paths.values()):continue
    d,f,c=[json.loads(paths[k].read_text()) for k in ['pathology','front_speed','cbc_score']]
    b=json.loads((out/'contribution/READY.json').read_text())
    with np.load(out/'contribution/annual_budget.npz') as z:
        rr,cc=z['land_rows'],z['land_cols'];tt=(z['years']>=2010)&(z['years']<=2025)
        den=z['input_adult'][tt].sum(0);C=np.divide(z['breeder_contribution'][tt].sum(0),den,out=np.full_like(den,np.nan),where=den>1e-20)
    with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
        xx,yy=rasterio.transform.xy(src.transform,rr,cc);lon,lat=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
    with np.load(out/'viz_ref/07_source_sink_fields.npz') as z:
        dead=z['allee_dead_mask'][rr,cc].astype(bool);K=z['K_modern'][rr,cc]
    maps[s]=C
    p=d['regions']['all_land']['modern']
    record={'step':s,'initial_multiplier':float(np.exp(point['log_initial_abundance'])),
      'nominal_bird_equivalent':point['conditional_bird_equivalent'],
      'Sa':p['Sa']['population']['q10_50_90'][1],'Sj_Sa':p['Sj_over_Sa']['population']['q10_50_90'][1],
      'ceiling_population_share':p['juvenile_ratio_above_095_population_share'],
      'Fmax':p['Fmax']['population']['q10_50_90'][1],
      'front_fitted':f['fitted']['10pct']['median_model_minus_observed_yr'],
      'front_frozen_release':f['frozen_release_only']['10pct']['median_model_minus_observed_yr'],
      'regions':{},'cbc':c,'BBS':point.get('paired_vs_ref'),
      'effective_budgets':b['regions'],'isolation':b['isolation'],
      'east_potential_growth':{e:d['regions']['east'][e]['lambda0']['land']['mean'] for e in ['introduction','modern']},
      'nest_readouts':d['nests']}
    for region,mask in [('East',(lon>=-80)&(lat>=25)&(lat<49)),('Great Plains',(lon>=-100)&(lon<-95)&(lat>=25)&(lat<49)),('West',(lon<-100)&(lat>=25)&(lat<49))]:
        keep=mask&np.isfinite(C)
        record['regions'][region]={'cells':int(mask.sum()),'C_land_mean':float(C[keep].mean()),
          'C_below_095_land_share':float((C[keep]<.95).mean()),'K_Allee_dead_land_share':float(dead[mask].mean()),
          'K_land_median':float(np.median(K[mask]))}
    move=Path('/mnt/e/Datasets/houfin/movement_followup_20261101')/s/'READY.json'
    record['movement']=json.loads(move.read_text()) if move.exists() else None
    records.append(record)
(exp/'CBC_MORNING_UPDATE_20261102.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
lines=['# CBC / later-state overnight update — November 2, 2026','',
 'Six optimizations completed and passed refinement checks: four joint CBC fits and two matched no-CBC controls. A post-fit execution error hid the GPU before a GPU-only diagnostic. The checkpoints and model fits were intact. CPU support was added to the external readout, the wrappers were corrected, and the six missing diagnostic suites were recovered locally without optimization. Duplicate watcher instances were also removed; collectors now use a lock and atomic status writes. Original logs/failed queue receipts remain as provenance.','',
 '**Interpretation:** CBC materially changes the inferred state/demography, but these first joint fits are not improvements across the full objective. CBC means are closer to observed levels than in the initial fixed-population calibration, while BBS scores degrade, adult survival falls, juvenile survival approaches its ceiling, fecundity demand rises and fitted invasion arrives earlier. Treat this as a reproducible observation/process conflict to resolve, not a reason to discard CBC or adopt a joint fit merely because it fits CBC better. The seasonal and reproductive-conversion sensitivities remain important.','',
 'All six use 1951 initialization, no release-cell Allee exemption, no old CBC backcast pin and no native departure. Initial multipliers are relative to the model reference density. Nominal bird-equivalents use the earlier conditional conversion convention, not an independently calibrated census or uncertainty interval. Free-abundance seed estimates are not posterior uncertainty.','',
 '| Fit | Initial multiplier | Nominal bird-equivalent | Adult survival | Juvenile/adult | Population near juvenile ceiling | Fmax | Fitted arrival error | Frozen-release arrival error |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in records:
    lines.append(f"| {r['step']} | {r['initial_multiplier']:.3f} | {r['nominal_bird_equivalent']:.1f} | {r['Sa']:.3f} | {r['Sj_Sa']:.4f} | {r['ceiling_population_share']:.1%} | {r['Fmax']:.2f} | {r['front_fitted']:.1f} | {r['front_frozen_release']:.1f} |")
lines+=['','Survival/Fmax are modern model-population-weighted medians. Ceiling means Sj/Sa>.95. Arrival error is model minus observation in years; negative is early. These fitted-state summaries do not establish vital-rate identification.','',
 '## CBC supervision and prediction','',
 '| Fit | Spatial observed/predicted mean | Temporal observed/predicted mean | Temporal observed/predicted detection | Historical training observed/predicted mean |',
 '|---|---|---|---|---|']
for r in records:
    q=r['cbc'].get('scores')
    if not q:continue
    fmt=lambda name:f"{q[name]['observed_mean']:.1f} / {q[name]['predicted_mean']:.1f}"
    t=q['temporal'];lines.append(f"| {r['step']} | {fmt('spatial')} | {fmt('temporal')} | {t['observed_detection']:.1%} / {t['predicted_detection']:.1%} | {fmt('historical_train')} |")
lines+=['','Unseen circle effects are integrated; other parameters are conditional on MAP. CBC spatial withholding is not independent BBS spatial transfer. Historical training counts above cover all regions, not just the founder cell. Aggregate mean calibration does not establish correct historical spatial allocation.','',
 '## Matched BBS prediction','',
 '| Fit | Temporal marginal change (90% interval) | Spatial marginal change (90% interval) |','|---|---|---|']
for r in records:
    q=(r['BBS'] or {}).get('diff',{})
    fmt=lambda k:'pending' if k not in q else f"{q[k]['mean']:.4f} [{q[k]['lo90']:.4f}, {q[k]['hi90']:.4f}]"
    lines.append(f"| {r['step']} | {fmt('T_marg')} | {fmt('S_real_marg')} |")
lines+=['','Common reference: no-CBC 1951 fixed-middle control. These exploratory block intervals condition on fitted points; they are not demographic credible intervals.','',
 '## East, Great Plains, West — contribution and establishment mechanism','',
 'Every regional row here uses equal land-cell weights and latitude 25–49N. East >=80W; primary Plains 100–95W; West <100W. C<.95 measures a greater-than-5% realized contribution deficit. K/Allee-dead selects positive potential growth but no positive equilibrium; it is not all unsuitable habitat.','',
 '| Fit | Region | Mean C | Land share C<.95 | K/Allee-dead land share | Median K |','|---|---|---:|---:|---:|---:|']
for r in records:
    for reg,q in r['regions'].items():
        lines.append(f"| {r['step']} | {reg} | {q['C_land_mean']:.3f} | {q['C_below_095_land_share']:.1%} | {q['K_Allee_dead_land_share']:.1%} | {q['K_land_median']:.5f} |")
lines+=['','## Matched aggregate-process changes: CBC versus no CBC','',
 'Same free-initial-abundance priors, matched seeds and modern equal-land support as the regional table. Similar regional mean C does not imply stable contribution geography. These comparisons show changes in fitted-state processes, not posterior identification.','',
 '| Seed | Region | Mean absolute C change | C map correlation | Direct reversals beyond .95/1.05 |',
 '|---|---|---:|---:|---:|']
for seed in [41,42]:
    x=f'step38_cbc1951_noncbc_free_s{seed}';y=f'step38_cbc1951_free_s{seed}'
    for reg,mask in [('East',(lon>=-80)&(lat>=25)&(lat<49)),('Great Plains',(lon>=-100)&(lon<-95)&(lat>=25)&(lat<49)),('West',(lon<-100)&(lat>=25)&(lat<49))]:
        q=compare(maps[x],maps[y],mask.astype(float),True)
        lines.append(f"| {seed} | {reg} | {q['mean_abs_change']:.3f} | {q['weighted_pearson']:.3f} | {q['categories']['0.05']['source_sink_reversal_cells']} |")
lines+=['','The broader [cross-model audit](CROSS_MODEL_PROCESS_AUDIT_20261101.md) now includes the matched CBC/no-CBC and free/fixed-abundance contrasts, with local replacement, potential lambda, component rates and alternative spatial support.','',
 '## Full diagnostic coverage and remaining interpretation','',
 '| Concern | Evidence/status |','|---|---|',
 '| Historical growth and timing (P1–P2) | Regional early/modern potential-growth values and fitted/frozen fronts are retained in JSON. Later initialization does not clear the historical opening or timing issues. |',
 '| Reproduction/age balance (P3–P5) | Fmax, nest likelihood readouts, adult survival, juvenile ceiling and matched seed/abundance contrasts included. Fixed annual conversion is retained in these first fits; uncertain-conversion runs remain queued. |',
 '| Source/sink, tiny K and establishment (P6–P8) | Actual contribution, retention/export, BIDE closure, isolation, effective survival, K and positive-growth/Allee-dead masks recovered. Regional rows above avoid population weights hiding deficient land cells. Finite-propagule results are in each movement JSON; deterministic tests do not identify stochastic establishment. |',
 '| Initialization (P9) | Free and fixed-middle comparisons complete; low profile running and high profile queued at the initial morning check. No initial-abundance confidence interval established. |',
 '| Distances, redistribution and movement loss (P10–P12) | Permanent-distance/tail and finite-propagule followups run automatically after source download/scoring. Full results or explicit pending status are in JSON. Distances are conditional on successful movers; seasonal CBC redistribution is separate. |',
 '| Projection flexibility, scale and corrections (P13–P15) | Same shared ecological field and structural source; new source-specific CBC nuisances are recorded. No claim that scale confounding or posterior geometry is solved. HMC feasibility of the new joint model is not tested. |','',
 'The low-abundance fit was running at the morning check. At 06:41 CST, the already-staged uncertain annual NestWatch conversion, 600-km winter mapping and broader circle heterogeneity tests were moved ahead of further abundance profiles and seed repeats, because the first four CBC fits reproduce the same compensation. Sources/checkpoints were unchanged. No new native-departure branch was added. The remaining A100 allocation had 23h38m at 06:25 CST; no extra allocation was requested. See [live queue status](CBC_ESTABLISHED_STATUS_20261101.md).']
(exp/'CBC_MORNING_UPDATE_20261102.md').write_text('\n'.join(lines)+'\n')
print('MORNING CBC FITS',len(records))

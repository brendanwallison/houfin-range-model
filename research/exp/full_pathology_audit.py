"""Evidence/coverage audit of every stored GRID54 run; no model execution.

Missing diagnostics are explicit. Descriptive statistics are not biological
pass/fail gates, and different input/life-cycle strata are never pooled.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import re
import numpy as np
from retrospective_checklist import inventory, finite_json, read
from range_watch import HostUnpickler


def observation_context(input_dir, fields, cache):
    if input_dir not in cache:
        with (Path(input_dir)/'metadata.pkl').open('rb') as stream:
            m=pickle.load(stream)
        years=np.asarray(m['years'])[np.asarray(m['obs_time_indices'],dtype=int)]
        mask=(years>=2010)&(years<=2025)&(np.asarray(m['observed_results'])>0)
        cache[input_dir]=(np.asarray(m['obs_rows'],int)[mask],np.asarray(m['obs_cols'],int)[mask],
                          np.asarray(m['observed_results'])[mask],float(m['pop_scalar']))
    rows,cols,y,scale=cache[input_dir]
    with np.load(fields,allow_pickle=False) as z:
        k=z['K_modern'][rows,cols]*scale
        dead=z['allee_dead_mask'][rows,cols]
        sink=z['lam_fundamental_modern'][rows,cols]<=1
    finite=np.isfinite(k)
    return {'positive_modern_records':int(len(y)),'finite_K_records':int(finite.sum()),
            'K_route_count_q10_50_90_at_positive_records':np.quantile(k[finite],[.1,.5,.9]).tolist() if finite.any() else None,
            'positive_record_share_K_below_005_route_counts':float((k[finite]<.05).mean()) if finite.any() else None,
            'positive_record_share_allee_dead':float(dead.mean()) if len(y) else None,
            'positive_record_share_fundamental_sink':float(sink.mean()) if len(y) else None}


def collect(root):
    rows=inventory(root/'runs',root.parents[1]/'processed/model_results')
    local_provenance=read(root/'monitor/local_map_observation_provenance.json') or {}
    cache={}
    for row in rows:
        run=root/'runs'/row['step'];out=run/'out'
        cfg=read(out/'resolved_config.json')
        if not cfg:cfg=read(run/'code/config/overlays'/f"grid54_{row['step']}.json")
        row['config']=cfg
        if not row['observation_log_available'] and row['step'] in local_provenance:
            proof=local_provenance[row['step']]
            cp=root.parents[1]/'processed/model_results'/f"age_map_float32_grid54_{row['step']}_quick90/map_checkpoint.pkl"
            if cp.exists():
                with cp.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
                if sha==proof['map_checkpoint_sha256']:
                    synthetic=re.search(r'\[synthetic\] observed counts replaced from (\S+)',proof['log_head'])
                    row['synthetic_observations']=synthetic.group(1) if synthetic else None
                    row['observation_log_available']=True
                    row['observation_provenance']='Local MAP log verified against identical remote MAP checkpoint SHA256'
        row['pathology']=read(out/'pathology.json')
        row['cbc_score']=read(out/'cbc_score.json')
        row['hmc_geometry']=read(out/'hmc_geometry/READY.json')
        row['hmc_screens']={k:v for k,p in [('diag',out/'hmc_screen/READY.json'),('block',out/'hmc_screen_block/READY.json')] if (v:=read(p)) is not None}
        row['precision']=read(root/'overnight'/f"precision_{row['step']}.json")
        row['state']='accepted' if row['valid'] else 'unaccepted_or_pending'
        for state in ('pending','running','failed','done','held_inputs'):
            if (root/'queue'/state/f"0600_{row['step']}.sh").exists():row['queue_state']=state
        row['founders_birds']=(row['pathology'] or {}).get('founders_birds_conversion_102_8')
        if not row['valid']:continue
        try:
            if cfg and row['fields'] and not row['synthetic_observations']:
                input_dir=cfg['input_dir'].replace('${HOUFIN_PROCESSED}',str(root.parents[1]/'processed'))
                row['positive_observation_context']=observation_context(input_dir,out/'viz_ref/07_source_sink_fields.npz',cache)
        except Exception as exc:row['observation_context_error']=str(exc)
        try:
            with (Path(row['refined_dir'])/'map_checkpoint.pkl').open('rb') as stream:
                params=HostUnpickler(stream).load()['params']
            row['parameter_summaries']={}
            for key,value in params.items():
                a=np.asarray(value,dtype=float)
                row['parameter_summaries'][key]={'size':int(a.size),'l2':float(np.linalg.norm(a.ravel())),
                    'max_abs':float(np.max(np.abs(a))),'scalar_or_small':a.tolist() if a.size<=8 else None}
            if row['founders_birds'] is None and cfg:
                pin=cfg.get('population_model',{}).get('pinned_sites',{})
                v=pin.get('log_inv_total',params.get('log_inv_total_auto_loc'))
                if v is not None:row['founders_birds']=float(102.8*np.exp(np.asarray(v)))
        except Exception as exc:row['parameter_read_error']=str(exc)
    return {'checked_utc':datetime.now(timezone.utc).isoformat(),'scope':'Stored TACC GRID54 run directories; earlier non-GRID54 archives are not re-audited here',
            'runs':rows,'counts':{'directories':len(rows),'accepted':sum(bool(r['valid']) for r in rows),
            'accepted_real':sum(bool(r['valid']) and not r['synthetic_observations'] and r['observation_log_available'] for r in rows)}}


def fmt(value):
    return 'missing' if value is None else f'{value:.3g}'


def dig(obj,*keys):
    for k in keys:
        if not isinstance(obj,dict):return None
        obj=obj.get(k)
    return obj


def median(obj,*keys):
    q=dig(obj,*keys,'q10_50_90')
    return q[1] if q else None


def render(data, path):
    rows=[r for r in data['runs'] if r['valid'] and not r['synthetic_observations'] and r['observation_log_available']]
    groups={s:i+1 for i,s in enumerate(sorted({r['input_dir'] or 'unknown' for r in rows}))}
    lines=['# Full stored-run pathology evidence and coverage audit','',f"Snapshot: {data['checked_utc']}",'',
      f"{data['counts']['directories']} GRID54 directories; {data['counts']['accepted']} accepted refinements; {len(rows)} accepted runs with real-observation logs. Synthetic and unknown-observation-provenance runs are excluded from the comparison tables, retained in JSON and listed below.",'',
      '**Correction to the morning note:** its few highlighted runs were a selective update. The automated biological readout covers rate/region/era summaries, founders and existing metrics; it does not establish that all fifteen pathologies were fully evaluated. The earlier retrospective table also covered only cached fields. This document makes that gap explicit rather than labeling unavailable checks as passes.','',
      'Every table includes every accepted real-data run in this stored GRID54 archive. Rows are descriptive evidence, not a pooled ranking: input basis, life cycle, observation model and fitted mechanisms differ. Earlier non-GRID54 archives and detailed map interpretation remain outside this snapshot. The JSON contains complete configs, available metrics, parameter norms, precision results and errors.','',
      f"Actual coverage: **{sum(bool(r['pathology']) for r in rows)}/{len(rows)} detailed biological replays; {sum(bool(r.get('front_speed')) for r in rows)}/{len(rows)} front replays; {sum(bool(r.get('positive_observation_context')) for r in rows)}/{len(rows)} positive-record K/Allee contexts; {sum(bool(r.get('parameter_summaries')) for r in rows)}/{len(rows)} parameter summaries; {sum(bool(r.get('precision')) for r in rows)}/{len(rows)} fixed-point precision replays.** These counts describe available evidence, not completed biological validation.",'',
      '## Checklist coverage and outstanding work','',
      '| ID | Evidence collected here | What remains unestablished |','|---|---|---|',
      '| P1 historical habitat/growth | Early/modern lambda and suitable area; regional trajectories where replayed | Independent validation of historical change and spatial-map review |',
      '| P2 expansion too fast | Fitted and frozen release-only front errors at both saved thresholds | Missing front replays for some historical runs; common fixed-threshold comparisons across models |',
      '| P3 fecundity/growth | Fmax and local lambda distributions; realized nest output where available | Correct lifecycle-specific biological calibration for every historical arm |',
      '| P4 juvenile ceiling | Resident Sj/Sa distribution and population share above .95 where replayed | Effective mover-weighted juvenile survival and observation-based weighting |',
      '| P5 compensation | Joint Sa/Sj/F/K/founder evidence, matched seed/assumption comparisons | Identification of a unique demographic explanation |',
      '| P6 western sinks | Land, fitted-population and observed-positive-record sink shares | Explicit immigration flux/replacement deficits; these weights are distinct |',
      '| P7 tiny K | Cached K and K at positive modern records in route-count units | Geographic continuity/map assessment and sensitivity of the absence mechanism |',
      '| P8 Allee-dead cells | Cached land share and positive-record share | Perturbed-establishment/starting-density experiments for every arm |',
      '| P9 founders | Fixed/fitted founder conversion; founder bracket comparisons | Joint early-CBC detection likelihood and release-history validation |',
      '| P10 movement tails | Fixed versus learned kernel configuration and saved scalar/vector parameters | Realized departure-weighted distances, tail masses and independent movement comparison |',
      '| P11 exodus/regathering | Departure/settlement structural switches | Actual departure curves and before/after redistribution maps |',
      '| P12 free Q escape route | Journey/decoupling structure, existing q_asymmetry metrics | Mover/stayer-weighted annual survival decomposition |',
      '| P13 projection flexibility | Shared/private/split config, manifold metrics and parameter norms | Consistent coefficient-angle/map/extrapolation audit |',
      '| P14 scale inflation | Parameter norms/maxima, amplitude and observer-effect settings | Effective field magnitude versus coefficient/amplitude compensation in common units |',
      '| P15 flexible corrections | Saved trend/drift/process-noise config and metrics | Spatial contribution maps and ablations demonstrating whether corrections bypass spread |','',
      '**No run is certified free of all pathologies.** A disabled mechanism can remove a particular escape route structurally; that is not a measured biological pass. Missing means unavailable, not zero.','',
      '## Comparison strata','']
    for inp,i in groups.items():lines.append(f'- Input {i}: `{inp}`')
    lines+=['','## P1–P5: demography and habitat history','',
      'Modern medians are weighted by fitted population. Lambda history below uses the existing whole-land diagnostic, not an independent estimate of habitat change. F-nest is the saved realized annual fecundity at nest locations, not fledglings per nesting attempt.','',
      '| Run | Input | Yearling | Early→modern lambda | Suitable area early→modern | Sa | Sj/Sa | Pop share ratio>.95 | Fmax | F-nest |','|---|---:|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        p=r['pathology'] or {};m=r['metrics'];modern=dig(p,'regions','all_land','modern') or {}
        y=bool((r['life_cycle'] or {}).get('yearling_breeding')) or (r['life_cycle'] or {}).get('form')=='veit'
        lines.append(f"| {r['step']} | {groups[r['input_dir'] or 'unknown']} | {y} | {fmt(m.get('early_mean_lambda'))}→{fmt(m.get('modern_mean_lambda'))} | {fmt(m.get('early_suitable_fraction'))}→{fmt(m.get('modern_suitable_fraction'))} | {fmt(median(modern,'Sa','population'))} | {fmt(median(modern,'Sj_over_Sa','population'))} | {fmt(modern.get('juvenile_ratio_above_095_population_share'))} | {fmt(median(modern,'Fmax','population'))} | {fmt(dig(p,'nests','nest_F_real','mean'))} |")
    lines+=['','## P2/P6–P9: fronts, sinks, K, establishment and founders','',
      'Front errors are model-minus-observed years at the saved 10% threshold; negative means early. K-low is the share of finite-K positive modern survey records with K < .05 route-count units. These are whole-domain records; western observed sink share uses the separate western audit. Allee-dead shares are whole-land and whole-domain positive-record shares.','',
      '| Run | Fitted front | Frozen release-only | West sink pop | West sink positive records | K-low positive records | Allee-dead land / positive records | Founders birds |','|---|---:|---:|---:|---:|---:|---|---:|']
    for r in rows:
        p=r['pathology'] or {};f=r.get('front_speed') or {};o=r.get('positive_observation_context') or {};w=r.get('west') or {}
        lines.append(f"| {r['step']} | {fmt(dig(f,'fitted','10pct','median_model_minus_observed_yr'))} | {fmt(dig(f,'frozen_release_only','10pct','median_model_minus_observed_yr'))} | {fmt(dig(p,'regions','west','modern','sink_population_share'))} | {fmt(dig(w,'observations','modern_bbs','record_fraction_fundamental_sink'))} | {fmt(o.get('positive_record_share_K_below_005_route_counts'))} | {fmt(dig(r,'cached','allee_dead_land_share'))} / {fmt(o.get('positive_record_share_allee_dead'))} | {fmt(r.get('founders_birds'))} |")
    lines+=['','## P10–P15: structural escape routes and numerical coverage','',
      'This table records structure, not a pass/fail verdict. Parameter norms and complete manifold/trend/Q diagnostics are retained in JSON. Precision means fixed-point float32/float64 replay exists; it does not imply optimizer-basin robustness.','',
      '| Run | Kernel bands | Departure / settlement | Journey | Shared field / split | Observer scale | Added corrections | Parameter summary | Precision replay |','|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        pop=(r['config'] or {}).get('population_model',{});journey=pop.get('journey',{});man=pop.get('manifold_prior',{})
        kernel='learned' if pop.get('dispersal_kernel',{}).get('band_weights') else 'fixed'
        dep=pop.get('dispersal_departure',{}).get('form','default')+('+native' if pop.get('native_departure',{}).get('enabled') else '')
        sett=pop.get('settlement',{}).get('form','default')
        js=','.join(k for k,v in journey.items() if v is True) or 'coupled/default'
        if pop.get('path_compounding',{}).get('enabled'):js+='+compounding'
        shared='tied' if man.get('fixed_coupling')==1 else man.get('form','default')
        observer=pop.get('observation',{}).get('observer_effect',{})
        corr=','.join(k for k in ('process_noise','st_correction','covariate_drift','habitat_change') if isinstance(pop.get(k),dict) and pop[k].get('enabled')) or 'no extra enabled flag'
        lines.append(f"| {r['step']} | {kernel} | {dep}/{sett} | {js} | {shared}/{pop.get('readout',{}).get('split',False)} | {observer.get('sigma_mode','default')} | {corr}; K trend retained | {'available' if r.get('parameter_summaries') else 'missing'} | {'available' if r.get('precision') else 'missing'} |")
    lines+=['','## Prediction and run availability','',
      'Legacy temporal/spatial log scores use their saved diagnostics and are not interchangeable with the local paired scorer. Paired comparison reference, uncertainty and parity live in the existing rescore artifacts; a direct matched contrast is needed before ranking configurations.','',
      '| Run | Temporal log score | Spatial log score | Biological replay | Front replay | Positive-record context |','|---|---:|---:|---|---|---|']
    for r in rows:
        h=r['metrics'].get('holdout',{})
        lines.append(f"| {r['step']} | {fmt(dig(h,'temporal','mean_lpd'))} | {fmt(dig(h,'spatial','mean_lpd'))} | {'available' if r['pathology'] else 'missing'} | {'available' if r.get('front_speed') else 'missing'} | {'available' if r.get('positive_observation_context') else 'missing'} |")
    contributions=[r for r in rows if r.get('contribution_budget')]
    if contributions:
        lines+=['','## S2–S4/S6: actual contribution and population budgets','',
          'Added after the initial audit. Exact frozen-trajectory replays, actual-state local replacement, BIDE flow accounting, Runge breeding contribution and current-stage-weighted contribution are distinct from the intrinsic-growth classifications above. Reported immigration and export refer to cell boundaries. Full era/cell arrays, nonlinear isolation and validation receipts are in CONTRIBUTION_RESULTS_20261101.md and its linked artifact root.','',
          '| Run | Region | C breeding | R breeding | C state | No regional immigration: 2025/fitted |','|---|---|---:|---:|---:|---:|']
        for r in contributions:
            c=r['contribution_budget']
            for reg in ['west','plains','broad_plains','east']:
                b=c['regions'][reg]['modern'];iso=c['isolation'][reg]
                lines.append(f"| {r['step']} | {reg} | {fmt(b['C_breeding'])} | {fmt(b['R_breeding'])} | {fmt(b['C_state'])} | {fmt(iso['regional_no_immigration_final_over_fitted'])} |")
        lines+=['',f'Contribution suite available for {len(contributions)}/{len(rows)} accepted real-data rows in this snapshot. Missing rows are not passes.']
    movement=[r for r in rows if r.get('movement_diagnostics')]
    if movement:
        lines+=['','## P10: realized successful permanent movement','',
          'Mean endpoint distance and tails among successful off-cell adult survivors and juvenile recruits to the next breeding census. These are conditional on successful movement, not distances of all birds or seasonal movements. Full finite-propagule comparisons are in MOVEMENT_ESTABLISHMENT_RESULTS_20261101.md.','',
          '| Run | Region | Mean km | Fraction >300 km | Fraction >1000 km |','|---|---|---:|---:|---:|']
        for r in movement:
            for reg in ['west','plains','east']:
                m=r['movement_diagnostics']['regions'][reg]
                lines.append(f"| {r['step']} | {reg} | {fmt(m['distance_km'])} | {fmt(m['tail_300km'])} | {fmt(m['tail_1000km'])} |")
    hmc=[r for r in rows if r.get('hmc_geometry') or r.get('hmc_status')]
    if hmc:
        lines+=['','## HMC geometry and efficiency (separate from biological P1–P15)','',
          'Full unconstrained posterior including transformation Jacobians and observer nuisance parameters. Curvature, roughness and sampler performance are separate evidence. Raw or prior-standardized Krylov Ritz ratios are not certified full Hessian condition numbers. No MAP-only result certifies HMC suitability. See HMC_DIAGNOSTIC_RESULTS_20261101.md for all cases and sampler metrics.','',
          '| Run | Geometry | Diagonal NUTS | Block-dense NUTS |','|---|---|---|---|']
        for r in hmc:
            status=r.get('hmc_status',{})
            label=lambda name:'complete' if status.get(name,{}).get('report') else status.get(name,{}).get('queue_state','not available')
            lines.append(f"| {r['step']} | {label('geometry')} | {label('nuts')} | {label('nuts_block')} |")
    lines+=['','## Excluded or unfinished runs','']
    for r in data['runs']:
        if r not in rows:
            reason='synthetic observations' if r['synthetic_observations'] else 'unaccepted/pending' if not r['valid'] else 'unknown observation provenance'
            lines.append(f"- {r['step']}: {reason}; queue={r.get('queue_state','not identified')}; checkpoint={r['checkpoint']}.")
    path.write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--render',type=Path)
    a=p.parse_args()
    data=read(a.render) if a.render else collect(a.root)
    data=finite_json(data)
    a.out.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    render(data,a.out.with_suffix('.md'))
    print(data['counts'])


if __name__=='__main__':main()

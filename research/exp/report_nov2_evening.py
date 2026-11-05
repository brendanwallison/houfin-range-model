"""Transparent regional review of newly completed range and movement fits."""
import json
from pathlib import Path
R=Path('/mnt/c/Dev/houfin-range-model');E=R/'research/experiments'
audit=json.loads((E/'FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json').read_text())
rows={r['step']:r for r in audit['runs']}
selected=[('Earlier no-CBC reference','step38_cbc1951_noncbc_free_s41'),
 ('Overflow adult priority, seed41','step39_adult_priority_fixed_s41'),
 ('Overflow adult priority, seed42','step39_adult_priority_fixed_s42'),
 ('Historical range only, seed41','step40_rangev2_historical_s41'),
 ('Historical + invasion range, seed41','step40_rangev2_invasion_s41'),
 ('Historical range only, seed42','step40_rangev2_historical_s42')]
lines=['# Evening review, November 2: new completions and departure tests','',
'The new departure proposal is implemented as six additional matched fits, with two existing Veit controls. It is separate from the completed overflow experiment. The primary movement comparison inherits the fixed nominal 1951 state, NestWatch and original pseudozeros, without CBC, a K floor or native-specific departure. This isolates movement rather than simultaneously changing historical supervision.','',
'## Newly completed results','',
'The following are population-weighted modern medians over each fit’s own population. They describe the fitted population; they are not fixed-cell map agreement or proof that the growth process is identified. Potential annual growth is Sa+Sj*Fmax before crowding, mate limitation and movement. Values near one for realized contribution do not clear potential-growth inflation.','',
'| Fit | Adult survival | Young/adult survival | Maximum offspring output | Potential annual growth | Fitted arrival error, years |',
'|---|---:|---:|---:|---:|---:|']
records=[]
for label,s in selected:
 if s not in rows:continue
 r=rows[s];m=r['pathology']['regions']['all_land']['modern']
 med={k:m[k]['population']['q10_50_90'][1] for k in ['Sa','Sj_over_Sa','Fmax','lambda0']}
 arrival=(r.get('front_speed') or {}).get('fitted',{}).get('10pct',{}).get('median_model_minus_observed_yr')
 lines.append(f"| {label} | {med['Sa']:.3f} | {med['Sj_over_Sa']:.3f} | {med['Fmax']:.2f} | {med['lambda0']:.2f} | {arrival if arrival is not None else 'pending'} |")
 records.append(dict(label=label,step=s,medians=med,arrival_error=arrival))
lines+=['','Negative arrival error means earlier than observed, using the existing 10%-of-occupied-density definition. The lower detection threshold is also retained in the underlying front audit; an unchanged median does not imply unchanged spread everywhere.','',
'## Regional demographic processes','',
'Contribution is the expected next-year breeders attributable to a starting breeder, including successful recruits elsewhere. Local replacement below is total-population replacement with movement removed at the fitted state. Export is successful off-cell next-year breeders per starting breeder. These are different quantities and retain their existing census definitions. K/Allee nonviability denotes cells with positive potential growth suppressed through the capacity/mate-limitation interaction; it is separate from genuinely poor demographic habitat. Regional contributions are ratios of totals, not percentages of source cells.','',
'| Fit | Region | Contribution | Local replacement | Successful export | K/Allee-nonviable land | Potential-growth sink land |',
'|---|---|---:|---:|---:|---:|---:|']
for rec in records:
 r=rows[rec['step']]
 for region,title in [('east','East'),('plains','Great Plains'),('west','West')]:
  b=r.get('contribution_budget',{}).get('regions',{}).get(region,{}).get('modern',{})
  if not b:continue
  lines.append(f"| {rec['label']} | {title} | {b['C_breeding']:.3f} | {b['local_total_replacement']:.3f} | {b['successful_export_per_breeder']:.3g} | {100*b['K_Allee_dead_land_share']:.1f}% | {100*b['potential_sink_land_share']:.1f}% |")
lines+=['','## Interpretation and outstanding checks','',
'Historical CBC range supervision has not produced the near-total movement shutdown or order-of-magnitude potential growth seen in the first overflow fit. The seed41 historical and invasion-supervised fits retain successful exchange and population-weighted potential growth around 2.6–2.7. That is not a clean bill of health: this growth remains high, the fitted invasion remains about three years early under the main timing definition, and separate survival components shift. The invasion-supervised fit also loses some spatial prediction in the Plains.','',
'| Fit versus earlier no-CBC reference | Temporal predictive difference [90% block interval] | Spatial real-record difference [90% block interval] |',
'|---|---:|---:|']
for rec in records:
 if not rec['step'].startswith('step40_'):continue
 p=R/f'.research_runtime/range_monitor/rescore_{rec["step"]}.json'
 if not p.exists():continue
 diff=json.loads(p.read_text()).get('paired_vs_ref',{}).get('diff',{})
 def fmt(k):
  b=diff.get(k)
  return 'pending' if b is None else f"{b['mean']:+.4f} [{b['lo90']:+.4f}, {b['hi90']:+.4f}]"
 lines.append(f"| {rec['label']} | {fmt('T_marg')} | {fmt('S_real_marg')} |")
lines+=['','These are exploratory prediction intervals conditional on fitted optima. Raw optimized objectives across different observation terms are not ranked as if they were the same likelihood.','',
'The full checklist remains applicable: historical growth opening, frozen-environment invasion, potential-growth inflation, age-survival extremes and compensation, western immigration dependence, Plains K/Allee artifacts, establishment at observed sites, source bridges, effective mover/stayer survival and distances, reproductive agreement, founder/seed sensitivity, predictive performance, parsimony and HMC geometry. The tables above provide regional levels; fixed-cell cross-model reversals, bridge topology, finite-propagule establishment, source-isolated timing, and curvature for the new historical-range fits still require their follow-up outputs. They are not implicitly passed.','',
'No candidate is selected from these preliminary completions. The second historical seed and remaining invasion seed provide the next replication checks. The six new movement fits will be collected and scored by the same persistent watchers.']
(E/'EVENING_UPDATE_20261102.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(records,indent=2))

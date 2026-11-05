"""Holistic review of recent fit families with uniform supports and explicit tradeoffs."""
import json
from pathlib import Path
R=Path('/mnt/c/Dev/houfin-range-model');E=R/'research/experiments'
audit=json.loads((E/'FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json').read_text())
rows={x['step']:x for x in audit['runs']}
cross=json.loads((E/'CROSS_MODEL_PROCESS_AUDIT_20261101.json').read_text())['records']
paired=json.loads((R/'.research_runtime/overflow_direct_paired.json').read_text())
names={'Control':'step39_veit_fixed_s41','Shared overflow':'step39_shared_fixed_s41','Adult priority':'step39_adult_priority_fixed_s41'}
def matched(label,region,era='2010-2025'):
    return next(x for x in cross if x['contrast']==label and x['region']==region and x['era']==era and x['support']=='fixed_observed_cells')
def number(x):return f'{x:.3f}'
lines=['# Recent results: joint pathology assessment, November 2','',
 'This review covers later established-state initialization, adding all-years CBC counts, uncertain annual reproductive conversion, and the first matched seed41 movement experiment. Overflow replicates, capacity floors and corrected historical-range fits are not complete at this review. The biological target remains a parsimonious joint explanation; no single vital rate, contribution mean, timing statistic or likelihood improvement selects a model.','',
 '## Later initialization: a useful formulation, not an identified initial state','',
 'Initializing an established release-cell population in1951 removes the ten-year release-cell Allee exemption and the old backcast anchor. That addresses a genuine spatial-resolution mismatch. Initial abundance, age composition and support remain conditional assumptions. The broad fixed280/560/1120 bird-equivalent profiles often reduce tiny-K/Allee nonviability but return juvenile/adult survival near .97–.98 in the no-native controls. Free-initialization no-CBC fits instead recover adult survival about .56 and juvenile/adult ratio about .47 across two seeds, with nominal initial equivalents near98–104. These differences show the initial-state/reproduction/process interaction matters; they do not turn the free estimate into an observed census.','',
 'The no-CBC baseline still has substantial historical habitat opening and overly fast frozen-environment expansion. Later initialization alone is not a solution to those pathologies. Holding initial abundance at the common nominal multiplier1 in the new movement trio closely reproduces the free baseline; it provides a matched control rather than another new biological mechanism.','',
 '## All-years CBC counts: added information exposes model conflict','',
 'The four initial joint fits reproduce a distinctive compromise: adult survival .28–.30, juvenile/adult ratio near1, Fmax about11.8–12.3, arrival about6 years early rather than3, and lower matched BBS temporal/spatial predictive scores. At the same time they improve CBC abundance correspondence and reduce positive-growth/K–Allee nonviability to roughly0–.3% of regional land. Those are real gains and real costs; reducing Allee-dead cells does not by itself make these fits preferable. Regional mean C can remain similar while maps and local replacement change. CBC is informative, but the current winter/count-to-annual-state correspondence is not established as adequate.','',
 'The600km seasonal-mapping sensitivity failed refinement convergence and is excluded, rather than treated as evidence against that displacement. The uncertain annual-conversion fit is accepted and adds another diagnostic:','',
 '| Fit | Adult survival | Juvenile/adult ratio | Fmax | Annual conversion | Annual output at nest sites | Predicted fledglings per attempt |','|---|---:|---:|---:|---:|---:|---:|']
for step,label in [('step38_cbc1951_noncbc_free_s41','No CBC, free initial state'),('step38_cbc1951_free_s41','CBC, fixed conversion'),('step38_cbc1951_broods_free_s41','CBC, uncertain conversion')]:
    row=rows[step];v=row['pathology']['regions']['all_land']['modern'];n=row['pathology']['nests']
    vals=[v[k]['population']['q10_50_90'][1] for k in ['Sa','Sj_over_Sa','Fmax']]
    vals += [n[k]['mean'] for k in ['nest_broods','nest_F_real','nest_mean_fledged']]
    lines.append('| '+label+' | '+' | '.join(number(x) for x in vals)+' |')
lines+=['','Survival/Fmax above are model-population-weighted medians; nest quantities are means at the same recorded nest cell-years. Annual conversion is the model’s effective annual multiplier, not an observed count of broods. Uncertain conversion rises to about4.82 on average, annual output rises from 5.62 to 6.42 while predicted per-attempt fledglings fall from 3.55 to 2.90, closer to the no-CBC baseline 3.02. Adult survival falls further and the juvenile ceiling remains. Thus annual-conversion uncertainty is consequential, and per-attempt NestWatch records alone do not fix annual fecundity or survival.','',
 '## Overflow: better expansion brake, a different fitted ecological explanation','',
 'All three below use1951 established-state initialization, identical fixed introduced abundance, standard yearling breeding, absolute aggregated Allee, one shared habitat field, NestWatch, no native-departure scalar and no CBC count likelihood. Only the departure allocation changes. Each is accepted after float64 refinement and has checked contribution/BIDE and movement audits. Only seed41 is complete, so basin/seed robustness remains open.','',
 '| Diagnostic | Control | Shared overflow | Adult priority |','|---|---:|---:|---:|']
for metric,label in [('Sa','Modern adult survival'),('Sj_over_Sa','Modern juvenile/adult survival'),('Fmax','Modern Fmax')]:
    vals=[rows[s]['pathology']['regions']['all_land']['modern'][metric]['population']['q10_50_90'][1] for s in names.values()]
    lines.append('| '+label+' | '+' | '.join(number(v) for v in vals)+' |')
for kind,label in [('fitted','Fitted arrival error, years'),('frozen_release_only','Frozen-habitat release-only arrival error, years'),('fitted_release_only','Fitted-habitat release-only arrival error, years')]:
    vals=[rows[s]['front_speed'][kind]['10pct']['median_model_minus_observed_yr'] for s in names.values()]
    lines.append('| '+label+' | '+' | '.join(number(v) for v in vals)+' |')
lines+=['','Negative arrival errors mean early. All use the10%-of-occupied-reference detection threshold. The frozen release-only improvement is substantial: about26 years early becomes9.5–11 years early. But fitted timing is unchanged, and the fitted-environment release-only counterfactual becomes27–31 years early versus2 years early in control. The interaction with the native population is therefore large and unresolved; the better frozen counterfactual cannot alone establish a correct early invasion mechanism. No origin-tagged ancestry reconstruction has been performed here.','',
 '### Lambda levels and realized processes on identical cells','',
 'The following means use exactly206 eastern,189 Plains and704 western cells with positive modern BBS records, equal cell weights, and2010–2025. Potential lambda=Sa+Sj*Fmax for the actual yearling life cycle, before density/Allee/movement suppression. C is breeder-to-breeder origin contribution; local replacement excludes movement at the actual state. C/replacement integrate each cell using its fitted stage exposure through the era, then average those cell values equally. This is cross-model fitted-state comparison, not proof of statistical identification or common-density response.','',
 '| Region | Potential lambda: control / shared / priority | Runge C: control / shared / priority | Local replacement: control / shared / priority |','|---|---|---|---|']
for region,title in [('east','East'),('plains','Great Plains'),('west','West')]:
    a=matched('shared overflow seed41',region);b=matched('adult_priority overflow seed41',region)
    groups=[]
    for key in ['potential_lambda','C','local_replacement']:
        groups.append(' / '.join(number(v) for v in [a['metrics'][key]['left_mean'],a['metrics'][key]['right_mean'],b['metrics'][key]['right_mean']]))
    lines.append('| '+title+' | '+' | '.join(groups)+' |')
lines+=['','This is severe P3 lambda inflation, not merely a trade among nonidentified vital-rate components. It is already present in1951–1965: eastern mean potential lambda is .770 in control versus10.246 shared and9.551 priority on these same cells. The early eastern field is no longer predominantly poor in potential growth, but its level becomes extreme. P1 historical-change magnitude and P3 absolute growth level must therefore be assessed separately. Near-one realized replacement in overflow reflects suppression/regulation of much larger potential growth and does not clear P3.','',
 '| Region | Mean absolute C change: shared / priority | Direct C reversals beyond .95/1.05: shared / priority |','|---|---:|---:|']
for reg,title in [('east','East'),('plains','Great Plains'),('west','West')]:
    a=matched('shared overflow seed41',reg)['metrics']['C'];b=matched('adult_priority overflow seed41',reg)['metrics']['C']
    lines.append(f"| {title} | {a['mean_abs_change']:.3f} / {b['mean_abs_change']:.3f} | {a['categories']['0.05']['source_sink_reversal_cells']} / {b['categories']['0.05']['source_sink_reversal_cells']} |")
lines+=['','### Establishment, exchange and distance','',
 'The next table uses the contribution audit’s regional land support, not the fixed-observed subset above. These are potential-growth-positive cells with K/Allee nonviability; fundamental demographic sinks are a separate class. The Plains band is100W–95W.','',
 '| Region | K/Allee nonviable land: control / shared / priority | Successful breeder export per starting breeder: control / shared / priority |','|---|---|---|']
for reg,title in [('east','East'),('plains','Great Plains'),('west','West')]:
    vals=[rows[s]['contribution_budget']['regions'][reg]['modern'] for s in names.values()]
    dead=' / '.join(f"{100*v['K_Allee_dead_land_share']:.1f}%" for v in vals)
    export=' / '.join(f"{v['successful_export_per_breeder']:.3g}" for v in vals)
    lines.append(f'| {title} | {dead} | {export} |')
lines+=['','Overflow makes successful exchange nearly vanish, especially in the East, while many more marginal cells are suppressed through K/Allee interactions. The control has65 saved K/Allee-nonviable cells over the whole domain, shared overflow824 and priority611. Finite-propagule tests retain this basic failure distinction; a few cached nonviable cells persist under the separate finite/disease replay, so the classification is not treated as an infallible hard label. Populations at positive BBS records are less often assigned tiny K under overflow, a genuine gain, but that does not describe the surrounding land.','',
 'Conditional on successful off-cell movement, mean distances increase rather than uniformly shorten: West215→262/317km; Plains242→322/388km; East237→217/294km (control→shared/priority). These are the very small set of successful movers, not all birds or seasonal movements. The rate of exchange and distance conditional on exchange must be kept separate.','',
 'Under the existing raw C>1 connectivity classification, none has a4-neighbor coast-to-coast source bridge in2010–2025. Each overflow fit has an8-neighbor bridge in3 of16 years, versus0 in control. Local-replacement bridges appear in6/16 and4/16 years respectively, versus0. Potential-lambda bridges are much more prevalent under overflow. Diagonal connectivity and distance from the source threshold must qualify any claim of a continuous biologically meaningful corridor.','',
 'A working mechanistic hypothesis is that the retention threshold is calibrated to a K that already determines no-movement demographic equilibrium. With Allee suppression, equilibrium can lie below that threshold. A strict excess-only rule can therefore nearly switch off export, prompting compensation in growth capacity and habitat/density allocation. Actual low exchange supports investigating this hypothesis, but it does not establish that adult priority itself is biologically wrong. The first test intentionally used inherited K units, not independently measured breeding-territory capacity.','',
 '### Prediction, parsimony and inference','',
 '| Direct comparison with matched control | Temporal marginal score change [90% block interval] | Spatial real-record score change [90% block interval] |','|---|---|---|']
for name,s in list(names.items())[1:]:
    q=paired[s];parts=[]
    for key in ['temporal','spatial_real']:
        a=q[key];parts.append(f"{a['mean']:+.4f} [{a['lo90']:+.4f}, {a['hi90']:+.4f}]")
    lines.append('| '+name+' | '+' | '.join(parts)+' |')
lines+=['','Both lose some temporal predictive performance; spatial differences are unresolved at this precision. These are paired exploratory intervals conditional on MAP, not posterior intervals. Overflow adds no learned scalar or new spatial/temporal field. That is a parsimony advantage, but similar parameter count does not ensure a credible fitted process. Geometry/HMC has not yet been measured for these new fits. The older absolute-Allee diagonal HMC screen took69 minutes and mixed poorly despite no divergences; that is evidence about that sampler/model setup, not a verdict on the new branches.','',
 '## Checklist accounting','',
 '| Concern | Recent assessment |','|---|---|',
 '| P1 historical growth change | Later initialization does not remove historical opening. Overflow raises early growth dramatically; absolute level and temporal change must both be reported. |',
 '| P2 invasion speed | Overflow improves frozen-habitat release-only timing, leaves fitted timing similar and changes native/release interaction counterfactuals markedly. |',
 '| P3 lambda/fecundity inflation | Strongly worse potential lambda under both overflow arms. Realized reproduction and per-attempt nest predictions are separate; smaller realized output does not excuse extreme potential growth. |',
 '| P4–P5 age balance and compensation | No-CBC free-state baseline avoids juvenile-ceiling saturation. Joint CBC and overflow return it; uncertain annual conversion exposes another compensator. |',
 '| P6 source/sink and subsidy | Runge maps and no-movement replacement change materially. Lower immigration is not automatically better: overflow nearly removes exchange. Regional totals do not establish cell-level stability. |',
 '| P7–P8 K and establishment | Later-state/CBC fits can reduce tiny-K/Allee nonviability. Overflow reduces tiny K at observed records but increases nonviable surrounding land. Genuine Plains inhospitability must remain distinct from this artifact. |',
 '| P9 initial state | Later-state formulation resolves the special release representation, but abundance/age/support remain assumptions. The overflow trio fixes the same state, preventing abundance refitting from explaining their difference. |',
 '| P10 movement tails | Successful-mover distances and tails often increase under overflow even while exchange becomes tiny. No new empirical permanent-dispersal identification is claimed. |',
 '| P11 departure/loss | Sparse-cell exodus is structurally reduced. The remaining question is whether the new retention law suppresses too much ordinary export. |',
 '| P12 settlement | No new settlement mechanism in these matched fits; no claim that this concern has been independently resolved. |',
 '| P13 shared-field compensation | All movement arms retain one shared habitat field. Large aggregate changes show that sharing the field alone does not identify the growth law. |',
 '| P14 scale/norm inflation | No added learned scale in overflow. Coefficient/field magnitudes and amplitude tradeoffs remain in archive diagnostics; no new norm-based clearance is asserted here. |',
 '| P15 flexible corrections | No extra process-noise, state-warp or regional correction mechanism added to the overflow trio. Effects arise through inherited demographic/field flexibility. |',
 '| Numerical/HMC | New fits are accepted and attribution checked; float32/64 trajectory sensitivity and HMC mixing of new variants remain untested. |','',
 '## Current judgment and unresolved tests','',
 'The established-state/no-CBC formulation remains a useful comparison baseline, not a selected final model. All-years CBC provides useful information but its current joint count mapping produces a reproducible conflict. Strict overflow provides a real expansion-brake benefit at no added parameter count, together with major growth inflation, loss of exchange and establishment artifacts. Adult priority moderates some costs relative to shared overflow but does not remove that fitted regime. Neither is shown to dominate the control across the checklist, and neither is discarded before the matched seed and capacity-floor results.','',
 'The queued K-floor tests directly address routing absence through tiny K; their effect on lambda, exchange and credible Plains limitations must still be measured. Corrected historical-range fits target occurrence/extent without the current all-years CBC abundance conversion. They have no completed optimization results yet. These are separate tests, not assumed solutions.','',
 'Evidence: CBC_MORNING_UPDATE_20261102.md; LATER_INITIALIZATION_UNIFIED_REVIEW_20261101.md; FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json; CROSS_MODEL_PROCESS_AUDIT_20261101.json; MOVEMENT_ESTABLISHMENT_RESULTS_20261101.md; PLAINS_SOURCE_CONNECTIVITY_20261101.json. Direct movement-control prediction contrasts were computed read-only from the same stored design/prediction arrays. Legacy metric prose describing a delayed-breeding eigenvalue is not used as the definition of current yearling potential lambda.']
import re
text='\n'.join(lines)+'\n'
text=re.sub(r'\b(in|use|fixed|near|multiplier|about|than|roughly|from|to|baseline|the|becomes|versus|exactly|and|is|has|overflow|priority|West|Plains|East|of|took|The|a|an)(?=\d)',r'\1 ',text)
text=re.sub(r'(?<=\d)(km|years)\b',r' \1',text)
(E/'RECENT_RESULTS_HOLISTIC_20261102.md').write_text(text)
print('WROTE RECENT_RESULTS_HOLISTIC_20261102.md')

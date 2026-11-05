# Recent results: joint pathology assessment, November 2

This review covers later established-state initialization, adding all-years CBC counts, uncertain annual reproductive conversion, and the first matched seed41 movement experiment. Overflow replicates, capacity floors and corrected historical-range fits are not complete at this review. The biological target remains a parsimonious joint explanation; no single vital rate, contribution mean, timing statistic or likelihood improvement selects a model.

## Later initialization: a useful formulation, not an identified initial state

Initializing an established release-cell population in 1951 removes the ten-year release-cell Allee exemption and the old backcast anchor. That addresses a genuine spatial-resolution mismatch. Initial abundance, age composition and support remain conditional assumptions. The broad fixed 280/560/1120 bird-equivalent profiles often reduce tiny-K/Allee nonviability but return juvenile/adult survival near .97–.98 in the no-native controls. Free-initialization no-CBC fits instead recover adult survival about .56 and juvenile/adult ratio about .47 across two seeds, with nominal initial equivalents near 98–104. These differences show the initial-state/reproduction/process interaction matters; they do not turn the free estimate into an observed census.

The no-CBC baseline still has substantial historical habitat opening and overly fast frozen-environment expansion. Later initialization alone is not a solution to those pathologies. Holding initial abundance at the common nominal multiplier 1 in the new movement trio closely reproduces the free baseline; it provides a matched control rather than another new biological mechanism.

## All-years CBC counts: added information exposes model conflict

The four initial joint fits reproduce a distinctive compromise: adult survival .28–.30, juvenile/adult ratio near 1, Fmax about 11.8–12.3, arrival about 6 years early rather than 3, and lower matched BBS temporal/spatial predictive scores. At the same time they improve CBC abundance correspondence and reduce positive-growth/K–Allee nonviability to roughly 0–.3% of regional land. Those are real gains and real costs; reducing Allee-dead cells does not by itself make these fits preferable. Regional mean C can remain similar while maps and local replacement change. CBC is informative, but the current winter/count-to-annual-state correspondence is not established as adequate.

The 600 km seasonal-mapping sensitivity failed refinement convergence and is excluded, rather than treated as evidence against that displacement. The uncertain annual-conversion fit is accepted and adds another diagnostic:

| Fit | Adult survival | Juvenile/adult ratio | Fmax | Annual conversion | Annual output at nest sites | Predicted fledglings per attempt |
|---|---:|---:|---:|---:|---:|---:|
| No CBC, free initial state | 0.562 | 0.477 | 8.313 | 2.500 | 3.574 | 3.019 |
| CBC, fixed conversion | 0.278 | 0.999 | 11.823 | 2.500 | 5.619 | 3.548 |
| CBC, uncertain conversion | 0.250 | 0.999 | 13.319 | 4.822 | 6.422 | 2.898 |

Survival/Fmax above are model-population-weighted medians; nest quantities are means at the same recorded nest cell-years. Annual conversion is the model’s effective annual multiplier, not an observed count of broods. Uncertain conversion rises to about 4.82 on average, annual output rises from 5.62 to 6.42 while predicted per-attempt fledglings fall from 3.55 to 2.90, closer to the no-CBC baseline 3.02. Adult survival falls further and the juvenile ceiling remains. Thus annual-conversion uncertainty is consequential, and per-attempt NestWatch records alone do not fix annual fecundity or survival.

## Overflow: better expansion brake, a different fitted ecological explanation

All three below use 1951 established-state initialization, identical fixed introduced abundance, standard yearling breeding, absolute aggregated Allee, one shared habitat field, NestWatch, no native-departure scalar and no CBC count likelihood. Only the departure allocation changes. Each is accepted after float64 refinement and has checked contribution/BIDE and movement audits. Only seed41 is complete, so basin/seed robustness remains open.

| Diagnostic | Control | Shared overflow | Adult priority |
|---|---:|---:|---:|
| Modern adult survival | 0.567 | 0.686 | 0.731 |
| Modern juvenile/adult survival | 0.473 | 0.999 | 0.999 |
| Modern Fmax | 8.378 | 18.291 | 14.853 |
| Fitted arrival error, years | -3.000 | -3.000 | -3.000 |
| Frozen-habitat release-only arrival error, years | -26.000 | -9.500 | -11.000 |
| Fitted-habitat release-only arrival error, years | -2.000 | -31.000 | -27.000 |

Negative arrival errors mean early. All use the 10%-of-occupied-reference detection threshold. The frozen release-only improvement is substantial: about 26 years early becomes 9.5–11 years early. But fitted timing is unchanged, and the fitted-environment release-only counterfactual becomes 27–31 years early versus 2 years early in control. The interaction with the native population is therefore large and unresolved; the better frozen counterfactual cannot alone establish a correct early invasion mechanism. No origin-tagged ancestry reconstruction has been performed here.

### Lambda levels and realized processes on identical cells

The following means use exactly 206 eastern,189 Plains and 704 western cells with positive modern BBS records, equal cell weights, and 2010–2025. Potential lambda=Sa+Sj*Fmax for the actual yearling life cycle, before density/Allee/movement suppression. C is breeder-to-breeder origin contribution; local replacement excludes movement at the actual state. C/replacement integrate each cell using its fitted stage exposure through the era, then average those cell values equally. This is cross-model fitted-state comparison, not proof of statistical identification or common-density response.

| Region | Potential lambda: control / shared / priority | Runge C: control / shared / priority | Local replacement: control / shared / priority |
|---|---|---|---|
| East | 2.258 / 11.505 / 10.458 | 0.876 / 0.994 / 0.995 | 0.779 / 0.999 / 0.998 |
| Great Plains | 1.237 / 9.311 / 9.075 | 0.771 / 0.998 / 0.999 | 0.704 / 0.999 / 1.002 |
| West | 1.692 / 10.264 / 9.483 | 0.816 / 0.975 / 0.978 | 0.696 / 0.976 / 0.978 |

This is severe P3 lambda inflation, not merely a trade among nonidentified vital-rate components. It is already present in 1951–1965: eastern mean potential lambda is .770 in control versus 10.246 shared and 9.551 priority on these same cells. The early eastern field is no longer predominantly poor in potential growth, but its level becomes extreme. P1 historical-change magnitude and P3 absolute growth level must therefore be assessed separately. Near-one realized replacement in overflow reflects suppression/regulation of much larger potential growth and does not clear P3.

| Region | Mean absolute C change: shared / priority | Direct C reversals beyond .95/1.05: shared / priority |
|---|---:|---:|
| East | 0.134 / 0.136 | 1 / 2 |
| Great Plains | 0.249 / 0.248 | 9 / 6 |
| West | 0.212 / 0.213 | 12 / 19 |

### Establishment, exchange and distance

The next table uses the contribution audit’s regional land support, not the fixed-observed subset above. These are potential-growth-positive cells with K/Allee nonviability; fundamental demographic sinks are a separate class. The Plains band is 100W–95W.

| Region | K/Allee nonviable land: control / shared / priority | Successful breeder export per starting breeder: control / shared / priority |
|---|---|---|
| East | 0.5% / 25.1% / 21.7% | 0.263 / 1.17e-07 / 8.27e-08 |
| Great Plains | 1.9% / 10.0% / 9.5% | 0.388 / 6.54e-05 / 2.66e-05 |
| West | 1.8% / 16.7% / 19.5% | 0.494 / 0.000401 / 0.000398 |

Overflow makes successful exchange nearly vanish, especially in the East, while many more marginal cells are suppressed through K/Allee interactions. The control has 65 saved K/Allee-nonviable cells over the whole domain, shared overflow 824 and priority 611. Finite-propagule tests retain this basic failure distinction; a few cached nonviable cells persist under the separate finite/disease replay, so the classification is not treated as an infallible hard label. Populations at positive BBS records are less often assigned tiny K under overflow, a genuine gain, but that does not describe the surrounding land.

Conditional on successful off-cell movement, mean distances increase rather than uniformly shorten: West 215→262/317 km; Plains 242→322/388 km; East 237→217/294 km (control→shared/priority). These are the very small set of successful movers, not all birds or seasonal movements. The rate of exchange and distance conditional on exchange must be kept separate.

Under the existing raw C>1 connectivity classification, none has a 4-neighbor coast-to-coast source bridge in 2010–2025. Each overflow fit has an 8-neighbor bridge in 3 of 16 years, versus 0 in control. Local-replacement bridges appear in 6/16 and 4/16 years respectively, versus 0. Potential-lambda bridges are much more prevalent under overflow. Diagonal connectivity and distance from the source threshold must qualify any claim of a continuous biologically meaningful corridor.

A working mechanistic hypothesis is that the retention threshold is calibrated to a K that already determines no-movement demographic equilibrium. With Allee suppression, equilibrium can lie below that threshold. A strict excess-only rule can therefore nearly switch off export, prompting compensation in growth capacity and habitat/density allocation. Actual low exchange supports investigating this hypothesis, but it does not establish that adult priority itself is biologically wrong. The first test intentionally used inherited K units, not independently measured breeding-territory capacity.

### Prediction, parsimony and inference

| Direct comparison with matched control | Temporal marginal score change [90% block interval] | Spatial real-record score change [90% block interval] |
|---|---|---|
| Shared overflow | -0.0293 [-0.0422, -0.0162] | -0.0122 [-0.0374, +0.0119] |
| Adult priority | -0.0182 [-0.0307, -0.0059] | -0.0017 [-0.0290, +0.0284] |

Both lose some temporal predictive performance; spatial differences are unresolved at this precision. These are paired exploratory intervals conditional on MAP, not posterior intervals. Overflow adds no learned scalar or new spatial/temporal field. That is a parsimony advantage, but similar parameter count does not ensure a credible fitted process. Geometry/HMC has not yet been measured for these new fits. The older absolute-Allee diagonal HMC screen took 69 minutes and mixed poorly despite no divergences; that is evidence about that sampler/model setup, not a verdict on the new branches.

## Checklist accounting

| Concern | Recent assessment |
|---|---|
| P1 historical growth change | Later initialization does not remove historical opening. Overflow raises early growth dramatically; absolute level and temporal change must both be reported. |
| P2 invasion speed | Overflow improves frozen-habitat release-only timing, leaves fitted timing similar and changes native/release interaction counterfactuals markedly. |
| P3 lambda/fecundity inflation | Strongly worse potential lambda under both overflow arms. Realized reproduction and per-attempt nest predictions are separate; smaller realized output does not excuse extreme potential growth. |
| P4–P5 age balance and compensation | No-CBC free-state baseline avoids juvenile-ceiling saturation. Joint CBC and overflow return it; uncertain annual conversion exposes another compensator. |
| P6 source/sink and subsidy | Runge maps and no-movement replacement change materially. Lower immigration is not automatically better: overflow nearly removes exchange. Regional totals do not establish cell-level stability. |
| P7–P8 K and establishment | Later-state/CBC fits can reduce tiny-K/Allee nonviability. Overflow reduces tiny K at observed records but increases nonviable surrounding land. Genuine Plains inhospitability must remain distinct from this artifact. |
| P9 initial state | Later-state formulation resolves the special release representation, but abundance/age/support remain assumptions. The overflow trio fixes the same state, preventing abundance refitting from explaining their difference. |
| P10 movement tails | Successful-mover distances and tails often increase under overflow even while exchange becomes tiny. No new empirical permanent-dispersal identification is claimed. |
| P11 departure/loss | Sparse-cell exodus is structurally reduced. The remaining question is whether the new retention law suppresses too much ordinary export. |
| P12 settlement | No new settlement mechanism in these matched fits; no claim that this concern has been independently resolved. |
| P13 shared-field compensation | All movement arms retain one shared habitat field. Large aggregate changes show that sharing the field alone does not identify the growth law. |
| P14 scale/norm inflation | No added learned scale in overflow. Coefficient/field magnitudes and amplitude tradeoffs remain in archive diagnostics; no new norm-based clearance is asserted here. |
| P15 flexible corrections | No extra process-noise, state-warp or regional correction mechanism added to the overflow trio. Effects arise through inherited demographic/field flexibility. |
| Numerical/HMC | New fits are accepted and attribution checked; float32/64 trajectory sensitivity and HMC mixing of new variants remain untested. |

## Current judgment and unresolved tests

The established-state/no-CBC formulation remains a useful comparison baseline, not a selected final model. All-years CBC provides useful information but its current joint count mapping produces a reproducible conflict. Strict overflow provides a real expansion-brake benefit at no added parameter count, together with major growth inflation, loss of exchange and establishment artifacts. Adult priority moderates some costs relative to shared overflow but does not remove that fitted regime. Neither is shown to dominate the control across the checklist, and neither is discarded before the matched seed and capacity-floor results.

The queued K-floor tests directly address routing absence through tiny K; their effect on lambda, exchange and credible Plains limitations must still be measured. Corrected historical-range fits target occurrence/extent without the current all-years CBC abundance conversion. They have no completed optimization results yet. These are separate tests, not assumed solutions.

Evidence: CBC_MORNING_UPDATE_20261102.md; LATER_INITIALIZATION_UNIFIED_REVIEW_20261101.md; FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json; CROSS_MODEL_PROCESS_AUDIT_20261101.json; MOVEMENT_ESTABLISHMENT_RESULTS_20261101.md; PLAINS_SOURCE_CONNECTIVITY_20261101.json. Direct movement-control prediction contrasts were computed read-only from the same stored design/prediction arrays. Legacy metric prose describing a delayed-breeding eigenvalue is not used as the definition of current yearling potential lambda.

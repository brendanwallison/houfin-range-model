# Historical lambda inflation: mechanisms, compensations and discriminating tests

November 3, 2026. This replaces the recommendation ordering in [the November 2 survey](LAMBDA_INFLATION_SYNTHESIS_20261102.md); it does not replace the archived experiment records. Lambda inflation here means an excessive historical increase in potential local growth, not a high modern level. Both can be pathological, but they are different questions.

The best-supported working explanation is **a conflict among retaining established populations, exporting and establishing colonists, and matching changes in observed counts. The model can partly resolve that conflict by making habitat progressively more productive.** None of the evidence identifies one missing mechanism. A change that improves one part of the conflict can expose or worsen another. The task is to find a parsimonious combination that reduces the need for habitat to carry population history, while retaining genuinely poor habitat and credible exchange.

This is not a ranking by historical lambda change. The informative objects are the *joint responses* of historical growth, realized replacement and contribution, immigration, establishment, demography, correction use and prediction. A numerical improvement can diagnose a missing process even when its implementation is biologically unacceptable. Conversely, an increase in temporal lambda change can expose a previously hidden compensation.

## What was actually re-examined

I revisited the original archived E020, E028, E043, E055/56, E063, E067, E070, E073, E074, E075, E080 and E081 records, the 54-km misspecification suite, its real-record predictive rescoring and dispersal synthesis, and the latest frozen-fit audits. These supplement the sixty-note inventory already archived with hashes in [SOURCE_MANIFEST.json](lambda_history_review_20261102/SOURCE_MANIFEST.json). This is not a new numerical replay of every old checkpoint. Withdrawn results, different life cycles and unconverged runs remain qualified.

I also implemented and ran [temporal_compensation_audit.py](../exp/temporal_compensation_audit.py) on fourteen collected, standard-yearling trajectories: 98 fit/region records. It checks identical grids and complete era coverage, uses equal weights on the same land cells, reports each era, and separates the adult-survival contribution from the combined juvenile-survival/reproduction contribution. Its means agree with the independently generated biological replay summaries to 8.9e-16. The decomposition is exact accounting, not causal identification. [Full table](TEMPORAL_COMPENSATION_AUDIT_20261103.md), [machine-readable results](TEMPORAL_COMPENSATION_AUDIT_20261103.json).

The uncollected larger-floor seed-42 budget is explicitly missing. No parameter estimates were fitted in this reanalysis. Current mechanism combinations and new temporal profiles are not represented as completed.

## The latest trajectories locate the compensation more precisely

The following are absolute changes in equal-land-cell mean potential lambda between 1902-1915 and 2010-2025. All entries use the actual yearling expression Sa + Sj*Fmax, not the older delayed-breeding eigenvalue. Geographic masks match the existing biological audit: its "East" is actually the northeastern mask east of 80W within 25-49N; Plains is 100-95W; West is west of 100W. The latter two retain the full audited land latitude range. These legacy masks are not an exhaustive three-region partition. The separate connectivity audit restricts paths to 25-49N; do not splice its percentages into this support. An exhaustive common-latitude partition is supplied below, so the analysis does not silently omit the central-eastern population.

| Configuration, seed 41 unless noted | East | Great Plains | West | Other consequential behavior |
|---|---:|---:|---:|---|
| Original departure, fixed 1951 state | +.973 | +.621 | +.646 | Reference; substantial exchange and remaining K/Allee failures |
| Smaller floor | +.975 | +.627 | +.649 | Almost unchanged processes; seed 42 closely agrees |
| Larger floor | +1.156 | +.724 | +.700 | Far fewer K/Allee failures; juvenile/adult survival approaches one |
| Gradual departure, reduced adult movement | +1.266 | +.928 | +.754 | Much greater local retention, less successful exchange, survival ratio near one |
| Strict adult-priority overflow | +.782 | +5.072 | +1.759 | Extreme growth levels and nearly absent exchange; smaller eastern increase is not improvement |
| Historical range supervision | +.937 | +.565 | +.611 | Free initial abundance; use its matched free-abundance control below |
| Historical plus dated invasion supervision | +.938 | +.518 | +.553 | Same lineage; improvement concentrated outside the East |
| Matched no-CBC free-abundance control | +.978 | +.614 | +.640 | Proper reference for the preceding two range arms |
| All-years CBC count likelihood | +1.885 | +1.884 | +.894 | Matched free-abundance lineage; winter/count compromise, survival ceiling and predictive cost |

The historical and dated-invasion conclusions broadly replicate in seed 42; the larger-floor and gradual-departure entries above are single-seed evidence. Range-only supervision is not the problematic all-years CBC count likelihood. The modest regional changes are not a declaration that the maps or initial abundance are identified.

As a spatial-support check, the same statistic on an exhaustive land partition within 25-49N is:

| Configuration | East, east of 95W | Plains, 100-95W | West, west of 100W |
|---|---:|---:|---:|
| Original fixed-state baseline | +1.267 | +.734 | +.819 |
| Larger floor | +1.425 | +.854 | +.902 |
| Gradual departure, reduced adult movement | +1.628 | +1.098 | +.958 |
| Strict adult-priority overflow | +4.232 | +6.001 | +2.142 |
| Matched free-state control | +1.259 | +.725 | +.809 |
| Historical range supervision | +1.211 | +.667 | +.772 |
| Historical plus dated invasion supervision | +1.204 | +.611 | +.707 |

The floor, gradual-departure and historical-range directions survive this support change. The apparent strict-overflow reduction in the narrower northeastern mask does not extend to the whole East. This illustrates why a selected region's improvement cannot establish a continental mechanism. Both supports and all seven region definitions are retained in the JSON.

Two distinctions matter:

* In the original baseline, eastern lambda progresses .285 -> .432 -> .628 -> 1.035 -> 1.257 across early, 1951-65, 1966-85, 1986-2009 and modern windows. Plains values are .252 -> .405 -> .457 -> .606 -> .873. About 65% and 67% of their total increases, respectively, remain between the early-BBS and modern means. The western share is about 50%. These are differences between window means, not a causal partition of individual calendar years. An early-release defect alone cannot account for the full fitted pattern.
* The larger floor reduces the adult-survival component of the whole-land temporal increase from +.082 to +.040, but increases the juvenile-survival/reproduction component from +.676 to +.803. Gradual departure gives +.017 and +.936. Lower adult survival or lower maximum output therefore need not mean less temporal growth inflation. This calculation does not separate juvenile survival from reproduction, and does not establish which parameter is biologically responsible.

Early lambda fields are extrapolated fitted quantities. An empty cell does not directly reveal its potential growth rate. In particular, no introduced eastern population exists before the later injection; low early eastern lambda is not itself an observation of poor breeding habitat. Historical extent and frozen-environment invasion counterfactuals are useful constraints, but neither alone validates the inferred habitat trajectory.

## Mechanism 1: local retention, useful export and long-distance loss are being traded

**Evidence.** Strict overflow nearly eliminated successful exchange and the optimizer produced extreme productive capacity; the floor does not repair its equilibrium/threshold mismatch. The new gradual-departure fit retains far more local production but also lowers successful export from .263/.388/.494 to .133/.146/.215 per breeder in East/Plains/West, while increasing historical growth change. Earlier learned long tails reduced historical change by about .20 across two seeds with essentially neutral held-out real-record prediction. However, those tails sent about 60% of juvenile kernel mass beyond 483 km and implied a mean around 625 km. The dispersal record explicitly proposed loss into sparse/sink habitat as one explanation. The season-filtered BBL encounters challenge that distance distribution, but are not an unbiased kernel sample.

**Working explanation.** The model may need established cells to retain enough breeders while exporting some productive young. It may also need *where* arrivals land to be less diffuse. A single departure law over a broad convolution kernel links retention, geographic redistribution and losses together. Habitat change can compensate when that balance is wrong.

**Competing explanation.** The long-tail improvement may primarily be a hidden mortality or dilution effect, not useful redistribution. The latest gradual result alone cannot decide: it changes both the density curve and the adult/juvenile ratio. Its queued curve-only and age-only controls are essential.

**Discriminating evidence.** Compare origin-to-destination successful contributions, journey loss, destinations with demographic deficits, local retention and frozen-environment fronts. At fixed fitted parameters, a long-tail versus original-kernel replay is a diagnostic perturbation, not an estimated treatment effect. Separate extra losses from birds successfully breeding in useful destinations. A causal allocation claim needs refitting matched mechanisms; a forward swap alone cannot establish it.

**Implication for combinations.** Limited placement of arriving birds, crossed with a floor, is worth testing because it changes destination concentration without expanding the continental kernel tail or moving residents. It should reduce lambda inflation only if useful establishment or native persistence improves sufficiently to reduce habitat gating. If the benefit requires concentration into a few artificial super-sources, it recreates the old exodus/regathering problem. Older E074 settlement fits preserved the Plains trough and raised Plains K, yet retained high fecundity and lost some predictive skill. That is support for a bounded experiment, not proof of rescue.

## Mechanism 2: a persistent habitat trajectory is absorbing in-era departures from the deterministic population model

**Evidence.** E070's small abundance noise could give comparable held-out prediction with a different demographic distortion: Sa about .78, Sj about .11 and more Plains Allee failures. This lowered a growth *level* and did not establish a temporal-inflation cure. The later 54-km fecundity deviations did directly reduce historical change by about .08 in the two prediction-neutral seeds. Larger resets reduced historical change more, but the population without those resets spread much too early and the resets explained excessive variation. A single free reset in 1964 or 1966 was barely used; repeated in-era resets were heavily used.

**Working explanation.** Some in-era variation is not well represented by the deterministic demographic trajectory. A persistent habitat response may partly stand in for it. The unused early reset argues against treating all remaining pressure as an accumulated release-state error; it does not prove the historical starting state is correct.

**Competing explanation.** Flexible terms may simply exploit an alternative optimizer basin or supply the growth/history that the biological model should explain. E070 and the later good/bad seeds make both possibilities concrete. A time series constrained to zero mean and zero linear trend is not guaranteed to have zero population effect: multiplication, density dependence and state dependence can make a centered term consequential.

**Implication for combinations.** The already queued modest reproductive-variation x floor comparison tests whether temporal pressure can be relieved without using tiny K to retain absences. It is *not* an exact E070 replay: it acts on fecundity, is low rank, and is active in 1966-2020. Its scientific attraction is the joint response: lower historical change, adequate native persistence and establishment, limited correction dependence, credible survival and neutral transfer. A smaller skeleton-only lambda change is insufficient. Report the correction's actual effect on reproductive growth and the full trajectory, plus the no-variation counterfactual. The zero-variation replay is descriptive sensitivity; reoptimizing without variation is the proper structural control.

## Mechanism 3: count variation and biological variation remain partly confounded

**Evidence.** The older count-only regional-year correction reduced historical change by about .27, even though it did not change dynamics. It also worsened held-out real-record prediction by about .037 nats/record, with the recorded 90% block interval excluding zero. Current all-years CBC count fits increase historical growth change and push age survival toward equality, whereas historical extent/presence supervision produces much smaller changes with a different geographic pattern. Route effects previously reduced capacity collapse, but unconstrained route-effect scales could become extreme. The corrected bounded versions did not eliminate other demographic problems.

**Working explanation.** Some regional count patterns can be matched either by changing true population history or by changing how the surveys see it. Historical winter abundance and breeding-season abundance are not interchangeable in a partially migratory population.

**Competing explanation.** A counts-only effect can imitate real biological variation. Its ability to lower temporal inflation does not prove observer effort is the cause. Its predictive cost argues against adopting that old correction as a solution.

**Implication.** Keep the later-state plus historical range/presence track. Do not treat early hull-derived absences as measured CBC nondetections, or count them again in an independent likelihood. Preserve original-versus-derived pseudozero footprint and strength comparisons. Modern eBird and CBC observation models remain useful independent constraints when their effort, seasonal correspondence and overlap are handled; merely adding more counts need not separate biological components. NestWatch informs realized reproduction per attempt, while uncertain annual conversion and BBL encounters constrain different parts of the demographic trade. They should be integrated through their appropriate observation processes, not converted into hard raw-rate targets.

## Mechanism 4: tiny capacity hides productive habitat; a floor exposes the remaining conflict

**Evidence.** E055 fixed a weak mating scale, but K collapsed instead: pinning the scale did not prevent positive-growth cells becoming nonviable. Relative Allee laws removed that particular mechanism by construction and shifted other rates. The latest larger absolute floor removes much of the problem while Plains fundamental sinks remain common, but temporal growth change and the juvenile survival ceiling worsen. This is exactly why a floor must be evaluated as a component of a configuration.

**Working explanation.** The model was using cheap tiny capacity to express persistent absence despite otherwise productive demography. Once that route is constrained, it must fit absences and abundances through demographic quality, movement, observation or temporal response. The larger floor reveals this reallocation; it does not identify the true alternative.

**Discriminating evidence.** In paired floor on/off fits, locate where absence pressure moves: does poor habitat become a genuine demographic sink, do observed native populations become more immigration-dependent, does the habitat improve more sharply over time, or do observation/correction effects inflate? A reduction in the number of K/Allee failures alone cannot answer this.

**Implication.** Re-test reproductive variation and limited arrival placement with the common floor, as already queued. The floor is not expected to rescue the independent strict-overflow departure threshold mismatch. It also does not guarantee viability for every cell with potential growth just above one; bad demography must remain possible. Establishment must be tested across actual starting densities and observed locations, not inferred from K divided by the fixed mating scale alone.

## Mechanism 5: habitat representation and response flexibility permit excessive change, but indiscriminate damping can hide real change

**Evidence.** E080 static/tied temporal channels greatly reduced century-scale change and retained a stronger Plains trough, while static models still demanded high reproduction. Some static fits reached iteration caps, and some free/calibrated controls stalled; those are incomplete comparisons, not clean adoption evidence. E075's rank-32 split readout produced plausible aggregate growth and few Allee failures but flattened the Plains trough and required extreme temporal survival directions. E067's habitat correction reduced western sink area while increasing fecundity and fitting substantial regional habitat change. Improvements in spatial ecology and deterioration in demographic/time interpretation can coexist.

**Working explanation.** BBS counts can reward particular temporal directions of the habitat field that are insufficiently constrained by ecological information. Because a shared field feeds several rates, its change can be amplified through juvenile survival times reproduction. A shared field reduces the number of response functions; it does not establish that its temporal projection is correct.

**Competing explanation.** Some actual habitat change is real, and suppressing it can force demographics or movement to absorb it. Poor temporal calibration in the old DESK is not a calibration of the improved basis. A time/position placebo measures predictability, not the cause of a trend. Low lambda change by construction is not evidence of an improved explanation.

**Implication.** A current-basis temporal restriction is useful as a controlled diagnostic, not as the initial preferred biological model. Hold spatial information fixed and report what observations, regions and eras resist less temporal change. Do not transplant the old .14 temporal scaling, add independent demographic fields, or assume a zero-change habitat is true. Evaluate fixed-cell growth maps, not only parameter amplitudes or continental means.

## Mechanism 6: numerical and optimizer effects select among explanations; they are not a complete ecological explanation

**Evidence.** E028 cold and warm starts reached qualitatively different departure regimes. Later reproductive-variation seeds separated a worse predictive basin from two prediction-neutral fits. The E063 fine sweep overturned an apparent precision cure: both independent-occupancy and mean-field versions remained sensitive near their own establishment transitions. Some older route-scale results were withdrawn and some settlement fits used confounded numerical regimes. Conversely, plausible synthetic truths could be fitted without automatically producing extreme growth.

**Interpretation.** Basin selection and weak identification qualify all causal claims. They do not justify throwing away every imperfect recovery result, nor dismissing a reproducible compensating pattern as merely numerical. Refit matched seeds and compare aggregate processes as well as vital rates. A parameter change with stable growth surfaces is a different result from the current larger-floor and gradual fits, which change temporal growth and exchange as well.

For HMC, retain existing curvature/roughness checks as cheap screens and judge inference on the regional temporal changes, growth surfaces, contribution and exchange, not solely individual parameters. A smooth mean-field change can still leave a narrow establishment ridge. Full HMC pilots are too costly to use as the first selector for every variant, and no Hessian statistic establishes mixing.

## The combined explanation, and what would falsify it

The leading combined hypothesis is that the model needs to express **established-population persistence, delayed colonization and in-era fluctuations on different effective scales**, but current crowding/departure/arrival rules and sparse auxiliary information tie those tasks together too tightly. Historical habitat improvement is one compensator; tiny K is another. This is a hypothesis about the model's division of labor, not evidence that historical habitat did not change.

This suggests a parsimonious candidate assembled from already motivated components: later established-state initialization; the shared habitat field; weak absolute clumped mating; a capacity floor that prevents the cheap absence route; modest, bounded in-era reproductive variation; and, if supported, limited concentration of arriving dispersers. Original gradual Veit departure remains the primary control. The new rational/adult-retention rule must earn its place through its age-only and curve-only contrasts. Native departure remains secondary. This is a candidate to test, not a recommendation to enable all terms at once.

The hypothesis weakens if credible local placement and modest reproductive variation leave temporal inflation unchanged, or if gains require correction-driven abundance, extreme age survival, new western subsidies or loss of the Plains barrier. It also weakens if a constrained temporal profile has a large, reproducible data-likelihood cost localized to independent ecological observations rather than count artifacts or front timing. If useful redistribution is the key, successful breeding destinations and native persistence should improve; a benefit explained mostly by extra losses supports the alternative loss hypothesis.

## Concrete tests, ordered by the question they resolve

| Test | Existing or new work | Distinguishing outcome and next action |
|---|---|---|
| Separate age retention from departure-curve shape | Existing step41 controls, both seeds; do not discard because the combined first fit is worse | If reduced adult departure alone changes survival/temporal growth adversely, retain the original age rule. If curve shape improves establishment/exchange without that cost, retain only that component. No result yet supports assuming the combined rule is needed. |
| Does crowding shape change the need for temporal habitat gating? | Existing step39/41/42/43 factorial: power 1/2 x original/combined departure x floor off/on, two seeds | Lower Fmax is not success. Look for reduced historical change with credible joint survival, establishment, timing and predictive transfer. If only growth level changes, label it a different gain. If the floor undoes a benefit, inspect how absence was previously achieved. |
| Does limited reproductive variation relieve in-era temporal pressure? | Four existing step43 variation fits, matched floor on/off controls | Prefer a replicated reduction of temporal change that survives forecasts and does not require large full-versus-no-variation trajectory differences or hidden persistent growth. If it only swaps survival extremes, the missing process is not resolved. |
| Does arrival concentration improve useful establishment rather than manufacture sources? | Four existing step43 settlement fits, matched floor on/off controls | Check contributions by origin/destination, native persistence, finite-propagule establishment, final movement distances and both local and continental source bridges. A lower temporal trend with excessive regathering is diagnostic rather than adoptable. |
| Do historical constraints remain helpful once tiny capacity is unavailable? | Proposed next matched transfer: original versus derived historical/invasion supervision x floor off/on, with identical initialization assumptions and both seeds | Reuse free-abundance controls only with free-abundance counterparts; do not silently compare them to fixed-state controls. The missing floor-on historical and invasion arms would require four new fits, not an unrestricted grid. A gain confined to geometry versus one in abundance/demography identifies what the historical data are actually constraining. Do not add all-years CBC counts simultaneously. |
| Where does the data resist less temporal change? | Proposed bounded constrained-MAP diagnostic, after selecting a useful structural contrast | Profile two reductions (25%, 50%) of regional historical lambda change, initially on the original baseline and one informative mechanism combination. Reoptimize compensators and report achieved constraints, real-data likelihood by source/region/era, absence penalties and prior separately. Less temporal change is a probe, not new ecological truth. If the optimizer simply raises early growth or creates western sinks/Plains bridges, expose that response rather than call the profile a success. |

The last diagnostic should keep a common fixed cell support and record both early and modern levels, not constrain one global scalar that slopes can evade. Report post-BBS change as well as century change; the original trend is not solely a pre-1966 problem. A constrained MAP path is not a posterior credible interval or a prior-free likelihood profile. Its penalty must be removed when reporting data cost. Multiple starts are required before attributing a large cost to the observations.

**Current execution decision:** preserve the existing running/queued fits and their immutable configurations. Their factorial controls address real competing mechanisms and should not be replaced by a new score-ranking queue. No new TACC fits or allocations are submitted by this synthesis. The two new follow-on designs above are explicitly proposed, not falsely described as implemented. Reuse their existing controls and choose the next new arm from the observed compensations, rather than first collecting a large Cartesian product.

## How to judge a combination

Use the same East/Plains/West locations, era definitions and weights in every direct contrast. Equal-cell all-land summaries expose extrapolated habitat; fixed observed-location summaries assess the habitats known to contain birds. Neither can substitute for the other. Report continuous map differences and threshold-crossing counts, not correlation alone. No new scalar pathology score is introduced.

Carry all P1-P15 concerns: historical habitat change; fitted and frozen-environment invasion timing; maximum and realized reproduction; age-survival ceiling and effective mover/stayer survival; demographic compensation; western native persistence and immigration deficits; tiny K; K/Allee failure and finite-propagule establishment; initial abundance/age/support assumptions; permanent distances and tails; exodus/regathering; journey survival replacing Sj; habitat-response flexibility; field/observation scale inflation; correction-driven growth. Add prediction, seed robustness, precision and HMC geometry as separate criteria. Missing checks remain missing.

Do not demand a source everywhere, zero immigration, no Plains bridge under every possible threshold, or zero habitat change. Ask whether the configuration supplies a credible explanation with fewer compensating extremes, and whether remaining biological conclusions are stable at the process level. A useful failure should specify the mechanism it implicates and the alternative it has not excluded.

## Literature and project provenance

The process-versus-observation separation, auxiliary-data integration and identifiability checks follow established state-space practice, rather than a new general modeling framework: [Auger-Methe et al. 2021](https://doi.org/10.1002/ecm.1470). Their review supports separating uncertainty sources and validating estimability; it does not identify which source drives this project. Targeted model checks rather than one predictive score are consistent with [Conn et al. 2018](https://doi.org/10.1002/ecm.1314). Resource-based alternatives to linear density dependence motivate the crowding comparison, but neither its exponent nor a reduction of historical growth inflation is established for House Finches by [Fronhofer et al. 2024](https://doi.org/10.1111/oik.09824).

The original spatial invasion framework remains [Veit and Lewis 1996](https://doi.org/10.1086/285924); its reproduction and movement assumptions differ from our current model. Historical range weighting, winter mapping, K floors and the local placement fraction remain explicit project assumptions, not literature-calibrated constants.

Key records: [misspecification suite](GRID54_LADDER_2026-10-30_MISSPEC_SUITE.md), [real-record rescoring](GRID54_SREAL_RESCORE_2026-10-30.md), [dispersal synthesis](DISPERSAL_SYNTHESIS_2026-10-31.md), [original E070](lambda_history_review_20261102/notes/E070_process_noise_design.md), [original E074](lambda_history_review_20261102/notes/E074_settlement_revisited.md), [original E075](lambda_history_review_20261102/notes/E075_numerics_retests.md), [original E080](lambda_history_review_20261102/notes/E080_temporal_change.md), [current floor combinations](KFLOOR_COMBINATION_EXECUTION_20261102.md), [historical range design](HISTORICAL_RANGE_IMPLEMENTATION_20261102.md), [full pathology checklist](HOUSE_FINCH_PATHOLOGY_CHECKLIST.md), [contribution results](CONTRIBUTION_RESULTS_20261101.md), [connectivity](PLAINS_SOURCE_CONNECTIVITY_20261101.md). Older predictive figures are conditional on their basis, life cycle and data design; no cross-family numerical pooling is used here.

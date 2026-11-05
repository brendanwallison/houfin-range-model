# Overnight experiment progress, November 3, approximately 06:20 CST

The six departure fits, four unfloored crowding-curvature fits and six new floored factorial fits have completed accepted float64 refinement and contribution replay. All four standalone floor fits are complete. Reproductive-variation and local-arrival-placement fits are still in progress or awaiting refinement; no biological result from those families is claimed here. Private-field decoupling and independent cell-year abundance corrections have not been implemented/launched by this batch.

## Historical lambda inflation: the crowding intervention has a replicated effect

The table reports absolute modern-minus-early mean potential lambda, using equal weights on fixed land cells in the common 25-49N belt. East is all land east of 95W, Plains 100-95W, West west of 100W. Early is 1902-1915; modern is 2010-2025. Values span seeds 41/42, not posterior uncertainty intervals. Potential lambda is Sa+Sj*Fmax for yearling breeding, before crowding, mating, disease and movement losses.

| Configuration | East | Great Plains | West |
|---|---:|---:|---:|
| Original Veit departure/crowding | +1.267 to +1.271 | +.734 to +.736 | +.819 to +.820 |
| Reduced adult departure only | +1.188 to +1.194 | +.724 to +.725 | +.768 to +.770 |
| New rational departure curve, both ages | +1.581 to +1.598 | +.977 to +1.024 | +.965 to +.969 |
| Rational curve plus reduced adult departure | +1.619 to +1.628 | +1.083 to +1.098 | +.956 to +.958 |
| Squared-density crowding, original departure | +.981 to +.983 | +.685 to +.697 | +.649 to +.672 |
| Squared-density crowding, rational/adult-retention departure | +1.104 to +1.115 | +.807 to +.817 | +.720 to +.724 |
| Squared-density crowding plus larger floor, original departure | +.967 to +.997 | +.712 to +.724 | +.666 to +.679 |
| Squared-density crowding plus larger floor, rational/adult-retention departure | +1.108 to +1.117 | +.805 to +.822 | +.716 to +.725 |

The strongest temporal reduction so far is the crowding change with the original departure law: roughly .28 in the East and .15-.17 in the West, with a smaller Plains reduction. This is an actual reduction in historical change, not merely lower reproductive output or a different modern level. It persists with the floor, though the Plains effect becomes small.

The departure factorial also separates mechanisms: the rational curve itself increases temporal inflation across both seeds. Lower adult departure alone gives a modest reduction without the survival-ratio ceiling. Combining that adult modifier with the rational curve does not rescue the curve's temporal behavior. There is no basis to make the combined rational rule the primary movement model on these results alone.

## Joint demographic and source-sink interpretation

The crowding intervention does not yet select a final configuration. With original departure, its modern population-weighted adult-survival parameter rises to .794-.795, juvenile/adult ratio falls to .320-.323, and maximum reproductive output falls to 5.70-5.79. With the floor these become .803, .310-.312 and 5.93-5.97. These are conditional parameter summaries, not independent survival/reproduction observations.

The high adult parameter must be interpreted with realized survival: western effective annual adult survival is .560-.565 in the unfloored crowding fits, rather than the raw .79 parameter. The raw number alone is not sufficient to reject the fit. Western effective juvenile survival is about .167; joint consistency with marked-bird and reproductive observations remains unresolved. Neither low Fmax nor an acceptable-looking effective adult survival clears the demographic checklist.

In the seed-41 original-departure comparison:

| Modern process diagnostic | Original crowding | Squared-density crowding |
|---|---:|---:|
| Western mean origin-attributed breeder contribution | .988 | .985 |
| Western local replacement excluding movement | .832 | .690 |
| Western successful export per breeder | .494 | .543 |
| Cell immigrants as share of next western breeders | .497 | .548 |
| West-wide population after cutting outside-West immigration, 2025 relative to fitted | .984 | .985 |
| Population after closing individual western cells to movement, 2025 relative to fitted | .808 | .691 |

Thus broadly similar western contribution and regional persistence coexist with greater reliance on redistribution among western cells. The immigrant share is immigration into cells, not immigration into the West from the East. These summaries do not establish cellwise map identification. The underlying growth and movement processes change, not just the split among vital-rate parameters.

The Plains remains sink-dominated under realized contribution and local replacement in the original-departure crowding fits: approximately 83% and 98% of equal-weight primary-band land cells, respectively, in the connectivity audit. The modern era has no coast-to-coast source bridge under those two metrics with either edge or diagonal connectivity. Potential-growth bridges remain if diagonal connectivity is allowed. With the new rational departure, even some crowding variants retain an edge-connected potential-growth bridge. These are separate statements, not interchangeable source definitions.

K/Allee nonviability does not disappear. In the contribution audit's regional support, the unfloored seed-41 original-departure crowding fit has approximately .23% East, 1.48% Plains and .59% West such land; its floored counterpart has .23%, 1.06% and .54%. The population is not required to establish in genuinely poor demographic habitat. Finite-propagule checks remain pending for these new families, so these cached classifications are not independently confirmed establishment outcomes.

Fitted arrival remains about three years early. The original-departure crowding fits retain a frozen-environment release-only front about 26 years early. The curvature intervention has therefore not solved the timing problem. Raw held-out scores are similar to, or slightly worse than, the original control for this arm; paired real-record rescoring is being repaired and rerun, so no predictive superiority is claimed.

## The larger-floor replicates expose near-equivalent demographic solutions

The two standalone larger-floor fits have objective values 135616.058 and 135615.651, only .407 apart under the same scientific configuration. Yet their modern population-weighted adult-survival parameters are .416 versus .575, juvenile/adult ratios .975 versus .454, and maximum outputs 6.32 versus 8.61. Historical lambda change in the full eastern belt is +1.425 versus +1.252. The latter is close to the original +1.267 baseline; the former is substantially larger.

These are two accepted but materially different fitted explanations with almost equal objective values. This supports the concern about optimizer-basin/weak-identification sensitivity. It does not by itself prove two isolated posterior modes. The earlier single-seed conclusion that the larger floor necessarily induces the survival ceiling or worsens temporal inflation is not robust. The replicated curvature fits provide stronger evidence about their intervention than the standalone floor's first seed did.

## Operational progress and repairs

At the initial 06:12 status check, the 3090 had completed **13 verified checkpoint handoffs**, was fitting the first local-arrival-placement arm, and had one failed local attempt. That failure was a ptxas compiler exit 139 in the unfloored reproductive-variation seed-41 MAP. Logs and partial checkpoints were preserved and the task returned to TACC; it is not a failed biological comparison. The other three reproductive-variation MAPs have been handed off but await float64 refinement.

TACC's previous small allocation ended at 06:03. Both regular three-GPU requests remain pending without start estimates. I corrected the resource watcher condition that let pending regular requests block replenishment of the already authorized development waves. Third development wave **3548968** is now pending, estimated at **08:10 CST** at the time checked. One bounded four-hour small replacement **3548970** was requested and is pending on resources without an estimate. The additional small request costs at most six SUs; the third development wave remains within the original three-wave limit. No new full-node regular request was added.

I also found and corrected a follow-up registration omission: step41/42/43 were absent from `range_followups.COMPARISONS`, so new-family rescoring failed and the dependent movement/finite-propagule checks were never scheduled. All **16 affected rescore jobs** were restarted with their registered controls. The first repaired job completed successfully with prediction parity; subsequent jobs and their movement follow-ups are now processing. This omission means the earlier claim that collection covered all these follow-ups was too broad. The biological interpretation above distinguishes available contribution/trajectory evidence from missing distance, finite-propagule and density-feedback evidence.

No new HMC geometry/sampling result is established for these configurations. The full checklist, actual movement distances, finite-propagule establishment, density-feedback oscillation checks, correction dependence and paired predictive transfer remain necessary before recommending a configuration.

Evidence: TEMPORAL_COMPENSATION_AUDIT_20261103.json; FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json; CONTRIBUTION_RESULTS_20261101.md; PLAINS_SOURCE_CONNECTIVITY_20261101.md; live later_state_status.json and local_overnight/status.json. Auto-updated files retain their original filename dates. The snapshot timing above distinguishes observations from newly queued repairs.

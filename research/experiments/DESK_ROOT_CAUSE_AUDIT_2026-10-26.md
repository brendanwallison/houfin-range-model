# Root-cause audit: the apparent common DESK ceiling

This replaces the preceding speculative architecture discussion. The task is to explain existing results before proposing another model. The preferred final boundary remains community-trained Z, frozen before House Finch inference.

## Concrete corrections to the recent analysis

The known-place/withheld-time comparison was NOT missing. E055g tested it and E055h diagnosed its apparent success as modern-end memorization. I should have read those before recommending that comparison as a new avenue. The original end-swap calculation was rerun locally from the archived expensive-training caches during this audit; all saved point differences agree exactly (max absolute difference0). Reproduction outputs: rootcause_audit_20261026/end_swap_{train,heldout}/end_swap.json.

The apparent ceiling is also not one common statistic. Community-coordinate change, change-distance fidelity, species-change correlation, per-species skill relative to persistence, and mean-change capture are different estimands. They use different ranks, species sets, locations, supervision periods and observation operators. Numerical comparisons across these families cannot establish a shared ceiling.

## Mechanisms actually isolated by experiments

### 1. A substantial part of repeatable change is not smooth at the model's spatial grain

E014 used effort-matched route-level ABBA observations, including cross-route covariance of roughly40-year change. Cross-route covariance divided by mean same-route change energy was.558 below10km and.187 at200–300km; a second effort/sampling setting gave.591 nearby. This is an aggregate kernel covariance ratio, not Pearson correlation, a fitted spatial decay rate, or an irreducible-error ceiling. It fixes one temporal span and cannot identify how regional sharing changes with span. At27km, cells averaged about1.1routes. ESK24 preserved45–71% of the shared component in the first analysis, but only~6% of the remainder under that decomposition.

The new local extension CHANGE_COHERENCE_BY_SPAN_2026-10-26.md reproduces E014's original energy and nearby covariance ratio to floating-point precision, then crosses temporal span with distance at fixed observation effort. On fixed route/pair panels, regional sharing increases materially from5 to30years at30–60km, in two anchor periods. This supports increasing multidecadal regional coherence; it does not establish that local differences shrink absolutely, identify ecological versus observer mechanisms, or demonstrate predictive skill. The strict5–50year common panel and the nearest-distance bins are poorly supported. See the extension for support and observer sensitivities.

This is a specific reason why high within-route repeatability can coexist with weak prediction at withheld locations. A repeatable local/route component need not transfer to a neighboring grid cell. It is not proof that the remainder is noise: fine habitat, observer changes and other local processes remain mixed. E028/E028b separately found observer-associated persistence fading across three eras; those analyses do not quantify all missing change as observer bias. E053's route-grain covariate checks did not establish a robust remedy by simply replacing cell-level land cover with finer land-cover inputs.

### 2. Most model variations reused the same information; some apparent source gains were dimensional-access gains

E085's head blocks were linear functions of the same Z. E086c dissected the apparent improvement: joint_full reproduced the sum readout exactly at fixed median amplitudes; reading all64 Z dimensions retained most of the gain. On the six E084 head models, sum minus iso64 was only about−.0007 to+.0013. Giving the heads only the leading24 dimensions removed their gain entirely. Thus this part of the apparent multi-source ceiling is largely multiple parameterizations of the same feature span, not independent attempts with genuinely different information.

E086's private heads were also deterministic functions of the same covariates at prediction time. Extra labels could teach different functions, but they did not provide a route to encode a local historical departure that the covariates cannot distinguish. That fact does not, by itself, prove that adding coordinates will generalize.

### 3. The teacher geometry and species readout can bottleneck before the neural encoder

E086x's exploratory observed-community oracle still recovered only~+.02–.03 of the+.151 mean rise in nine sedentary-common targets, compared with~+.06 from DESK. This isolates a shortfall that cannot all be repaired by a better covariate-to-Z network: even the observed96-species community geometry plus the tested linear readout did not imply the targets' mean change. It does not establish that the true ecological community lacks that information under every representation.

E086y changed the OBSERVED kernel: on the same~480joint cells, r64 sum's sedentary-common change correlation was.535 versus breed.518, and level-readout mean capture.80 versus breed.49. E086y2-b's species-cross-fitted comparison raised r64 sum correlation from.451 for the old reference pool to.547 for all353 eligible species. These are real examples where changing the community information/definition moved performance, rather than all methods reaching an invariant ceiling. They are not numerically comparable to the final296-species confirmation scores.

E066's contribution audit is relevant to the law-of-large-numbers premise: the original96-species log1p-Ruzicka kernel had an effective contribution count~27; the common half nearly reproduced the whole kernel. A201-species resident-expanded pool had effective count~39. Widespread species dominate contributions; hundreds of named species do not supply hundreds of independent, equally weighted signals. These results do not authorize changing the user's entire-community definition or selecting a House-Finch-specific reference set.

### 4. Spatial fidelity does not certify temporal metric fidelity

E086m's saved artifact verifies, at T0 1996/r64, spatial kernel fidelity.910 for DESK but change-distance fidelity.025; the corresponding observed-community reference was.988 spatial and.526 change-distance. RRR's change-distance fidelity was.102 and covariate PCA's.119. Thus the covariate representations fail to rank where communities changed most, even while preserving broad spatial geometry. These are correlations of change SIZE, not vector-direction correlations.

The existing raw/smoothed feature and head tests also showed random-pair fidelity is dominated by geography. A high random-pair Ruzicka score does not establish that the temporal differences relevant to the historical simulation have been learned.

### 5. Conditioning on historical observations was already tested, with an information-scale and season limitation

E086r's own-species CBC-conditioning test initially fit weights on annual deviations; CBC weights were near zero, an anticipated attenuation problem. E086r-b repeated it at the epoch-change scale and compared against a separately calibrated DESK control. Own CBC change still added nothing material for sedentary species in that test: within combo−desk_cal−.003(full),−.006(sparse); level−.003/−.015. Supports were only55–70evaluation cells and28–46sparse fitting cells. It therefore constrains a simple species-wise conditioning strategy; it is not a decisive test of a shared community reconstruction model.

E066c directly measured the winter/breeding similarity bridge: resident/sedentary spatial agreement~.83–.84, temporal agreement~.51–.52 after the stated disattenuation. E079 found lower temporal agreement in urbanized strata. E082's species rescaling did not eliminate the seasonal gap. Thus a winter observation is not simply a noisy summer observation with more data guaranteed to solve the problem.

### 6. Target-era information mattered; early-quality observation routes often failed to transmit it

E086u is a decisive data-availability ablation: with sparse target data, CBC Ruzicka supervision improved sedentary-common change by+.080 when target-era winters were included, versus+.017 when they were excluded. Difference−.063, with all four seeds lower. At full target data, the two gains were+.037/+.034. This rejects a blanket claim that target-era winter observations contain no useful transferable information.

E070 showed why simply extending count-similarity supervision backward was not sound: realistic damage generated spurious dissimilarity through count size; detection-based approaches were more robust but did not preserve the full count-similarity signal. E086v/v-b then tested the full learning route. Correcting the presence observation operator removed the level harm (roughly−.03 to0) but also removed the apparent sparse within gain (roughly+.03 to0). Under the tested weight and representation, early presence supervision did not carry the good-quality count-kernel gain.

This is a concrete unresolved bottleneck: information in noisy historical winter observations → constraints on a community representation → transfer to the intended target. It is not equivalent to a shortage of NN capacity. The old v-b result does not establish that the final clean128 early term is inert; its actual influence requires a matched current-candidate diagnostic. Merely listing early CBC in its training inputs is not evidence it learned an early trajectory.

### 7. Modern-end improvement can masquerade as historical improvement

E055g/h was reproduced from original caches. At known places, epoch1900's change correlation improved+.093[.070,.115], but the historical-end swap gave−.078 while the modern-end swap gave+.114; inserting observed modern coordinates alone gave+.109. At withheld places the complete treatment gain was+.001[−.057,.044]. Coordinate mismatch makes the two swaps imperfect standalone causal effects, as the original report noted; their contrast and the memorizer control identified the modern-end mechanism.

This directly invalidates my earlier suggestion that merely changing to known-place conditioning was an untested answer. Any such test must use independent modern evaluation observations or explicitly score the historical endpoint; otherwise conditioning can improve the reported change score without improving reconstruction.

## Checks performed in this audit

- Reran the original E055h end-swap script for known and withheld locations; point estimates reproduce the saved results exactly.
- Read the original E086m, E086c-summary and E086y numerical artifacts, not just narrative verdicts; recorded their paths and SHA256 in rootcause_audit_20261026/artifact_manifest.json.
- Checked one possible averaging-operator artifact using the same archived targets: mean annual ESK projection versus ESK projection of epoch-mean counts had centered change agreement.951/.952 at r24 and.928/.930 at r64 across the two caches. These estimands differ, but the check does not reveal a catastrophic disagreement explaining the whole shortfall. It is same-data operator agreement, not a predictive ceiling. Saved in averaging_operator_audit.json.
- Read the later corrections/results rather than initial verdicts in E004/E014, E028, E039/E044, E043, E053, E055, E066/E070/E079/E082/E085, and E086's c/e/m/n/q/r/s/u/v/v-b/x/y/y2/ab chain. This is not a claim to have independently reproduced every experiment.

## What remains unidentified

There is no verified partition assigning the total missing ecological change to observation bias, fine-scale ecology, missing environmental drivers, teacher-kernel compression or the learned map. The experiments isolate several mechanisms but do not add into one variance budget: their targets, supports and operators differ. 'All methods hit the same ceiling' is therefore not an established root-cause conclusion.

The viable next analysis is a matched bottleneck decomposition on the current clean seasonal candidate, reusing these tests: observed community information versus learned Z, spatial versus temporal metric fidelity, and historical versus modern endpoint contributions. The winter operator/early-loss influence must be measured before interpreting early input availability as effective supervision. No new architecture or broad metric search is justified by this audit alone.

Constraints on interpretation: all original experiment scopes and caveats remain; the historical observation route differs from the modern tests; resident-season similarity correlations are not universal upper bounds; repeatable route effects are not automatically noise; old rank24/96-species conclusions are not automatically current128/all-species conclusions. Historical evaluation datasets already opened remain exploratory.

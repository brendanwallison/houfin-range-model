# Results and queue review, November 1, 2026

**Mechanism clarification:** the regional Allee-dead counts isolate positive potential growth with K/Allee-induced failure of local replacement, not generic inhospitable habitat. A Plains establishment barrier is expected. SOURCE_SINK_DIAGNOSTIC_RECONCILIATION_20261101.md reconciles the implemented viability/crossing diagnostics with the missing actual demographic and immigration budgets; prevalence alone does not rank biological credibility.

**Assessment corrected after regional review:** absolute Allee plus native departure reduces western sink dependence but worsens Great Plains Allee-dead-cell coverage. It is not established as the preferred whole-configuration solution. See CONFIGURATION_CHECKLIST_CORRECTION_20261101.md for the regional numbers, all fifteen pathology assessments, and the missing later-initialization comparison.

Checked approximately 15:25–15:31 America/Chicago. This review preserves the user-added queue and remote controller settings.

Two TACC A100 40-GB allocations remain active: 3542329 on v330-019 and 3544518 on v330-022. They own step34_rel_comp_s42 and step34_rel_v2_s42 respectively. Both have active GPU processes using about 35.2 GB; their MAP logs reached approximately 306/400 and 382/400 during review. These are initialization stages, before float64 refinement. Allocation 3541710 ended at its 48-hour time limit. Remaining allocation times at 15:25 were approximately 8 hours and 38.6 hours. There are no pending shared-queue scripts behind the two running jobs.

The persistent watcher is updating. Direct GPU telemetry for v330-022 currently fails SSH host-key verification; a read-only srun inside its existing allocation independently confirmed the GPU process memory, and its progressing log confirms work. This telemetry gap is not evidence of an idle node. The controller now has a 2099 staging deadline and 10,000 GPU-hour ceiling, apparently changed since the earlier bounded plan; those settings were left unchanged. Its charged/reserved total was 44.81 GPU-hours. An open budget does not create additional pending experiments.

## New evidence and implications

The refreshed complete stored-run audit contains 135 GRID54 directories, 110 accepted refinements and 86 accepted real-data fits, compared with 120/87/63 in the morning snapshot. Among the 86 real fits, 54 have detailed biological readouts, 42 have front readouts, all 86 have positive-record K/Allee contexts and parameter summaries, and four have fixed-point precision comparisons. This is a refreshed evidence inventory, not certification that every pathology has been resolved. See FULL_PATHOLOGY_AUDIT_20261101_AFTERNOON.md and its JSON for all runs, configurations and missing checks.

Modern survival medians below use fitted-population weights. The Allee column instead uses actual positive modern BBS records over the whole domain; it is not population weighted. Sink share is western fitted population in fundamental sinks. Values separated by a slash represent seeds 41/42.

| Configuration, with yearling breeding and NestWatch/CBC constraints | Adult survival | Resident juvenile/adult survival | West population in sinks | Positive records in Allee-dead cells |
|---|---:|---:|---:|---:|
| Absolute Allee reference (seed 41) | .560 | .479 | 10.4% | 1.01% |
| Absolute + native departure | .555/.577 | .479/.437 | 2.2/1.6% | 1.85/4.58% |
| Absolute + path compounding + native departure | .574/.518 | .437/.544 | 1.5/2.7% | 4.98/1.32% |
| Absolute + DESK v2 + native departure | .386/.469 | .962/.661 | 3.5/2.9% | .23/1.11% |

The simple absolute/native candidate improves the reported survival and western sink summaries, but its Great Plains establishment diagnostic deteriorates substantially; those improvements do not justify a preferred whole-configuration ranking. Compounding does not consistently improve that balance across seeds. Its paired spatial log-score gains against the fixed absolute reference are +.0110 [-.0092,+.0305] and +.0049 [-.0058,+.0196], exploratory 90% block intervals. These are not direct compounding-versus-native-only contrasts, so they do not establish an incremental benefit from compounding.

DESK v2 improves the positive-record establishment diagnostic in the native-departure fits. It also permits a different demographic explanation: seed 41 has Sj/Sa=.962 and 59.2% of fitted modern population in cells with ratio above .95. Its spatial gains against the same absolute reference are +.0062 [-.0008,+.0132] and +.0081 [+.0028,+.0142]. Thus predictive gains and establishment improvement can coexist with renewed survival compensation. This is a reason to retain both diagnostics, not to discard the improved basis.

Distance-cost additions remain weakly supported as an extra scalar. Absolute v2 hazard estimates are .0035/.0033 and spatial gains against the original absolute reference are +.0064/+.0031, close to v2-only +.0062/+.0032. Absolute hazard/native estimates differ strongly between seeds (.0195 versus .0017). Direct matched contrasts would be needed to claim an incremental improvement.

The historical timing concern persists across these new absolute variants: median fitted arrival errors at the saved 10% threshold remain -3 years; frozen release-only checks remain approximately -36 to -38 years. Frozen modern-environment experiments are diagnostic counterfactuals, not reconstructions of historical habitat. Their common rapid spread nevertheless leaves historical habitat compensation and invasion timing unresolved. Effective mover survival, immigration replacement requirements and realized permanent movement distances remain missing decisive checks; resident survival is not a substitute for them.

## Path compounding and incomplete runs

The segmented-feature discrepancy was resolved by restoring the 1e-4 land-share floor used to build stored Z_disp. Across 124 years, maximum absolute discrepancy is .000154 and relative L2 error is 8.86e-6; the registered equivalence thresholds pass. This replaces the former held-input status, and explains why compounding arms could proceed.

step34_abs_comp_s41 has a MAP checkpoint but no accepted refined result or RUN_RECEIPT. Its original allocation timed out; the queue failure log records a later dispatch refused by the existing `attempt already started; automatic retry forbidden` safeguard. This is unfinished computation, not biological rejection. No checkpoint was overwritten or retry forced during this review. The auto-results table's generic running/pending label is inaccurate for this arm. Some other registered but unstaged arms also appear as pending in that table; the actual pending queue is empty.

## Observation work since the previous review

US and Canadian raw eBird extraction and source-calibration baselines have finished, as has CBC preparation/calibration. No integrated demographic refit has yet been performed with those new likelihoods.

The local full-count eBird control originally failed while serializing a NumPy boolean in its first result receipt. Casting the flag to a Python bool fixes the receipt without changing the likelihood. The failed attempt remains preserved; task 1035 reran into a separate directory and completed all 11 arms. The independent NB likelihood/gradient test passed.

This uses the same 25,749 June Canadian checklists, fixed demographic references, train-only effort coefficients and blocked holdouts. All arms pass the registered numerical checks. Concentrations are finite, approximately .027–.050, rather than the problematic positive-count hurdle boundary. Recent temporal-test detection is 4.84%, while population-linked predictions are about 3.10–3.13%. The no-young absolute reference scores better than the relative or absolute/native fixed points in this comparison, but still loses to the matched geographic NB baseline: temporal mean log-score difference -.0121, exploratory 90% block interval [-.0248,-.0010]. Its spatial difference is -.0026 [-.0111,+.0078]. Native departure's corresponding temporal difference is -.0218 [-.0408,-.0072]. These intervals condition on the fitted models and are not demographic uncertainty intervals.

This supplies a numerically usable external observation diagnostic and a measurable temporal mismatch. It does not show that native departure is biologically wrong, that the observation model is fully calibrated, or that raw eBird already identifies vital rates. The immediate interpretation is that BBS-fitted demographic points do not transfer adequately to these recent Canadian checklists under the deliberately simple effort-only source mapping. Young-visibility sensitivities do not remove that mismatch. The same comparison should be extended to US breeding data and the response of the shared likelihood checked before a demographic refit. Winter CBC/eBird still require an appropriate seasonal observation mapping; no sedentary-winter assumption was silently introduced.

Artifacts: `/mnt/e/Datasets/houfin/joint_observations_20261101/full_count_link_CA_serialization_fix/{results.json,SUMMARY.md}`. Source pilot progress remains in JOINT_OBSERVATION_PROGRESS_20261101.md; the new shared-population comparison above supersedes its old next-step paragraph.

Priority remains a parsimonious configuration judged jointly on prediction, observed-location establishment, survival balance, immigration dependence, permanent movement and invasion timing. New basis and movement mechanisms should be compared at that configuration level; none of the current variants resolves the full checklist.

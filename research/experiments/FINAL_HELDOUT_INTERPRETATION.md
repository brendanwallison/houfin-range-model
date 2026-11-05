# Interpretation of the final held-out DESK evaluation

Completed 2026-10-25 at 09:06 CDT: 26 summer scores, eight winter periods, and a fixed-model pseudo-early observation check. Used the three existing TACC allocations and the local RTX 3090. Full results: FINAL_HELDOUT_RESULTS.md; execution/provenance: FINAL_HELDOUT_STATUS.md.

## Assessment

V2 at 64 summer + 64 winter dimensions improves the summer proxy relative to v1, but the evidence does not identify an all-season winner. V1 scores substantially better on early winter lists. Historical interpretation remains limited by observation effects and an indirect training-data path through CBC completeness.

## Summer evidence

The primary comparison predicts 1966-1986 -> 2005-2025 change at 306 held-out cells, across 296 species excluded from direct encoder supervision. Target readouts calibrate on 48,844 training-cell observations from 1996 onward. Metric is noise-corrected pooled correlation, not RMSE skill or a fraction of change explained.

| Representation | Within-cell readout | Pooled readout applied to change |
|---|---|---|
| v1, summer64 | .305 | .217 |
| v2, summer32+winter32 | .327 | .195 |
| v2, summer64+winter64 | .333 | .264 |

Paired v2(64+64)-v1 intervals: within +.028 [.011,.046]; pooled +.047 [.030,.063]. The corresponding improvements remain positive in each separate early window (1966-1975, 1976-1985, 1986-1995): within +.027 to +.030, pooled +.039 to +.042. Window support differs (237/287/365 cells), so this is consistency of the within-window comparisons, not an isolated estimate of reach dependence.

At equal total width, 32+32 loses about .022-.025 on the pooled readout in all four windows. Its within-cell advantage is positive, with a CI crossing zero in 1986-1995. V2's summer-only half is close to v1 in the primary window, suggesting the combined seasonal access matters more than a change in the summer half alone.

The 128-feature advantage depends on available target data. With 3% of target training rows:
- thinning seed 1 (1,518 rows): within - .004 [-.025,.018]; pooled +.021 [.006,.035].
- thinning seed 2 (1,445 rows): within - .041 [-.062,-.017]; pooled +.021 [.004,.039].

Thus the pooled gain survives both sparse repeats, whereas the within-cell gain does not. This matters when choosing downstream priors/readouts; it is not evidence that more features universally help.

The established strong linear/trend rival scores .336 on the primary within-change task. V2(64+64) alone is statistically level with it: -.003 [-.024,.020]. Adding V2 to that rival gives a small gain, about .008 [.000,.015]. Candidate improvement over v1 should not be mistaken for a broad defeat of the fair rivals.

## Winter evidence and observation checks

The original CBC preparation/taxonomy rules leave 253 directly held-out species; 211 meet the fixed calibration-support rule. Modern calibration uses 44,529 lists outside test/buffer cells. Evaluation uses the same 6,243 lists at held-out locations for every candidate. Two circles that moved between train and test grid cells are wholly excluded from calibration.

For 1966-1985, v2(32+32) improves absolute conditional list score over v1 by .029 [.010,.047], but its improvement in *temporal gain over its own modern-feature placebo* is only .003 [-.013,.020]. V2(64+64)'s temporal difference is -.003 [-.019,.015]. There is no convincing general winter temporal advantage from the second half in these results.

V1's early-list performance is materially better: relative temporal gain over v2(64+64) is about .298 in 1902-1919 and .243 in 1920-1939, with intervals excluding zero. The low-counter sensitivity retains the difference. However, the same-circle test has only 3 and 5 circles before 1940, below the predeclared 10-circle gate. These scores estimate recorded composition under changing survey conditions, not latent abundance or habitat truth.

A separate check reused the existing E086v pseudo-early data on the *same* 1966-1995 sites/years, holding features and readouts fixed. Reconstructed readouts reproduced their original scores within 1e-6. Observation damage alone increased apparent temporal gain by .015-.020 across the candidates. It did not reproduce the large real early v1/v2 gap. The gap remains unresolved rather than explained away.

## Holdout boundary and next decision

The completeness reconstruction exactly matches the input used for training. Recognized sealed species constitute about 37% of benchmark members; House Finch is a member in about 80% of circle-years' benchmark sets. These detections affect completeness, which enters the auxiliary training pipeline. Therefore direct target exclusion does not make these models strictly untouched with respect to the confirmation species or focal species.

Retain both candidates provisionally. Preserve all 128 source features for V2: a first-64 positional cut would discard its winter half, and 32+32 has a measurable pooled-readout cost. Before production finalization, regenerate CBC operators/auxiliary inputs with the already prepared training-species-only completeness file, retrain the relevant DESK candidates, and repeat the fixed key comparisons. ESK bases need not be rebuilt solely to remove this completeness path: the identified path is in the CBC auxiliary preparation.

The clean completeness artifact is at the WSL run root, audit/completeness_training_only.npz. This evaluation did not retrain any encoder or change age-model production settings. The downstream age model still requires explicit seasonal-block ingestion/prior support for V2.

## Verification and interpretation limits

Twelve focused tests pass. Candidate pairing checks identical target columns and truth bootstrap components; component-derived correlations agree with the established scorer's direct calculations. All source snapshots and local/remote sealed-read logs are retained. The pooled readout now honors the existing --desk-split flag, matching its documented behavior; older split-readout results are not automatically interchangeable with this correction. ABBA year halves address sampling noise but do not guarantee independence of route/observer effects. These results concern observational proxies and remain conditional on the fitted ensemble and readout specification.

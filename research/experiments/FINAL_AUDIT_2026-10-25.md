# DESK final candidates: audit of 2026-10-25

Assessment: eight completed training runs and complete per-seed cube file sets; no demonstrated winner on downstream temporal prediction. V2 is a viable experimental candidate, not yet a validated improvement. Do not equate completion or summer kernel validation with early-century backcast accuracy.

Evidence read directly on TACC through the existing SSH control socket: processed/encoder/desk_final{,_v2}_s{1..4}/run_summary.json and train_trajectory.jsonl; cube arrays for 1902, 1940, 1966, 2020, 2025 and cube_meta.json; desk_temporal task logs and queue; Slurm accounting from October 24. Research source: local HEAD 36df2bf; v2 task log records 199a1afe8. The attached plan was treated as historical context, not fresh authorization to execute its research loop.

## Training results

All runs: 500 epochs, best checkpoint restored, 532 held-out cells, 579 buffer cells, no temporal holdout, select_on_withheld=false. Values are per seed, not ensemble scores; lower kernel errors are better.

| Candidate | Selected epochs | Summer validation kernel MSE | Raw summer kernel MSE | Learned EMA half-life |
|---|---|---|---|---|
| v1 | 166,124,107,91 | .004828,.005007,.004700,.004857 | .005303,.005488,.005248,.005367 | 10.42-10.68 y |
| v2 | 123,132,111,131 | .004894,.004910,.004685,.004834 | .005432,.005498,.005145,.005367 | 10.46-10.60 y |

Mean summer kernel MSE is about .004848 vs .004831 (v2 about 0.35% lower); raw errors average .005352 vs .005361. These do not establish a practical v2 advantage. The validation score measures the summer block, not the joint summer/winter kernel or early temporal extrapolation. Training losses for the winter head decline, but are not held-out winter predictive validation.

Selected-epoch temporal diagnostic direction cosine: v1 .589-.637, v2 .605-.623. Rotation ratios: v1 .443-.742, v2 .611-.682. At epoch 500 ratios rise to .882-.951 and .932-.976 respectively; summer validation error deteriorates to roughly .0061-.0064. This repeats the known spatial-fit/temporal-movement tradeoff. These are observed-era diagnostics, not pre-1966 accuracy.

## Export and ensemble findings

Each candidate seed has 124 yearly files (1902-2025). Inspected arrays have shape (133,224,64) for v1 and (133,224,128) for v2; 17,209 cells are finite in 2020. V2 metadata reports zero gapfilled cell-years and raw_instantaneous output. All years were counted; numerical contents were sampled at five years.

V2 cube_meta.json incorrectly declares latent_dim=64 for 128-dimensional arrays and describes only a single uncentered Ruzicka kernel, identifying only the summer basis. It does not describe the winter block or the sum kernel. build_final_z_cube.py writes the summer latent_dim into metadata even when include_private doubles the export. The checked-in downstream config expects 64 source dimensions and retains the first 24; adopting v2 requires explicit block-aware ingestion and kernel metadata. The separate evolving age-model worktree may differ; this is a finding about this checkout and exported metadata.

Spatial 2020 per-dimension seed-pair correlation medians:
- v1: .795-.859. One pair fails final_mean_cube.py's default .8 threshold.
- v2 summer: .828-.856; winter: .800-.847 (minimum .79973).

For 1902-to-2025 change, medians are lower: v1 .674-.773; v2 summer .729-.772; v2 winter .664-.681. Several summer dimensions have negative cross-seed change correlations. These do not prove coordinate nonidentifiability (both halves have fixed coordinate targets), but show appreciable uncertainty and the potential for averaging to suppress disagreement in change. A whole-vector median gate can conceal a weak half or individual directions. Validate the resulting mean kernel and temporal predictions, not just alignment. No mean-cube directory matching *final*mean* was found; this glob does not exclude an ensemble stored under another name.

## Prior evidence and remaining inference

E086ab-b supports all-species r64 at the late state versus the old 96: sparse-data within change +.052, CI [-.001,.106]. At the selected state, however, level change loses .058-.082. Thus late-state breadth evidence cannot be transferred wholesale to selected-state candidates. The prior 40+24 sum improves level change about .03 and costs within change .024; it is not a direct evaluation of v2's coordinate-supervised 64+64 model.

E086v-b shows the early presence term through a winter operator was approximately inert in predecessor experiments. The presence NLL itself is not evidence of useful early habitat change. V2 changes that route to a separate coordinate-supervised winter half, so the old result neither validates nor definitively refutes its early-year benefit.

The all-species versus all-sedentary E086ac evaluation actually completed locally at 2026-10-25 01:17 CDT, exit 0, 27,100.6 s, despite FINAL.md calling it pending. Its pooled comparison has not been rederived in this audit; no winner is inferred from the last log section alone.

## Job state

At inspection no Slurm DESK jobs and no pending/running DESK task files remained. Earlier desk_worker timeouts are not evidence these final candidates failed: their summaries and cube exports exist. Workers 3532787 and 3533084 subsequently exited after idle 15 minutes. Age workers 3530553/4/5 remained allocated; logs showed E081 tasks followed by E082. These are not evidence of v1/v2 performance.

## Adoption recommendation

Keep both candidate families for direct evaluation. Resolve metadata and block ingestion before using v2 downstream; evaluate the four-seed mean as the actual artifact to be shipped. Compare v1 r64, v2 32+32, and v2 64+64 to separate source benefit from dimensional capacity. Score held-out cells/species with raw features, both level and within change, full and sparse data; add held-out winter validation and pseudo-early degraded-list tests. A temporal-holdout replica is required for a claim about extrapolation: these full-history final runs cannot establish it. Recover E086ac's completed pooled comparison before declaring the reference-community question closed. Preserve the sealed set for the final chosen candidate.

No training, queue, production config, or existing research records were changed by this audit.

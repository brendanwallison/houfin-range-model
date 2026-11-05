# Final summer/winter evaluation readiness — 2026-10-25

Scope: reuse established estimands and scoring implementations for final v1/v2, with the minimum adapters needed for explicit feature blocks and sealed species. This document records preparation, not scored results or a launch of the sealed confirmation.

## Existing compute, inspected about 07:40 CDT

- Running small-queue workers 3530553 (v330-009), 3530554 (v330-004), 3530555 (v330-016), approximately 7h40m, 7h55m, 11h remaining. Live age queue is queue2; pending/running directories empty at inspection. The logs show completed E081/E082 jobs.
- Verified directly on v330-009: worker shell alive, sleeping in its poll loop; A100 40 GiB, 0 MiB used, 0% GPU utilization.
- Existing queue2/done/0330a_desk_drain.sh already drains the DESK queue inside an age worker using the DESK worktree and atomic task claiming. This establishes the existing allocation handoff; new Slurm submissions are not required while these workers remain available.
- Local RTX 3090: 24 GiB total, approximately 20.4 GiB free; pueue gpu/cpu queues have no active tasks. Prefer cached-array/BLR evaluation locally; use the already allocated A100 workers for larger encoding. Do not assume a 1902-window 128-dimension encoder fits the local card without measuring peak memory.

## Existing analysis to reuse

- fair_rivals.py, score_both.sh: ABBA epoch-mean log1p species targets, cross-half estimate of signal energy, paired 6x6-cell block bootstrap, level and within-cell readouts, full/sparse target regimes, spatial/local change checks, strong/placebo rivals; BLRs use lib/blr.py with separate feature-block amplitudes.
- species_strata.py and saved species_components: identical species sets and paired draws for comparisons; avoid averaging unresolvable per-species correlations. Preserve the current signal/denominator rules and compare mean-change bias in addition to centered correlation.
- e069_presence_score.py and e086p_early_lists.py: conditional within-list composition likelihood, offsets fitted outside test circles, backcast versus modern and regional placebos, paired circle bootstrap. Shared list effort is controlled conditionally; species-specific detectability, completeness and observer shifts still need checks. Presence composition is not absolute abundance.
- e065_eval.py / e070_eval.py and pseudo-early builders: effort sensitivity, independent circle halves, early-quality degradation recovery and support gates. These are essential before interpreting early winter changes.
- e086x_handoff.py: boundary discontinuities and species-mean change bias that centered correlation misses.

## Required adapters / interpretation checks

1. Explicitly select v1 summer64; v2 summer32+winter32, summer64+winter64; season-only ablations. Read raw final cubes as primary, EMA as a named sensitivity. Several existing scripts hard-code 96 reference species, EMA, or pre-1940 lists unseen during training; those statements do not apply unchanged to final candidates.
2. Final encoders trained through 2025: sealed-species/spatial tests by epoch are possible, but they are not encoder temporal-holdout tests. Readout calibration restricted to modern/training cells and years tests backward transfer of the readout; temporal-holdout encoder replicas test the additional extrapolation claim.
3. Use established summer early/modern windows and early-winter era bins; freeze support thresholds, species sets, calibration folds, baselines, feature access and bootstrap draws before reading sealed outcomes. Evaluate summer and winter on their own appropriate observation models and report common species/cell support separately.
4. Prepared CBC tables exclude the sealed species. Build a separate confirmation extraction from raw CBC, reuse existing taxonomy/split/effort handling, and log access. Do not weaken existing preparation guards silently.
5. Additional holdout caveat found in source audit: cbc_completeness.py computes list length and benchmark sets/hits from all raw species, not only the non-sealed training set. final_cbc_aux.py / final_v2_aux.py use the resulting completeness covariate. Thus zero overlap in supervised species lists establishes direct exclusion, not total independence. Audit completeness_v3 provenance and whether sealed species influence that training covariate before claiming a strict sealed test. Merely leave-one-species-out adjusting the scoring covariate cannot undo any influence in an already trained encoder. Spatially held-out circles and a clean-completeness sensitivity/replica can distinguish this issue.

Ready to begin adapters and preflight on the current resources; strict sealed scoring waits for the above holdout boundary and fixed evaluation manifest. No new Slurm jobs or sealed scoring runs were launched during this readiness check.

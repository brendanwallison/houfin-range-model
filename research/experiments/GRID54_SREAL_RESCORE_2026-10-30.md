# GRID54 ladder rescored on S-real (2026-10-30)

Why: B0 (research/experiments/SDM_BASELINE_B0_2026-10-30.md) found (i) T marginal ties cell persistence, so it cannot
rank dynamics; (ii) 36% of the ladder's spatial held-out rows are 1902-39 pseudo-zeros, which every model scores near 0
nats, so S-all (the ladder's S) hides the scale; S-real (real BBS rows in the pinned 216 km blocks, n 14,524) is the
spatial score that discriminates (age model beats kernel interpolation by +.13, mostly in the east).

Tool (any grid54 run): research/exp/rescore_run.sh STEP [REF] -- exports the run's design + predictive with the run's
OWN frozen code and overlay (runs/<STEP>/code, the single grid54_<...>.json overlay there) from the refined point
(model_results/refined_float64__age_map_float32_grid54_<STEP>_quick90), then research/exp/sdm_rescore.py scores it
(plug-in / marginal / conditional exactly as map_diagnostics; S-all, S-real; region; region x era; 216 km
block-bootstrap 90% intervals; paired differences vs REF when the observation design hash matches). Parity: every
run's T/S plug-in, marginal and conditional are checked against its own stored viz_ref/metrics.json holdout block
(< .001 nats/obs). Table: `sdm_rescore.py --table STEP ...`. Outputs /mnt/e/Datasets/houfin/grid54/baselines/rescore/
<STEP>/rescore.json, table.md. CPU only, 2 threads, cores 0-3, nice 19.

## Table (refined points; mean NB log density per observation, higher is better; paired = arm minus step 3)
| step | status | parity | T plug | T marg | T cond | S-all marg | S-real marg [90% CI] | west | plains | midwest | east | paired S-real [90% CI] | paired T marg [90% CI] |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| step3_obsroute (reference) | valid | ok | -1.844 | -1.844 | -1.731 | -1.219 | -1.918 [-2.341, -1.387] | -2.384 | -0.695 | -1.494 | -2.306 | -- | -- |
| step9_fdi_kernel (fdinterp input) | valid | ok | -1.853 | -1.849 | -1.733 | -1.234 | -1.941 [-2.384, -1.393] | -2.366 | -0.691 | -1.548 | -2.356 | -0.023 [-0.046, +0.009] | -0.005 [-0.015, +0.004] |
| step8a_kernel | valid | ok | -1.841 | -1.841 | -1.729 | -1.215 | -1.911 [-2.334, -1.383] | -2.375 | -0.681 | -1.485 | -2.305 | +0.007 [-0.002, +0.016] | +0.003 [+0.000, +0.006] |
| step10_pn05 | valid | ok | -1.911 | -1.875 | -1.748 | -1.252 | -1.969 [-2.424, -1.422] | -2.495 | -0.706 | -1.514 | -2.346 | -0.051 [-0.098, -0.007] | -0.032 [-0.055, -0.014] |
| step10_pn10 | valid | ok | -1.862 | -1.841 | -1.733 | -1.219 | -1.917 [-2.340, -1.387] | -2.392 | -0.695 | -1.480 | -2.305 | +0.001 [-0.009, +0.010] | +0.003 [-0.004, +0.010] |
| step10_pn20 | valid | ok | -1.885 | -1.846 | -1.750 | -1.214 | -1.910 [-2.330, -1.382] | -2.391 | -0.671 | -1.469 | -2.300 | +0.009 [-0.003, +0.019] | -0.002 [-0.012, +0.008] |
| step18_warp64 | UNCONVERGED (UNCONVERGED_READOUT; no REFINEMENT_DONE) | ok | -1.884 | -1.863 | -1.735 | -1.261 | -1.983 [-2.431, -1.435] | -2.501 | -0.718 | -1.527 | -2.365 | -0.065 [-0.109, -0.021] | -0.019 [-0.038, -0.004] |
| step11_reset3 | valid | ok | -1.853 | -1.844 | -1.730 | -1.232 | -1.938 [-2.368, -1.404] | -2.393 | -0.687 | -1.521 | -2.341 | -0.019 [-0.033, -0.002] | +0.000 [-0.007, +0.007] |
| step7_allee020 | valid | ok | -1.915 | -1.912 | -1.776 | -1.345 | -2.114 [-2.653, -1.489] | -2.604 | -0.686 | -1.665 | -2.585 | -0.196 [-0.311, -0.072] | -0.068 [-0.106, -0.038] |
| step8b2_brake | valid | ok | -1.873 | -1.874 | -1.748 | -1.265 | -1.990 [-2.442, -1.448] | -2.527 | -0.714 | -1.570 | -2.351 | -0.072 [-0.130, -0.020] | -0.030 [-0.048, -0.014] |

Paired S-real by region (arm minus step 3, 90% block-bootstrap) and paired T conditional:
| step | west | plains | midwest | east | T conditional |
|---|---|---|---|---|---|
| step9_fdi_kernel | +0.017 [-0.051, +0.085] | +0.004 [-0.012, +0.018] | -0.053 [-0.061, -0.044] | -0.049 [-0.068, -0.026] | -0.002 [-0.011, +0.006] |
| step8a_kernel | +0.008 [-0.022, +0.031] | +0.013 [+0.007, +0.019] | +0.010 [-0.002, +0.025] | +0.002 [-0.012, +0.013] | +0.002 [-0.000, +0.005] |
| step10_pn05 | -0.112 [-0.254, +0.020] | -0.012 [-0.024, +0.003] | -0.019 [-0.023, -0.013] | -0.040 [-0.084, +0.014] | -0.017 [-0.032, -0.006] |
| step10_pn10 | -0.009 [-0.033, +0.013] | -0.000 [-0.015, +0.016] | +0.015 [-0.001, +0.036] | +0.002 [-0.005, +0.008] | -0.002 [-0.008, +0.005] |
| step10_pn20 | -0.008 [-0.038, +0.016] | +0.023 [-0.000, +0.049] | +0.026 [-0.001, +0.058] | +0.006 [-0.009, +0.011] | -0.019 [-0.031, -0.008] |
| step18_warp64 (unconverged) | -0.118 [-0.256, +0.010] | -0.024 [-0.031, -0.014] | -0.033 [-0.049, -0.018] | -0.059 [-0.101, +0.004] | -0.004 [-0.015, +0.006] |
| step11_reset3 | -0.009 [-0.040, +0.019] | +0.007 [-0.020, +0.033] | -0.026 [-0.042, -0.015] | -0.035 [-0.053, +0.001] | +0.001 [-0.005, +0.007] |
| step7_allee020 | -0.221 [-0.394, -0.043] | +0.009 [+0.001, +0.017] | -0.171 [-0.256, -0.075] | -0.279 [-0.518, -0.014] | -0.044 [-0.078, -0.020] |
| step8b2_brake | -0.144 [-0.301, +0.018] | -0.020 [-0.032, -0.006] | -0.075 [-0.160, -0.010] | -0.045 [-0.085, -0.001] | -0.017 [-0.031, -0.005] |
- fdinterp habitat (step 9) costs .05 in the midwest and east (intervals exclude 0) while being neutral in the west and
  plains: the fdinterp Z is worse where the invasion happened (established for this pair).
- pn20 is a small plains/midwest gain on S-real (+.02-.03, intervals touching 0) but costs .019 on T conditional.

Notes:
- All ten pass parity with their own stored scores. step9 (fdinterp Z) has the same observation design hash as
  step 3 (same rows, masks, observer-routes), so its paired comparison is valid; only the habitat input differs.
- The unpaired S-real intervals ([-2.34, -1.39] for step 3) are wide because blocks differ greatly in level; arm
  comparisons must use the PAIRED intervals (right two columns), which are 5-20x narrower.
- step18_warp64 is an UNCONVERGED point (its own UNCONVERGED_READOUT marker); its scores are a readout, not a result.

## Reading (established for these fits unless marked)
- Clearly worse than step 3 on S-real (paired interval excludes 0): step7_allee020 (-.196; the largest loss, all
  regions except plains), step8b2_brake (-.072), step18_warp64 (-.065, unconverged), step10_pn05 (-.051), step11_reset3
  (-.019, small). The same arms are also worse on T marginal except reset3 (0).
- Indistinguishable from step 3 on S-real (interval spans 0, |diff| <= .01): step8a_kernel (+.007), step10_pn10
  (+.001), step10_pn20 (+.009). step9_fdi_kernel -.023 (interval just spans 0; midwest/east lower).
- So on held-out counts, none of the screened forms beats step 3 by more than .01; the strong Allee (n50/K .20) and
  the brake cost real spatial predictive skill. Mechanism (LIKELY, untested): forms that slow or reshape the eastern
  front also lower expected counts in parts of the established east/midwest that the blocks sample; process noise at
  .05 is worse than at .10/.20 -- non-monotone, so not simply "noise hurts" (UNKNOWN why; a single fit per arm).
- Against B0: every arm's T marginal (-1.84 to -1.91) is at or below cell persistence (-1.836); every valid arm's
  S-real except step7 beats kernel interpolation (-2.050).
- Implication for ladder rules (reported, not decided): S-real paired differences separate arms where T marginal
  differences are small; the screening rule "held-out marginal within .01 of the no-term twin" should be applied to
  paired S-real (and T conditional), with S-all retired or reported alongside.

## Batch 2 (2026-10-30, later; same tool, paired vs step3_obsroute; all REFINEMENT_DONE, all parity ok)
| step | T plug | T marg | T cond | S-all | S-real | west | plains | midwest | east | paired S-real [90% CI] | paired T marg [90% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| step3_seed42 (seed) | -1.843 | -1.845 | -1.732 | -1.219 | -1.918 | -2.390 | -0.693 | -1.491 | -2.304 | +0.000 [-0.005, +0.004] | -0.001 [-0.002, +0.001] |
| step3_seed43 (seed) | -1.844 | -1.845 | -1.732 | -1.220 | -1.918 | -2.383 | -0.692 | -1.487 | -2.311 | -0.000 [-0.004, +0.004] | -0.001 [-0.003, +0.001] |
| step12_lam10 | -1.901 | -1.871 | -1.744 | -1.250 | -1.966 | -2.504 | -0.705 | -1.515 | -2.333 | -0.048 [-0.097, -0.006] | -0.027 [-0.049, -0.011] |
| step16_sthab | -1.840 | -1.838 | -1.727 | -1.220 | -1.919 | -2.397 | -0.694 | -1.479 | -2.306 | -0.001 [-0.010, +0.007] | +0.005 [+0.002, +0.009] |
| step11_reset1966 | -1.845 | -1.844 | -1.731 | -1.218 | -1.916 | -2.379 | -0.691 | -1.493 | -2.307 | +0.002 [-0.001, +0.005] | +0.000 [-0.001, +0.002] |
| step17_r64_free | -1.845 | -1.843 | -1.730 | -1.218 | -1.916 | -2.378 | -0.693 | -1.494 | -2.306 | +0.002 [-0.000, +0.005] | +0.001 [-0.001, +0.002] |
| step4_amp8 | -1.880 | -1.863 | -1.743 | -1.251 | -1.968 | -2.391 | -0.693 | -1.556 | -2.401 | -0.050 [-0.087, -0.007] | -0.019 [-0.031, -0.009] |
| step6_fdi_step3 | -1.867 | -1.859 | -1.740 | -1.246 | -1.960 | -2.415 | -0.689 | -1.538 | -2.376 | -0.042 [-0.067, -0.012] | -0.015 [-0.025, -0.005] |

Paired by region (arm minus step 3) and paired T conditional / S-all:
| step | west | plains | midwest | east | T conditional | S-all |
|---|---|---|---|---|---|---|
| step3_seed42 | -0.006 [-0.018, +0.004] | +0.002 [-0.003, +0.007] | +0.003 [+0.001, +0.007] | +0.003 [-0.005, +0.011] | -0.001 [-0.002, +0.001] | +0.000 [-0.003, +0.003] |
| step3_seed43 | +0.000 [-0.003, +0.003] | +0.002 [-0.002, +0.007] | +0.007 [+0.000, +0.016] | -0.005 [-0.012, +0.001] | -0.001 [-0.002, +0.000] | -0.000 [-0.002, +0.002] |
| step12_lam10 | -0.121 [-0.272, +0.012] | -0.011 [-0.026, +0.002] | -0.021 [-0.032, -0.012] | -0.027 [-0.053, +0.011] | -0.013 [-0.025, -0.003] | -0.031 [-0.062, -0.003] |
| step16_sthab | -0.014 [-0.038, +0.010] | +0.000 [-0.006, +0.007] | +0.015 [-0.000, +0.039] | +0.000 [-0.002, +0.003] | +0.004 [+0.001, +0.007] | -0.001 [-0.006, +0.004] |
| step11_reset1966 | +0.004 [-0.002, +0.010] | +0.003 [-0.001, +0.009] | +0.001 [-0.000, +0.002] | -0.001 [-0.004, +0.003] | +0.000 [-0.000, +0.001] | +0.001 [-0.000, +0.003] |
| step17_r64_free | +0.005 [-0.002, +0.012] | +0.002 [-0.001, +0.005] | +0.001 [-0.001, +0.002] | -0.000 [-0.002, +0.002] | +0.001 [-0.000, +0.002] | +0.001 [-0.000, +0.003] |
| step4_amp8 | -0.007 [-0.049, +0.038] | +0.002 [-0.011, +0.011] | -0.062 [-0.135, -0.008] | -0.095 [-0.159, -0.007] | -0.012 [-0.022, -0.003] | -0.032 [-0.056, -0.005] |
| step6_fdi_step3 | -0.031 [-0.080, +0.017] | +0.006 [-0.010, +0.019] | -0.043 [-0.046, -0.036] | -0.069 [-0.116, -0.015] | -0.009 [-0.018, -0.001] | -0.027 [-0.044, -0.008] |

### NOISE FLOOR (two MAP-init seeds of step 3, same data and code; established for step 3)
- Paired S-real: +.000 [-.005, +.004] (seed 42) and -.000 [-.004, +.004] (seed 43). Paired T marginal -.001 both.
- By region the seed spread reaches ~.006-.007 (west seed 42 -.006; midwest seed 43 +.007; east seed 43 -.005).
- Reading rule implied: an arm's overall paired S-real difference smaller than ~.005, or a regional one smaller than
  ~.01, is within seed-to-seed refit noise, whatever its bootstrap interval says (the block bootstrap measures
  sampling over blocks, not refit variability). One seed pair only: the floor itself is rough.

### Reading (batch 2)
- Within the noise floor (no difference from step 3): step11_reset1966 (+.002), step17_r64_free (+.002), step16_sthab
  (-.001; +.005 on T marginal, at the floor).
- Clearly worse (beyond the floor, interval excludes 0): step4_amp8 (-.050; midwest -.06, east -.10),
  step12_lam10 (-.048; west -.12 with a wide interval, midwest -.02), step6_fdi_step3 (-.042; midwest -.04, east -.07).
- fdinterp again costs the midwest and east (step6 here, step9 in batch 1): two independent arms on fdinterp Z, same
  regional signature. ESTABLISHED for these fits: the fdinterp habitat input predicts held-out eastern/midwest routes
  worse than the step-3 input.
- Across both batches, no arm improves on step 3 beyond the seed floor on S-real.

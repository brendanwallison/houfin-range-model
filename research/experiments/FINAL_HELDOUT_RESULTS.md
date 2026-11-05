# Final DESK held-out evaluation results

Updated 2026-10-25T09:06:05-05:00. Summer tasks: 26/26; winter complete: True; observation-damage diagnostic complete: True.

PROVISIONAL: direct species targets were excluded, but sealed species and House Finch enter the training CBC completeness benchmark. These are not strictly untouched-species results. The encoders used the full training timeline: this is unseen-species/location evaluation and readout backcasting, not encoder temporal holdout.

## Summer

Noise-corrected correlation of epoch change across held-out cells, pooled over 296 species. Intervals use the existing paired 6x6-block bootstrap. Modern window 2005-2025; readout calibration 1996+. Within fits cell deviations; pooled fits levels and is then applied to temporal change. v2 r64=32+32; r128=64+64.

### desk_within

| Early window / training fraction / seed | v1 r64 | v2 r64 | v2 r128 | v2 r64 minus v1 | v2 r128 minus v1 |
|---|---|---|---|---|---|
| 1966_1975/f1/s1 | +0.291 [+0.257, +0.321] | +0.316 [+0.271, +0.352] | +0.318 [+0.279, +0.351] | +0.025 [+0.006, +0.044] | +0.027 [+0.011, +0.045] |
| 1966_1986/f0.03/s1 | +0.176 [+0.143, +0.206] | +0.199 [+0.164, +0.234] | +0.173 [+0.145, +0.200] | +0.023 [-0.001, +0.046] | -0.004 [-0.025, +0.018] |
| 1966_1986/f0.03/s2 | +0.153 [+0.114, +0.193] | +0.155 [+0.114, +0.196] | +0.112 [+0.077, +0.150] | +0.002 [-0.020, +0.026] | -0.041 [-0.062, -0.017] |
| 1966_1986/f1/s1 | +0.305 [+0.271, +0.335] | +0.327 [+0.287, +0.360] | +0.333 [+0.295, +0.364] | +0.022 [+0.003, +0.039] | +0.028 [+0.011, +0.046] |
| 1976_1985/f1/s1 | +0.298 [+0.261, +0.331] | +0.320 [+0.279, +0.354] | +0.326 [+0.287, +0.361] | +0.021 [+0.004, +0.039] | +0.027 [+0.010, +0.046] |
| 1986_1995/f1/s1 | +0.300 [+0.257, +0.341] | +0.308 [+0.270, +0.345] | +0.330 [+0.293, +0.366] | +0.008 [-0.003, +0.021] | +0.030 [+0.017, +0.044] |

### desk_pooled

| Early window / training fraction / seed | v1 r64 | v2 r64 | v2 r128 | v2 r64 minus v1 | v2 r128 minus v1 |
|---|---|---|---|---|---|
| 1966_1975/f1/s1 | +0.237 [+0.200, +0.268] | +0.214 [+0.177, +0.245] | +0.278 [+0.241, +0.310] | -0.022 [-0.044, -0.004] | +0.042 [+0.022, +0.059] |
| 1966_1986/f0.03/s1 | +0.205 [+0.175, +0.232] | +0.193 [+0.160, +0.222] | +0.226 [+0.194, +0.254] | -0.012 [-0.028, +0.002] | +0.021 [+0.006, +0.035] |
| 1966_1986/f0.03/s2 | +0.189 [+0.161, +0.217] | +0.187 [+0.156, +0.215] | +0.209 [+0.178, +0.239] | -0.002 [-0.015, +0.011] | +0.021 [+0.004, +0.039] |
| 1966_1986/f1/s1 | +0.217 [+0.187, +0.244] | +0.195 [+0.166, +0.222] | +0.264 [+0.233, +0.292] | -0.022 [-0.040, -0.005] | +0.047 [+0.030, +0.063] |
| 1976_1985/f1/s1 | +0.195 [+0.167, +0.219] | +0.170 [+0.143, +0.195] | +0.237 [+0.205, +0.264] | -0.025 [-0.044, -0.007] | +0.042 [+0.024, +0.057] |
| 1986_1995/f1/s1 | +0.162 [+0.134, +0.189] | +0.139 [+0.110, +0.164] | +0.201 [+0.171, +0.231] | -0.023 [-0.043, -0.003] | +0.039 [+0.025, +0.052] |

### Established rivals, primary summer window

A gain over v1 does not by itself establish a gain over the existing fair linear/trend rivals.

| Candidate | Within readout minus strong rival | Additional gain when added to strong rival |
|---|---|---|
| v1_r64 | -0.032 [-0.059, -0.003] | +0.006 [+0.001, +0.010] |
| v2_r128 | -0.003 [-0.024, +0.020] | +0.008 [+0.000, +0.015] |
| v2_r64 | -0.010 [-0.031, +0.013] | +0.005 [-0.001, +0.010] |
| v2_s64 | -0.029 [-0.053, -0.003] | +0.004 [-0.001, +0.009] |
| v2_w64 | -0.023 [-0.045, +0.000] | +0.004 [-0.001, +0.009] |

Strong rival correlation: 0.336; position-time placebo: 0.310.

## Winter

211 eligible species; 44,529 modern calibration lists, 6,243 held-out lists. Two relocated test circles excluded from calibration. Uses the existing conditional list likelihood, winter-specific ridge and modern training-circle offsets. Scores measure relative composition, not absolute abundance. Differences below compare each model's gain over its fixed modern-feature placebo; intervals bootstrap circles.

| Era | Lists / circles | v1 backcast minus modern | v2 r64 gain minus v1 | v2 r128 gain minus v1 |
|---|---|---|---|---|
| 1902_1919 | 74 / 35 | +0.562 [+0.353, +0.765] | -0.383 [-0.514, -0.226] | -0.298 [-0.441, -0.138] |
| 1920_1939 | 168 / 52 | +0.419 [+0.285, +0.546] | -0.331 [-0.439, -0.223] | -0.243 [-0.311, -0.175] |
| 1940_1952 | 204 / 50 | +0.163 [+0.073, +0.254] | -0.181 [-0.247, -0.111] | -0.126 [-0.179, -0.073] |
| 1953_1965 | 382 / 68 | +0.039 [-0.007, +0.085] | -0.040 [-0.067, -0.012] | -0.039 [-0.070, -0.006] |
| 1966_1985 | 1257 / 109 | +0.025 [-0.003, +0.048] | +0.003 [-0.013, +0.020] | -0.003 [-0.019, +0.015] |
| 1986_1995 | 876 / 113 | -0.006 [-0.016, +0.005] | -0.004 [-0.011, +0.003] | -0.002 [-0.010, +0.007] |
| 1996_2004 | 910 / 113 | -0.015 [-0.022, -0.008] | +0.005 [+0.000, +0.010] | +0.009 [+0.004, +0.015] |
| 2005_2024 | 2372 / 146 | -0.005 [-0.006, -0.004] | -0.001 [-0.002, -0.000] | -0.002 [-0.003, -0.001] |

Same-circle sensitivity has only 3 and 5 circles before 1940, below the scoring gate. Early advantages require the observation-damage and effort checks before a habitat interpretation.

## Holdout audit and validation

The original completeness arrays were reproduced exactly. Recognized sealed taxa make up about 37% of benchmark members; House Finch is a member in about 80% of circle-years. A clean-completeness training sensitivity is needed before production adoption. ABBA year halves address sampling noise but do not guarantee independent route/observer effects. Twelve focused tests pass. Component correlations are independently checked against the established scorer's direct results; paired comparisons require identical columns and truth bootstrap components.

## Observation-damage diagnostic

Reconstructed fixed readouts reproduce original scores within 1e-6. Reuses E086v's 1966-1995 damaged observations, with original taxonomy exclusions frozen. Features and readouts are unchanged between original and degraded observations.

Matched lists/circles: 1740 / 136.

- v1_r64/original/backcast_minus_modern: +0.008 [-0.009, +0.024]
- v1_r64/original/gain_minus_v1: +0.000 [+0.000, +0.000]
- v1_r64/damaged/backcast_minus_modern: +0.028 [+0.007, +0.046]
- v1_r64/damaged/gain_minus_v1: +0.000 [+0.000, +0.000]
- v1_r64/damage_shift_in_gain: +0.020 [+0.010, +0.029]
- v2_r64/original/backcast_minus_modern: +0.006 [-0.012, +0.024]
- v2_r64/original/gain_minus_v1: -0.002 [-0.012, +0.010]
- v2_r64/damaged/backcast_minus_modern: +0.020 [+0.001, +0.041]
- v2_r64/damaged/gain_minus_v1: -0.007 [-0.019, +0.006]
- v2_r64/damage_shift_in_gain: +0.015 [+0.005, +0.024]
- v2_r128/original/backcast_minus_modern: +0.004 [-0.013, +0.022]
- v2_r128/original/gain_minus_v1: -0.004 [-0.016, +0.010]
- v2_r128/damaged/backcast_minus_modern: +0.022 [+0.001, +0.045]
- v2_r128/damaged/gain_minus_v1: -0.006 [-0.018, +0.009]
- v2_r128/damage_shift_in_gain: +0.018 [+0.006, +0.029]

Observation degradation increases apparent backcast gain with habitat features held fixed. This scenario does not reproduce the large real early-era v1/v2 separation; that discrepancy remains unresolved.

Full results, source snapshots/manifests and local/remote read ledgers: `/home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025`. No training checkpoints or age-model production configuration were changed.

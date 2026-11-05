# Matched clean-completeness evaluation

Updated 2026-10-25T17:27:58-05:00.

Matched seeds1/2, fixed scoring and reused heldouts. Completeness intervention is not fresh confirmation; encoder training spans the full timeline.

Positive differences favor the clean encoder. Both ensembles use seeds1/2; this does not compare a two-seed model to a four-seed model.

## Summer

1966–1986 to2005–2025; modern target calibration1996+. Noise-corrected change correlation with the existing paired6×6 block bootstrap.

| Calibration / thinning seed | Representation | Within clean−original [95%] | Pooled clean−original [95%] |
|---|---|---|---|
|f1/s1|v1_r64|-0.001 [-0.006,+0.003]|-0.003 [-0.008,+0.003]|
|f1/s1|v2_r64|-0.004 [-0.007,-0.001]|+0.003 [-0.000,+0.006]|
|f1/s1|v2_r128|+0.001 [-0.002,+0.004]|+0.000 [-0.004,+0.004]|
|f0.03/s1|v1_r64|+0.005 [-0.003,+0.010]|-0.004 [-0.010,+0.001]|
|f0.03/s1|v2_r64|-0.010 [-0.015,-0.005]|+0.001 [-0.002,+0.003]|
|f0.03/s1|v2_r128|-0.004 [-0.010,+0.002]|-0.001 [-0.004,+0.002]|
|f0.03/s2|v1_r64|+0.002 [-0.006,+0.010]|-0.002 [-0.007,+0.004]|
|f0.03/s2|v2_r64|+0.001 [-0.003,+0.006]|+0.003 [+0.000,+0.007]|
|f0.03/s2|v2_r128|+0.003 [-0.002,+0.008]|+0.003 [-0.000,+0.006]|

## Winter

Conditional list-composition score with the existing circle bootstrap. Temporal gain subtracts each fitted model’s own fixed modern-feature placebo; it is not abundance truth.

| Era | Lists / circles | v1 clean−original temporal gain | v2 32+32 | v2 64+64 |
|---|---|---|---|---|
|1902_1919|74/35|+0.015 [-0.012,+0.043]|+0.001 [-0.015,+0.017]|-0.021 [-0.037,-0.006]|
|1920_1939|168/52|-0.018 [-0.046,+0.010]|+0.016 [+0.001,+0.028]|-0.009 [-0.020,+0.001]|
|1940_1952|204/50|-0.014 [-0.029,+0.003]|+0.021 [+0.010,+0.032]|-0.008 [-0.017,+0.001]|
|1953_1965|382/68|-0.000 [-0.008,+0.008]|+0.013 [+0.005,+0.020]|+0.002 [-0.004,+0.007]|
|1966_1985|1257/109|+0.000 [-0.004,+0.004]|+0.007 [+0.005,+0.009]|+0.008 [+0.005,+0.010]|
|1986_1995|876/113|+0.001 [-0.002,+0.003]|+0.004 [+0.003,+0.005]|+0.005 [+0.004,+0.007]|
|1996_2004|910/113|+0.001 [-0.001,+0.002]|+0.002 [+0.001,+0.003]|+0.003 [+0.002,+0.004]|
|2005_2024|2372/146|-0.000 [-0.000,+0.000]|-0.000 [-0.000,+0.000]|+0.000 [+0.000,+0.000]|

JSON retains absolute summer scores, winter absolute backcast differences, same-circle and low-counter sensitivities. Supports, species, calibration counts and paired bootstrap truth must match or this summary fails. Sparse repeats share test observations; they are not independent ecological replications. No production adoption is automated.

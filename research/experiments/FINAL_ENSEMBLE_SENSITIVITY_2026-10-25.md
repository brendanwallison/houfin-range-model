# Original two-seed versus four-seed sensitivity

This compares the original candidates' mean features from seeds1/2 against seeds1/2/3/4, using completed evaluations. It does not involve the clean-completeness intervention. Target columns, truth bootstrap components, winter species and test rows match. Intervals are paired over observations/blocks/circles, conditional on these fixed fitted models; they do not sample the distribution of possible training-seed subsets.

## Summer: full calibration

Mean2 minus mean4 in noise-corrected change correlation,1966–1986 to2005–2025.

| Representation | Within difference [95%] | Pooled difference [95%] |
|---|---|---|
|v1 summer64|−.004 [−.013,.005]|−.011 [−.019,−.005]|
|v2 summer32+winter32|−.009 [−.019,.000]|+.002 [−.005,.009]|
|v2 summer64+winter64|−.008 [−.015,−.001]|−.011 [−.022,.001]|

Some absolute score offsets are measurable even though both constructions use the same trained seeds/bases. At3% calibration, the full-v2 within difference is−.018 [−.032,−.003] for thinning seed1 but+.007 [−.015,.028] for seed2. This is another reason to use matched ensemble sizes in the clean-effect comparison.

## Winter: early temporal gain

Mean2 minus mean4 in conditional list score gain over each fitted readout's own modern-feature placebo. Positive favors mean2. This measures recorded composition, not latent abundance.

| Era | v1 difference [95%] | v2 full128 difference [95%] |
|---|---|---|
|1902–1919|−.040 [−.074,−.003]|+.099 [.061,.138]|
|1920–1939|−.012 [−.037,.011]|+.066 [.044,.097]|
|1966–1985|+.010 [.005,.015]|+.002 [−.006,.010]|

Four seeds do not uniformly improve early recorded-list scores. The change is material to historical comparisons, so the earlier four-seed winter contrasts should not be silently substituted for the matched two-seed clean baseline. Existing early-era observation and same-circle support limitations still apply.

The stored global feature-disagreement diagnostic has only modestly larger early RMS seed spread: mean1902–1939 versus1996–2025 is.0190/.0168 for v1 and.0191/.0169 for v2. These are aggregates over evaluation cells and coordinates; they do not explain the list-score shifts or calibrate historical habitat uncertainty.

Retain the planned four-seed final ensembles as a separate finalization step. The current clean two-seed exercise tests the completeness intervention with matched controls; it is not evidence that two seeds suffice for production.

Full numerical results: FINAL_ENSEMBLE_SENSITIVITY_20261025.json. Source: research/exp/final_ensemble_sensitivity.py; inputs are the original FINAL_HOLDOUT run and FINAL_CLEAN/evaluation/original2.

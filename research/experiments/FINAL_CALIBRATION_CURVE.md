# Calibration-size sensitivity

The fixed10% experiment completed all6scores (three representations × thinning seeds1/2). It reuses the original four-seed encoders and the existing observation-aware scorer. The clean-completeness evaluation remains separate. All comparisons use identical target columns and paired truth/bootstrap components.

Full v2 (summer64+winter64) minus v1 (summer64), in noise-corrected change correlation:

| Calibration fraction | Actual rows | Within difference [95%] | Pooled difference [95%] |
|---|---:|---|---|
|3%, seed1|1518|−.004 [−.025,.018]|+.021 [.006,.035]|
|3%, seed2|1445|−.041 [−.062,−.017]|+.021 [.004,.039]|
|10%, seed1|4895|+.012 [−.004,.028]|+.041 [.025,.054]|
|10%, seed2|4814|+.027 [.012,.041]|+.032 [.016,.048]|
|100%, shared fit|48844|+.028 [.011,.046]|+.047 [.030,.063]|

At10%, both within-cell point differences become positive, with seed2's interval excluding zero. At3%, all four thinning-repeat point differences were negative (the additional seeds3/4 are reported separately). Pooled gains remain positive throughout. This supports calibration-size dependence; it does not estimate a universal threshold or prove monotonic improvement for every thinning realization.

The equal-width32+32 representation retains a within-cell advantage but loses pooled performance relative to v1 at10% and full calibration. The figure includes both widths and both readouts.

These are conditional paired block-bootstrap intervals on shared test observations, not independent ecological replications or a new confirmation set. The original encoders still have the completeness leakage. Do not directly translate this sample-size curve into House Finch prior settings: its likelihood, parameter sharing and effective information differ.

Figure: FINAL_CALIBRATION_CURVE.png. Numerical details: FINAL_CALIBRATION_CURVE.json. Early window1966–1986; modern contrast2005–2025; target-readout calibration1996+. The100% point is one common fit, not two independent repeats.

# Why Plains Runge contribution changes across fits

Modern 2010-2025, 380 equal-weight land cells in 100-95W and 25-49N. The 65-91% range describes the fraction of cells below C=1 across model versions; it is not a confidence interval or a temporal fluctuation.

The exact implemented identity is C_i = p_adult,i + (J_i/A_i)*p_recruit,i. Each p sums successful survival/recruitment over all destinations from that origin in the fitted environment. J/A is actual offspring per breeder in the starting postbirth state; it is not Fmax. C also equals retained contribution plus successful export. Reproduction/Allee suppression therefore enters through actual offspring, while origin movement and destination/journey survival enter through the two p terms. Emigration alone does not lower C: successful emigrants are credited to their origins.

| Fit | Adult contribution | Offspring contribution | Total C | Retained | Successful export | Sink cells | C<.8 cells | Within .05 of 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step34_abs_comp_natdep_s41 | 0.348 | 0.493 | 0.841 | 0.396 | 0.445 | 246 | 137 | 106 |
| step34_broodhi | 0.156 | 0.510 | 0.666 | 0.473 | 0.193 | 344 | 215 | 61 |
| step35_est1951_ctl_mid_s41 | 0.199 | 0.453 | 0.651 | 0.482 | 0.170 | 333 | 223 | 58 |
| step35_est1951_nat_mid_s41 | 0.201 | 0.564 | 0.765 | 0.487 | 0.278 | 290 | 173 | 100 |

The matched later-state middle-abundance control/native contrast changes C .651 to .765: adult contribution .199 to .201, offspring contribution .453 to .564. Retention is nearly unchanged (.482 to .487), while successful export increases .170 to .278. This attributes the change algebraically, not to one isolated causal coefficient: both are fully refitted models.

The two fits at the endpoints of the sink-prevalence range have mean absolute C difference 0.179, 102 exact crossings of 1 and 27 reversals beyond .95/1.05. Hence threshold sensitivity contributes but does not explain the whole range. These endpoints differ in multiple assumptions and are not a matched single-factor experiment.

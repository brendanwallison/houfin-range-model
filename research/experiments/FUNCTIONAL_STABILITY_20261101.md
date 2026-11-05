# Functional process stability across fitted experiments

Compares realized source/sink structure, local replacement and movement directly. These are cross-fit robustness measurements, not proof of statistical identifiability or posterior uncertainty. Component non-identifiability does not imply process non-identifiability. Agreement imposed by fixed kernels/shared fields must also be distinguished from information supplied by data.

All comparisons use common grid/time support. The primary table weights cells by the same actual positive BBS record counts, independently of either fitted population. Equal-land and symmetric pooled-breeder weights, four regions and early/modern eras are retained in JSON. Coverage measures common finite support; absent support is not classified as agreement. Replacement/contribution classes use sink <.95, source >1.05, and a neutral band; .01 thresholds are also saved. This avoids treating tiny fluctuations around replacement as large structural changes.

| Contrast | Left to right | C map correlation | Mean absolute C change | C class agreement | Source/sink reversal | Local replacement correlation | Net flux correlation | Mean absolute export change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| initial abundance | step35_est1951_ctl_mid_s41 → step35_est1951_ctl_lo_s41 | 0.999 | 0.006 | 0.955 | 0.000 | 0.999 | 0.999 | 0.007 |
| initial abundance | step35_est1951_ctl_mid_s41 → step35_est1951_ctl_hi_s41 | 0.997 | 0.009 | 0.949 | 0.000 | 0.997 | 0.997 | 0.018 |
| later-state seed | step35_est1951_ctl_mid_s41 → step35_est1951_ctl_mid_s42 | 0.999 | 0.003 | 0.983 | 0.000 | 0.999 | 0.999 | 0.004 |
| later-state NestWatch | step35_est1951_ctl_mid_s41 → step36_est1951_nonest_ctl_s41 | 0.965 | 0.035 | 0.829 | 0.001 | 0.969 | 0.947 | 0.072 |
| initial abundance | step35_est1951_nat_mid_s41 → step35_est1951_nat_lo_s41 | 0.977 | 0.021 | 0.892 | 0.000 | 0.974 | 0.973 | 0.025 |
| initial abundance | step35_est1951_nat_mid_s41 → step35_est1951_nat_hi_s41 | 0.994 | 0.010 | 0.940 | 0.000 | 0.991 | 0.996 | 0.017 |
| later-state seed | step35_est1951_nat_mid_s41 → step35_est1951_nat_mid_s42 | 0.984 | 0.013 | 0.936 | 0.000 | 0.979 | 0.949 | 0.019 |
| original absolute seed | step33_nestcbc_abs_s41 → step34_abs_s42 | 0.999 | 0.005 | 0.967 | 0.000 | 0.999 | 0.999 | 0.007 |
| original native seed | step34_abs_natdep_s41 → step34_abs_natdep_s42 | 0.954 | 0.020 | 0.894 | 0.002 | 0.941 | 0.931 | 0.036 |
| initialization formulation | step33_nestcbc_abs_s41 → step35_est1951_ctl_mid_s41 | 0.952 | 0.044 | 0.771 | 0.006 | 0.957 | 0.962 | 0.117 |
| initialization formulation | step34_abs_natdep_s41 → step35_est1951_nat_mid_s41 | 0.977 | 0.023 | 0.872 | 0.000 | 0.950 | 0.949 | 0.042 |
| native departure | step33_nestcbc_abs_s41 → step34_abs_natdep_s41 | 0.829 | 0.063 | 0.662 | 0.034 | 0.770 | 0.676 | 0.158 |
| later-state native departure | step35_est1951_ctl_mid_s41 → step35_est1951_nat_mid_s41 | 0.864 | 0.062 | 0.746 | 0.035 | 0.829 | 0.817 | 0.166 |
| DESK basis | step34_abs_natdep_s41 → step34_abs_v2_natdep_s41 | 0.956 | 0.039 | 0.788 | 0.001 | 0.922 | 0.945 | 0.058 |
| NestWatch | step33_nestcbc_abs_s41 → step36_nonest_ctl_s41 | 0.969 | 0.027 | 0.842 | 0.002 | 0.974 | 0.944 | 0.051 |
| NestWatch | step34_abs_natdep_s41 → step36_nonest_nat_s41 | 0.954 | 0.026 | 0.870 | 0.001 | 0.966 | 0.920 | 0.034 |
| Allee law | step33_nestcbc_s41 → step33_nestcbc_abs_s41 | 0.948 | 0.046 | 0.737 | 0.007 | 0.961 | 0.952 | 0.097 |

Local total replacement is the actual-state no-movement next-year/previous-year postbirth population ratio; C breeding follows breeder-origin survivors/recruits across all destinations. They use different censuses. Net cell flux is gross immigration minus successful emigration per starting individual; it includes exchanges within regions. Gross export above is successful breeder export per source breeder.

The fitted-state comparison does not establish that growth responses agree away from fitted densities, that complete origin/destination transport matrices agree, or that stochastic establishment/forecasts are identified. Those require common-state perturbations, prediction tests and posterior/simulation-based evaluation.

## Interpretation

Founder abundance is not making every realized process fluctuate equally. Across the 280/560/1120 scenarios, modern C map correlations against the middle scenario are .997-.999 for the control and .977-.994 with native departure. Mean absolute C changes are .006-.009 and .010-.021, respectively. There are no direct source-to-sink reversals on modern positive-record support at the .05 threshold; disagreements are with the neutral band. The control retains 94.9-95.5% class agreement, native departure 89.2-94.0%. These are substantially more stable than the native fits' age-specific survival allocation (adult .389-.445, juvenile/adult .801-.962).

Changing the movement mechanism is a different contrast. Adding native departure in the later-state middle scenario changes the western C map appreciably: modern positive-record-weighted correlation .769, mean absolute change .121, class agreement 54.6%, and direct source/sink reversal 10.7%. National values dilute this regional change. Early western reversal is 14.0%. Thus stable regional average C does not establish stable spatial contribution structure.

Equal-land results also preserve the initial-abundance C-map similarity (.993-.999 correlations). However, net-flux correlation for native low versus middle abundance falls to .794 on equal land, versus .973 on positive-record support. Stability is weaker away from observed habitat and cannot be generalized to every movement quantity.

Independent historical/abundance and demographic observations remain valuable to estimate initial conditions and separate survival/reproduction. The present comparisons do not justify saying that the entire experiment is unconstrained, or that all realized growth and movement outputs are unidentified. These scenarios fix abundance rather than estimating a freely floating parameter; they are a sensitivity bracket, not a profile likelihood.

## Regional movement-mechanism comparison

Later-state middle abundance, control versus native departure; common positive-record weights. Fractions below are record-weighted shares, not independent-cell percentages.

| Era | Region | C correlation | Mean absolute C change | Class agreement | Direct source/sink reversal | Net flux correlation |
|---|---|---:|---:|---:|---:|---:|
| bbs_early | all | 0.852 | 0.137 | 0.598 | 0.097 | 0.817 |
| bbs_early | west | 0.620 | 0.166 | 0.433 | 0.140 | 0.825 |
| bbs_early | plains | 0.952 | 0.233 | 0.570 | 0.059 | 0.954 |
| bbs_early | east | 0.982 | 0.056 | 0.992 | 0.000 | 0.975 |
| modern | all | 0.864 | 0.062 | 0.746 | 0.035 | 0.817 |
| modern | west | 0.769 | 0.121 | 0.546 | 0.107 | 0.833 |
| modern | plains | 0.925 | 0.093 | 0.695 | 0.013 | 0.914 |
| modern | east | 0.973 | 0.018 | 0.876 | 0.000 | 0.931 |

## Contribution versus potential lambda stability

Potential lambda here is Sa + Sj * Fmax before Allee/density suppression, density-dependent disease and dispersal losses. In this yearling-breeding life cycle it is a potential local growth diagnostic, not actual growth at zero density under absolute Allee. Annual lambda values are weighted by starting breeders, matching the within-cell temporal exposure weighting of breeding contribution. The same positive-record cell weights are used for both. Actual-state local total replacement is included separately; it is a one-step population ratio, not an eigenvalue.

| Contrast | Right fit (relative to left in main table) | C correlation | Potential lambda correlation | C mean absolute change | Lambda mean absolute change | C=1 crossing cells | Lambda=1 crossing cells | Local replacement correlation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| initial abundance | step35_est1951_ctl_lo_s41 | 0.999 | 0.999 | 0.006 | 0.085 | 27 | 15 | 0.999 |
| initial abundance | step35_est1951_ctl_hi_s41 | 0.997 | 0.998 | 0.009 | 0.119 | 39 | 38 | 0.997 |
| later-state seed | step35_est1951_ctl_mid_s42 | 0.999 | 1.000 | 0.003 | 0.039 | 13 | 11 | 0.999 |
| later-state NestWatch | step36_est1951_nonest_ctl_s41 | 0.965 | 0.985 | 0.035 | 0.876 | 148 | 68 | 0.969 |
| initial abundance | step35_est1951_nat_lo_s41 | 0.977 | 0.990 | 0.021 | 0.361 | 75 | 93 | 0.974 |
| initial abundance | step35_est1951_nat_hi_s41 | 0.994 | 0.995 | 0.010 | 0.152 | 53 | 34 | 0.991 |
| later-state seed | step35_est1951_nat_mid_s42 | 0.984 | 0.993 | 0.013 | 0.180 | 56 | 62 | 0.979 |
| original absolute seed | step34_abs_s42 | 0.999 | 1.000 | 0.005 | 0.039 | 42 | 20 | 0.999 |
| original native seed | step34_abs_natdep_s42 | 0.954 | 0.988 | 0.020 | 0.175 | 101 | 101 | 0.941 |
| initialization formulation | step35_est1951_ctl_mid_s41 | 0.952 | 0.992 | 0.044 | 0.330 | 175 | 56 | 0.957 |
| initialization formulation | step35_est1951_nat_mid_s41 | 0.977 | 0.987 | 0.023 | 0.359 | 109 | 65 | 0.950 |
| native departure | step34_abs_natdep_s41 | 0.829 | 0.960 | 0.063 | 0.328 | 334 | 234 | 0.770 |
| later-state native departure | step35_est1951_nat_mid_s41 | 0.864 | 0.965 | 0.062 | 0.341 | 276 | 183 | 0.829 |
| DESK basis | step34_abs_v2_natdep_s41 | 0.956 | 0.984 | 0.039 | 0.236 | 138 | 122 | 0.922 |
| NestWatch | step36_nonest_ctl_s41 | 0.969 | 0.983 | 0.027 | 0.667 | 144 | 79 | 0.974 |
| NestWatch | step36_nonest_nat_s41 | 0.954 | 0.992 | 0.026 | 0.239 | 121 | 54 | 0.966 |
| Allee law | step33_nestcbc_abs_s41 | 0.948 | 0.977 | 0.046 | 1.067 | 180 | 99 | 0.961 |

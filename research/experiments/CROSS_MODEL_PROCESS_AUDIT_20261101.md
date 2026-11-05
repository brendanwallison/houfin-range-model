# Cross-model vital rates and aggregate growth processes

Includes all 101 locally READY contribution audits with the same standard yearling-breeding life cycle. This includes Allee laws, DESK bases, native departure, journey penalties, nesting/annual conversion, founder anchors and initialization formulations, plus seeds. It does not include older unsupported/different-life-cycle versions. Every fit is compared against the same reference (step33_nestcbc_abs_s41); selected matched structural contrasts are also shown.

The question is whether different model versions trade vital rates while preserving aggregate growth and movement at matched locations and years. Within-era cross-model comparisons below answer that question; historical change within one fit is a separate question. Correlation is accompanied by absolute differences and exact crossings of 1.

All-land comparisons give each finite land cell equal weight. Fixed-observed support gives equal weight to the same cells with a positive modern BBS record, regardless of era or fitted model abundance. Potential lambda is annual-mean Sa+Sj*Fmax before density/Allee suppression, density-dependent disease and movement losses. Sa/Sj are the saved local survivals after density-dependent disease; reproduction is realized offspring per breeder, not Fmax. C and no-movement replacement use their respective fitted-stage exposure weights within each era; they therefore compare realized fitted-state processes, not responses at a common imposed density. Net flux is immigration minus successful emigration per starting individual. Cross-fit robustness does not establish statistical identification or agree on responses away from fitted states.

## Uniform metrics for the three illustrative structural contrasts

All rows below use 2010-2025, identical fixed-observed cells within each region, and equal cell weights (East 206; Great Plains 189; West 704). The primary Great Plains band is 100W-95W. Mean absolute changes are calculated cell by cell before averaging; they are not changes in regional means. These three diagnostics have different ranges and meanings, so their raw magnitudes are not a calibrated measure of relative identifiability.

| Contrast | Region | Mean absolute C change | Mean absolute potential lambda change | Mean absolute local replacement change |
|---|---|---:|---:|---:|
| annual nest conversion low | East | 0.026 | 0.069 | 0.044 |
| annual nest conversion low | Great Plains | 0.028 | 0.069 | 0.042 |
| annual nest conversion low | West | 0.023 | 0.089 | 0.051 |
| Allee law | East | 0.050 | 1.032 | 0.050 |
| Allee law | Great Plains | 0.046 | 0.576 | 0.055 |
| Allee law | West | 0.057 | 0.704 | 0.062 |
| later-state native departure | East | 0.027 | 0.144 | 0.044 |
| later-state native departure | Great Plains | 0.107 | 0.219 | 0.098 |
| later-state native departure | West | 0.133 | 0.534 | 0.152 |

For a consistent comparison between components and aggregates, the next table applies the SAME mean symmetric relative difference to every quantity: 100 * mean(2*abs(right-left)/(abs(left)+abs(right))). It is not conventional baseline-relative percent change and is not posterior uncertainty.

| Contrast | Region | Adult survival | Juvenile survival | Realized reproduction | Potential lambda | C | Local replacement |
|---|---|---:|---:|---:|---:|---:|---:|
| annual nest conversion low | East | 6.1% | 23.2% | 24.6% | 3.5% | 3.5% | 5.7% |
| annual nest conversion low | Great Plains | 7.6% | 23.6% | 23.8% | 6.3% | 4.4% | 6.3% |
| annual nest conversion low | West | 6.7% | 26.0% | 24.2% | 6.7% | 3.3% | 8.0% |
| Allee law | East | 49.8% | 19.4% | 16.0% | 34.4% | 8.0% | 9.4% |
| Allee law | Great Plains | 41.4% | 22.6% | 16.1% | 34.4% | 7.8% | 10.9% |
| Allee law | West | 38.1% | 37.5% | 16.0% | 29.4% | 9.8% | 13.8% |
| later-state native departure | East | 9.0% | 22.2% | 21.8% | 8.4% | 4.7% | 7.2% |
| later-state native departure | Great Plains | 6.1% | 30.8% | 14.1% | 22.8% | 16.8% | 18.5% |
| later-state native departure | West | 11.8% | 41.9% | 21.4% | 40.1% | 19.2% | 27.5% |

## Interpretation of cross-model compensation

There is evidence for partial conservation of realized growth, but not a single invariant growth process across all structures. Three matched modern contrasts illustrate the distinction, using equal weights on fixed positive-BBS cells (206 eastern cells or 704 western cells). These supports differ from the earlier record-weighted summaries.

- Lowering annual nest conversion in the absolute control changes eastern juvenile survival and realized reproduction by mean symmetric relative differences of 23.2% and 24.6%, versus 3.5% for C. Mean potential lambda is 2.222 versus 2.249; mean absolute map changes are .069 for lambda, .026 for C and .044 for actual-state local replacement. This supports demographic compensation with relatively conserved realized growth in that contrast. Symmetric relative difference is 2*abs(right-left)/(abs(left)+abs(right)), averaged equally over the fixed cells; it is not an uncertainty interval.

- Changing relative to absolute Allee changes eastern mean potential lambda from 3.246 to 2.222 (mean absolute map change 1.032). Yet mean absolute changes in C and actual-state local replacement are .050 and .050. Thus fitted-state replacement is appreciably more stable than potential growth capacity: this is not conservation of the whole growth law.

- Adding native departure to the later-state middle-abundance fit changes western mean potential lambda from 1.778 to 2.253; mean absolute changes are .133 in C and .152 in local replacement, with 50/704 cells reversing between C<.95 and C>1.05. This is a movement-structure change that is not fully absorbed by vital-rate compensation.

These statements compare model versions at the same places and eras; none relies on historical changes within a model. Stable fitted-state replacement can reflect the count likelihood constraining trajectories while allowing different density dependence, maximum growth or transport mechanisms. Establishing an identifiable growth response requires testing common-state perturbations and uncertainty, not just fitted-state agreement.

## Matched structural comparisons in the East (all land)

| Contrast | Era | Adult survival means | Juvenile survival means | Realized reproduction means | Potential lambda means | Lambda mean absolute change | Lambda crossing cells | C mean absolute change | C crossing cells | Local replacement mean absolute change |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| Allee law | 1951-1965 | 0.134 to 0.142 | 0.044 to 0.065 | 0.032 to 0.023 | 0.336 to 0.429 | 0.134 | 29 | 0.045 | 0 | 0.047 |
| Allee law | 1966-1985 | 0.155 to 0.177 | 0.076 to 0.082 | 0.786 to 0.763 | 0.611 to 0.624 | 0.135 | 10 | 0.054 | 5 | 0.069 |
| Allee law | 2010-2025 | 0.152 to 0.192 | 0.094 to 0.090 | 2.528 to 2.257 | 1.792 to 1.239 | 0.564 | 12 | 0.054 | 24 | 0.058 |
| native departure | 1951-1965 | 0.142 to 0.218 | 0.065 to 0.097 | 0.023 to 0.023 | 0.429 to 0.505 | 0.093 | 13 | 0.076 | 0 | 0.075 |
| native departure | 1966-1985 | 0.177 to 0.244 | 0.082 to 0.110 | 0.763 to 0.723 | 0.624 to 0.717 | 0.103 | 15 | 0.081 | 10 | 0.084 |
| native departure | 2010-2025 | 0.192 to 0.232 | 0.090 to 0.106 | 2.257 to 2.241 | 1.239 to 1.342 | 0.133 | 12 | 0.054 | 27 | 0.061 |
| DESK basis with native departure | 1951-1965 | 0.218 to 0.173 | 0.097 to 0.071 | 0.023 to 0.021 | 0.505 to 0.435 | 0.081 | 15 | 0.050 | 0 | 0.050 |
| DESK basis with native departure | 1966-1985 | 0.244 to 0.188 | 0.110 to 0.097 | 0.723 to 0.677 | 0.717 to 0.637 | 0.090 | 14 | 0.049 | 12 | 0.066 |
| DESK basis with native departure | 2010-2025 | 0.232 to 0.174 | 0.106 to 0.101 | 2.241 to 2.072 | 1.342 to 1.238 | 0.129 | 13 | 0.034 | 21 | 0.038 |
| DESK basis control | 1951-1965 | 0.142 to 0.133 | 0.065 to 0.073 | 0.023 to 0.021 | 0.429 to 0.388 | 0.048 | 14 | 0.012 | 0 | 0.012 |
| DESK basis control | 1966-1985 | 0.177 to 0.167 | 0.082 to 0.094 | 0.763 to 0.678 | 0.624 to 0.592 | 0.045 | 7 | 0.022 | 6 | 0.028 |
| DESK basis control | 2010-2025 | 0.192 to 0.177 | 0.090 to 0.101 | 2.257 to 2.001 | 1.239 to 1.163 | 0.093 | 5 | 0.023 | 19 | 0.023 |
| initialization formulation control | 1951-1965 | 0.142 to 0.132 | 0.065 to 0.051 | 0.023 to 0.045 | 0.429 to 0.303 | 0.149 | 44 | 0.043 | 2 | 0.044 |
| initialization formulation control | 1966-1985 | 0.177 to 0.160 | 0.082 to 0.087 | 0.763 to 0.638 | 0.624 to 0.538 | 0.109 | 14 | 0.041 | 4 | 0.064 |
| initialization formulation control | 2010-2025 | 0.192 to 0.163 | 0.090 to 0.109 | 2.257 to 1.905 | 1.239 to 1.430 | 0.203 | 5 | 0.044 | 32 | 0.049 |
| initialization formulation native | 1951-1965 | 0.218 to 0.122 | 0.097 to 0.098 | 0.023 to 0.056 | 0.505 to 0.366 | 0.140 | 33 | 0.097 | 2 | 0.098 |
| initialization formulation native | 1966-1985 | 0.244 to 0.148 | 0.110 to 0.121 | 0.723 to 0.711 | 0.717 to 0.612 | 0.105 | 16 | 0.068 | 10 | 0.086 |
| initialization formulation native | 2010-2025 | 0.232 to 0.143 | 0.106 to 0.120 | 2.241 to 2.143 | 1.342 to 1.435 | 0.202 | 8 | 0.047 | 16 | 0.056 |
| later-state native departure | 1951-1965 | 0.132 to 0.122 | 0.051 to 0.098 | 0.045 to 0.056 | 0.303 to 0.366 | 0.077 | 20 | 0.013 | 0 | 0.014 |
| later-state native departure | 1966-1985 | 0.160 to 0.148 | 0.087 to 0.121 | 0.638 to 0.711 | 0.538 to 0.612 | 0.091 | 15 | 0.039 | 6 | 0.046 |
| later-state native departure | 2010-2025 | 0.163 to 0.143 | 0.109 to 0.120 | 1.905 to 2.143 | 1.430 to 1.435 | 0.096 | 7 | 0.027 | 11 | 0.038 |
| NestWatch control | 1951-1965 | 0.142 to 0.169 | 0.065 to 0.078 | 0.023 to 0.017 | 0.429 to 0.391 | 0.059 | 14 | 0.028 | 0 | 0.028 |
| NestWatch control | 1966-1985 | 0.177 to 0.208 | 0.082 to 0.098 | 0.763 to 0.628 | 0.624 to 0.654 | 0.055 | 3 | 0.039 | 5 | 0.033 |
| NestWatch control | 2010-2025 | 0.192 to 0.215 | 0.090 to 0.103 | 2.257 to 2.108 | 1.239 to 1.533 | 0.298 | 10 | 0.037 | 21 | 0.030 |
| NestWatch native | 1951-1965 | 0.218 to 0.394 | 0.097 to 0.139 | 0.023 to 0.022 | 0.505 to 0.630 | 0.138 | 8 | 0.175 | 0 | 0.175 |
| NestWatch native | 1966-1985 | 0.244 to 0.412 | 0.110 to 0.147 | 0.723 to 0.635 | 0.717 to 0.876 | 0.163 | 7 | 0.139 | 11 | 0.130 |
| NestWatch native | 2010-2025 | 0.232 to 0.359 | 0.106 to 0.129 | 2.241 to 2.358 | 1.342 to 1.528 | 0.189 | 7 | 0.115 | 26 | 0.113 |
| annual nest conversion low | 1951-1965 | 0.142 to 0.137 | 0.065 to 0.084 | 0.023 to 0.020 | 0.429 to 0.394 | 0.038 | 13 | 0.007 | 0 | 0.007 |
| annual nest conversion low | 1966-1985 | 0.177 to 0.171 | 0.082 to 0.106 | 0.763 to 0.611 | 0.624 to 0.588 | 0.039 | 7 | 0.014 | 1 | 0.026 |
| annual nest conversion low | 2010-2025 | 0.192 to 0.183 | 0.090 to 0.115 | 2.257 to 1.777 | 1.239 to 1.250 | 0.041 | 2 | 0.019 | 28 | 0.025 |
| annual nest conversion high | 1951-1965 | 0.142 to 0.125 | 0.065 to 0.049 | 0.023 to 0.022 | 0.429 to 0.325 | 0.124 | 33 | 0.042 | 0 | 0.042 |
| annual nest conversion high | 1966-1985 | 0.177 to 0.146 | 0.082 to 0.079 | 0.763 to 0.704 | 0.624 to 0.549 | 0.097 | 11 | 0.045 | 5 | 0.061 |
| annual nest conversion high | 2010-2025 | 0.192 to 0.149 | 0.090 to 0.097 | 2.257 to 2.209 | 1.239 to 1.376 | 0.151 | 8 | 0.044 | 24 | 0.042 |
| yearling baseline to absolute/nest/anchor | 1951-1965 | 0.121 to 0.142 | 0.084 to 0.065 | 0.023 to 0.023 | 0.299 to 0.429 | 0.143 | 34 | 0.033 | 0 | 0.033 |
| yearling baseline to absolute/nest/anchor | 1966-1985 | 0.156 to 0.177 | 0.111 to 0.082 | 0.636 to 0.763 | 0.609 to 0.624 | 0.144 | 12 | 0.047 | 7 | 0.046 |
| yearling baseline to absolute/nest/anchor | 2010-2025 | 0.168 to 0.192 | 0.123 to 0.090 | 2.153 to 2.257 | 2.119 to 1.239 | 0.890 | 15 | 0.051 | 35 | 0.055 |
| CBC likelihood free abundance seed41 | 1951-1965 | 0.141 to 0.133 | 0.064 to 0.027 | 0.031 to 0.124 | 0.426 to 0.368 | 0.357 | 84 | 0.086 | 6 | 0.092 |
| CBC likelihood free abundance seed41 | 1966-1985 | 0.177 to 0.147 | 0.082 to 0.050 | 0.763 to 1.688 | 0.624 to 0.617 | 0.360 | 68 | 0.151 | 53 | 0.155 |
| CBC likelihood free abundance seed41 | 2010-2025 | 0.192 to 0.125 | 0.090 to 0.076 | 2.265 to 4.154 | 1.258 to 2.117 | 1.006 | 87 | 0.113 | 65 | 0.120 |
| CBC likelihood free abundance seed42 | 1951-1965 | 0.144 to 0.137 | 0.065 to 0.028 | 0.030 to 0.117 | 0.433 to 0.364 | 0.360 | 83 | 0.091 | 6 | 0.095 |
| CBC likelihood free abundance seed42 | 1966-1985 | 0.179 to 0.153 | 0.082 to 0.052 | 0.762 to 1.594 | 0.629 to 0.626 | 0.371 | 66 | 0.158 | 55 | 0.161 |
| CBC likelihood free abundance seed42 | 2010-2025 | 0.194 to 0.129 | 0.090 to 0.079 | 2.258 to 4.080 | 1.256 to 2.212 | 1.074 | 92 | 0.123 | 57 | 0.127 |
| CBC initial abundance profile seed41 | 1951-1965 | 0.133 to 0.121 | 0.027 to 0.026 | 0.124 to 0.141 | 0.368 to 0.329 | 0.052 | 7 | 0.021 | 2 | 0.022 |
| CBC initial abundance profile seed41 | 1966-1985 | 0.147 to 0.139 | 0.050 to 0.054 | 1.688 to 1.551 | 0.617 to 0.625 | 0.058 | 8 | 0.027 | 14 | 0.032 |
| CBC initial abundance profile seed41 | 2010-2025 | 0.125 to 0.123 | 0.076 to 0.077 | 4.154 to 3.888 | 2.117 to 2.321 | 0.340 | 9 | 0.028 | 12 | 0.031 |
| CBC initial abundance profile seed42 | 1951-1965 | 0.137 to 0.122 | 0.028 to 0.027 | 0.117 to 0.146 | 0.364 to 0.340 | 0.038 | 8 | 0.025 | 2 | 0.024 |
| CBC initial abundance profile seed42 | 1966-1985 | 0.153 to 0.139 | 0.052 to 0.052 | 1.594 to 1.570 | 0.626 to 0.621 | 0.051 | 4 | 0.024 | 11 | 0.027 |
| CBC initial abundance profile seed42 | 2010-2025 | 0.129 to 0.123 | 0.079 to 0.077 | 4.080 to 4.015 | 2.212 to 2.278 | 0.182 | 13 | 0.025 | 6 | 0.027 |
| historical range replacement seed41 | 1951-1965 | 0.141 to 0.132 | 0.064 to 0.068 | 0.031 to 0.029 | 0.426 to 0.415 | 0.029 | 12 | 0.012 | 0 | 0.012 |
| historical range replacement seed41 | 1966-1985 | 0.177 to 0.164 | 0.082 to 0.085 | 0.763 to 0.778 | 0.624 to 0.602 | 0.038 | 7 | 0.015 | 2 | 0.018 |
| historical range replacement seed41 | 2010-2025 | 0.192 to 0.179 | 0.090 to 0.094 | 2.265 to 2.198 | 1.258 to 1.222 | 0.088 | 7 | 0.015 | 9 | 0.018 |
| dated1951-1965 range supervision seed41 | 1951-1965 | 0.132 to 0.115 | 0.068 to 0.068 | 0.029 to 0.067 | 0.415 to 0.381 | 0.046 | 16 | 0.019 | 2 | 0.021 |
| dated1951-1965 range supervision seed41 | 1966-1985 | 0.164 to 0.143 | 0.085 to 0.087 | 0.778 to 0.871 | 0.602 to 0.554 | 0.058 | 10 | 0.020 | 5 | 0.023 |
| dated1951-1965 range supervision seed41 | 2010-2025 | 0.179 to 0.160 | 0.094 to 0.100 | 2.198 to 2.196 | 1.222 to 1.205 | 0.074 | 3 | 0.015 | 20 | 0.017 |
| floor with factor_p1_rational_adult20 seed41 | 1951-1965 | 0.203 to 0.200 | 0.075 to 0.074 | 0.023 to 0.024 | 0.459 to 0.454 | 0.006 | 1 | 0.003 | 0 | 0.003 |
| floor with factor_p1_rational_adult20 seed41 | 1966-1985 | 0.224 to 0.221 | 0.111 to 0.109 | 0.607 to 0.616 | 0.696 to 0.689 | 0.008 | 0 | 0.004 | 1 | 0.004 |
| floor with factor_p1_rational_adult20 seed41 | 2010-2025 | 0.203 to 0.201 | 0.120 to 0.119 | 1.910 to 1.931 | 1.550 to 1.539 | 0.012 | 3 | 0.003 | 1 | 0.003 |
| floor with factor_p2_veit seed41 | 1951-1965 | 0.173 to 0.174 | 0.053 to 0.052 | 0.024 to 0.025 | 0.347 to 0.346 | 0.009 | 1 | 0.004 | 0 | 0.004 |
| floor with factor_p2_veit seed41 | 1966-1985 | 0.229 to 0.231 | 0.071 to 0.070 | 0.653 to 0.658 | 0.511 to 0.513 | 0.009 | 3 | 0.014 | 1 | 0.007 |
| floor with factor_p2_veit seed41 | 2010-2025 | 0.248 to 0.249 | 0.077 to 0.076 | 2.011 to 2.042 | 0.947 to 0.958 | 0.018 | 2 | 0.012 | 6 | 0.006 |
| floor with factor_p2_rational_adult20 seed41 | 1951-1965 | 0.200 to 0.199 | 0.087 to 0.087 | 0.026 to 0.026 | 0.450 to 0.450 | 0.002 | 1 | 0.001 | 0 | 0.001 |
| floor with factor_p2_rational_adult20 seed41 | 1966-1985 | 0.244 to 0.244 | 0.107 to 0.107 | 0.641 to 0.639 | 0.617 to 0.618 | 0.002 | 0 | 0.002 | 0 | 0.001 |
| floor with factor_p2_rational_adult20 seed41 | 2010-2025 | 0.227 to 0.227 | 0.101 to 0.101 | 2.060 to 2.060 | 1.137 to 1.141 | 0.005 | 1 | 0.001 | 1 | 0.001 |
| floored departure power1 seed41 | 1951-1965 | 0.135 to 0.200 | 0.056 to 0.074 | 0.022 to 0.024 | 0.330 to 0.454 | 0.124 | 25 | 0.068 | 0 | 0.068 |
| floored departure power1 seed41 | 1966-1985 | 0.160 to 0.221 | 0.089 to 0.109 | 0.625 to 0.616 | 0.552 to 0.689 | 0.137 | 19 | 0.091 | 9 | 0.093 |
| floored departure power1 seed41 | 2010-2025 | 0.163 to 0.201 | 0.108 to 0.119 | 1.918 to 1.931 | 1.349 to 1.539 | 0.192 | 15 | 0.056 | 9 | 0.062 |
| floored curvature Veit seed41 | 1951-1965 | 0.135 to 0.174 | 0.056 to 0.052 | 0.022 to 0.025 | 0.330 to 0.346 | 0.056 | 15 | 0.074 | 0 | 0.074 |
| floored curvature Veit seed41 | 1966-1985 | 0.160 to 0.231 | 0.089 to 0.070 | 0.625 to 0.658 | 0.552 to 0.513 | 0.077 | 6 | 0.045 | 6 | 0.101 |
| floored curvature Veit seed41 | 2010-2025 | 0.163 to 0.249 | 0.108 to 0.076 | 1.918 to 2.042 | 1.349 to 0.958 | 0.394 | 11 | 0.040 | 11 | 0.071 |
| floored curvature rational seed41 | 1951-1965 | 0.200 to 0.199 | 0.074 to 0.087 | 0.024 to 0.026 | 0.454 to 0.450 | 0.077 | 6 | 0.073 | 0 | 0.073 |
| floored curvature rational seed41 | 1966-1985 | 0.221 to 0.244 | 0.109 to 0.107 | 0.616 to 0.639 | 0.689 to 0.618 | 0.119 | 3 | 0.056 | 2 | 0.072 |
| floored curvature rational seed41 | 2010-2025 | 0.201 to 0.227 | 0.119 to 0.101 | 1.931 to 2.060 | 1.539 to 1.141 | 0.400 | 12 | 0.041 | 4 | 0.048 |
| floored departure power2 seed41 | 1951-1965 | 0.174 to 0.199 | 0.052 to 0.087 | 0.025 to 0.026 | 0.346 to 0.450 | 0.104 | 18 | 0.037 | 0 | 0.037 |
| floored departure power2 seed41 | 1966-1985 | 0.231 to 0.244 | 0.070 to 0.107 | 0.658 to 0.639 | 0.513 to 0.618 | 0.108 | 22 | 0.077 | 13 | 0.107 |
| floored departure power2 seed41 | 2010-2025 | 0.249 to 0.227 | 0.076 to 0.101 | 2.042 to 2.060 | 0.958 to 1.141 | 0.183 | 14 | 0.058 | 20 | 0.084 |
| gradual departure veit_adult20 seed41 | 1951-1965 | 0.144 to 0.187 | 0.065 to 0.068 | 0.031 to 0.034 | 0.432 to 0.502 | 0.070 | 15 | 0.044 | 0 | 0.045 |
| gradual departure veit_adult20 seed41 | 1966-1985 | 0.179 to 0.219 | 0.082 to 0.081 | 0.761 to 0.779 | 0.628 to 0.677 | 0.050 | 9 | 0.029 | 3 | 0.034 |
| gradual departure veit_adult20 seed41 | 2010-2025 | 0.194 to 0.226 | 0.090 to 0.086 | 2.260 to 2.230 | 1.257 to 1.212 | 0.069 | 3 | 0.017 | 20 | 0.018 |
| gradual departure rational_shared seed41 | 1951-1965 | 0.144 to 0.163 | 0.065 to 0.071 | 0.031 to 0.024 | 0.432 to 0.418 | 0.067 | 12 | 0.049 | 0 | 0.048 |
| gradual departure rational_shared seed41 | 1966-1985 | 0.179 to 0.185 | 0.082 to 0.104 | 0.761 to 0.652 | 0.628 to 0.649 | 0.056 | 5 | 0.058 | 8 | 0.064 |
| gradual departure rational_shared seed41 | 2010-2025 | 0.194 to 0.174 | 0.090 to 0.114 | 2.260 to 2.018 | 1.257 to 1.515 | 0.257 | 18 | 0.067 | 39 | 0.071 |
| gradual departure rational_adult20 seed41 | 1951-1965 | 0.144 to 0.203 | 0.065 to 0.075 | 0.031 to 0.023 | 0.432 to 0.459 | 0.075 | 11 | 0.073 | 0 | 0.072 |
| gradual departure rational_adult20 seed41 | 1966-1985 | 0.179 to 0.224 | 0.082 to 0.111 | 0.761 to 0.607 | 0.628 to 0.696 | 0.081 | 8 | 0.089 | 11 | 0.094 |
| gradual departure rational_adult20 seed41 | 2010-2025 | 0.194 to 0.203 | 0.090 to 0.120 | 2.260 to 1.910 | 1.257 to 1.550 | 0.293 | 22 | 0.088 | 39 | 0.096 |
| crowding power2 veit seed41 | 1951-1965 | 0.144 to 0.173 | 0.065 to 0.053 | 0.031 to 0.024 | 0.432 to 0.347 | 0.085 | 25 | 0.037 | 0 | 0.038 |
| crowding power2 veit seed41 | 1966-1985 | 0.179 to 0.229 | 0.082 to 0.071 | 0.761 to 0.653 | 0.628 to 0.511 | 0.117 | 16 | 0.030 | 6 | 0.066 |
| crowding power2 veit seed41 | 2010-2025 | 0.194 to 0.248 | 0.090 to 0.077 | 2.260 to 2.011 | 1.257 to 0.947 | 0.313 | 5 | 0.024 | 26 | 0.044 |
| crowding power2 rational_adult20 seed41 | 1951-1965 | 0.203 to 0.200 | 0.075 to 0.087 | 0.023 to 0.026 | 0.459 to 0.450 | 0.078 | 6 | 0.072 | 0 | 0.072 |
| crowding power2 rational_adult20 seed41 | 1966-1985 | 0.224 to 0.244 | 0.111 to 0.107 | 0.607 to 0.641 | 0.696 to 0.617 | 0.124 | 3 | 0.055 | 1 | 0.072 |
| crowding power2 rational_adult20 seed41 | 2010-2025 | 0.203 to 0.227 | 0.120 to 0.101 | 1.910 to 2.060 | 1.550 to 1.137 | 0.415 | 16 | 0.042 | 4 | 0.049 |
| departure under power2 crowding seed41 | 1951-1965 | 0.173 to 0.200 | 0.053 to 0.087 | 0.024 to 0.026 | 0.347 to 0.450 | 0.104 | 16 | 0.038 | 0 | 0.038 |
| departure under power2 crowding seed41 | 1966-1985 | 0.229 to 0.244 | 0.071 to 0.107 | 0.653 to 0.641 | 0.511 to 0.617 | 0.109 | 25 | 0.074 | 12 | 0.106 |
| departure under power2 crowding seed41 | 2010-2025 | 0.248 to 0.227 | 0.077 to 0.101 | 2.011 to 2.060 | 0.947 to 1.137 | 0.189 | 11 | 0.055 | 23 | 0.082 |
| shared overflow seed41 | 1951-1965 | 0.144 to 0.683 | 0.065 to 0.402 | 0.031 to 0.075 | 0.432 to 7.935 | 7.504 | 352 | 0.604 | 9 | 0.603 |
| shared overflow seed41 | 1966-1985 | 0.179 to 0.683 | 0.082 to 0.433 | 0.761 to 0.557 | 0.628 to 8.512 | 7.884 | 319 | 0.749 | 97 | 0.753 |
| shared overflow seed41 | 2010-2025 | 0.194 to 0.512 | 0.090 to 0.218 | 2.260 to 1.473 | 1.257 to 7.480 | 6.222 | 252 | 0.393 | 100 | 0.447 |
| adult_priority overflow seed41 | 1951-1965 | 0.144 to 0.726 | 0.065 to 0.416 | 0.031 to 0.049 | 0.432 to 6.842 | 6.410 | 319 | 0.609 | 6 | 0.611 |
| adult_priority overflow seed41 | 1966-1985 | 0.179 to 0.727 | 0.082 to 0.446 | 0.761 to 0.509 | 0.628 to 7.300 | 6.672 | 287 | 0.765 | 120 | 0.786 |
| adult_priority overflow seed41 | 2010-2025 | 0.194 to 0.560 | 0.090 to 0.258 | 2.260 to 0.994 | 1.257 to 6.872 | 5.615 | 210 | 0.399 | 120 | 0.452 |
| kfloor1p25 seed41 | 1951-1965 | 0.144 to 0.145 | 0.065 to 0.065 | 0.031 to 0.031 | 0.432 to 0.432 | 0.003 | 0 | 0.001 | 0 | 0.001 |
| kfloor1p25 seed41 | 1966-1985 | 0.179 to 0.180 | 0.082 to 0.081 | 0.761 to 0.763 | 0.628 to 0.629 | 0.004 | 0 | 0.007 | 0 | 0.003 |
| kfloor1p25 seed41 | 2010-2025 | 0.194 to 0.195 | 0.090 to 0.089 | 2.260 to 2.278 | 1.257 to 1.260 | 0.007 | 0 | 0.007 | 2 | 0.002 |
| kfloor4p0 seed41 | 1951-1965 | 0.144 to 0.135 | 0.065 to 0.056 | 0.031 to 0.022 | 0.432 to 0.330 | 0.124 | 35 | 0.038 | 0 | 0.037 |
| kfloor4p0 seed41 | 1966-1985 | 0.179 to 0.160 | 0.082 to 0.089 | 0.761 to 0.625 | 0.628 to 0.552 | 0.100 | 11 | 0.042 | 5 | 0.063 |
| kfloor4p0 seed41 | 2010-2025 | 0.194 to 0.163 | 0.090 to 0.108 | 2.260 to 1.918 | 1.257 to 1.349 | 0.112 | 6 | 0.047 | 33 | 0.048 |
| historical range replacement seed42 | 1951-1965 | 0.144 to 0.134 | 0.065 to 0.068 | 0.030 to 0.029 | 0.433 to 0.420 | 0.029 | 10 | 0.012 | 0 | 0.012 |
| historical range replacement seed42 | 1966-1985 | 0.179 to 0.166 | 0.082 to 0.086 | 0.762 to 0.769 | 0.629 to 0.609 | 0.036 | 5 | 0.013 | 2 | 0.017 |
| historical range replacement seed42 | 2010-2025 | 0.194 to 0.181 | 0.090 to 0.095 | 2.258 to 2.180 | 1.256 to 1.226 | 0.083 | 4 | 0.014 | 11 | 0.017 |
| dated1951-1965 range supervision seed42 | 1951-1965 | 0.134 to 0.115 | 0.068 to 0.067 | 0.029 to 0.067 | 0.420 to 0.381 | 0.050 | 17 | 0.021 | 2 | 0.023 |
| dated1951-1965 range supervision seed42 | 1966-1985 | 0.166 to 0.143 | 0.086 to 0.085 | 0.769 to 0.884 | 0.609 to 0.556 | 0.062 | 12 | 0.021 | 5 | 0.023 |
| dated1951-1965 range supervision seed42 | 2010-2025 | 0.181 to 0.161 | 0.095 to 0.098 | 2.180 to 2.233 | 1.226 to 1.211 | 0.073 | 3 | 0.015 | 21 | 0.017 |
| floor with factor_p1_rational_adult20 seed42 | 1951-1965 | 0.202 to 0.201 | 0.075 to 0.076 | 0.024 to 0.024 | 0.461 to 0.461 | 0.003 | 1 | 0.001 | 0 | 0.001 |
| floor with factor_p1_rational_adult20 seed42 | 1966-1985 | 0.223 to 0.222 | 0.110 to 0.111 | 0.610 to 0.611 | 0.694 to 0.694 | 0.003 | 0 | 0.002 | 0 | 0.002 |
| floor with factor_p1_rational_adult20 seed42 | 2010-2025 | 0.202 to 0.201 | 0.119 to 0.120 | 1.915 to 1.918 | 1.544 to 1.543 | 0.006 | 0 | 0.001 | 0 | 0.001 |
| floor with factor_p2_veit seed42 | 1951-1965 | 0.173 to 0.178 | 0.053 to 0.053 | 0.024 to 0.025 | 0.349 to 0.355 | 0.011 | 4 | 0.006 | 0 | 0.006 |
| floor with factor_p2_veit seed42 | 1966-1985 | 0.227 to 0.232 | 0.071 to 0.070 | 0.650 to 0.655 | 0.507 to 0.514 | 0.012 | 4 | 0.013 | 0 | 0.009 |
| floor with factor_p2_veit seed42 | 2010-2025 | 0.245 to 0.248 | 0.077 to 0.075 | 2.014 to 2.027 | 0.949 to 0.933 | 0.019 | 0 | 0.013 | 6 | 0.013 |
| floor with factor_p2_rational_adult20 seed42 | 1951-1965 | 0.199 to 0.202 | 0.087 to 0.086 | 0.026 to 0.026 | 0.450 to 0.450 | 0.004 | 4 | 0.003 | 0 | 0.003 |
| floor with factor_p2_rational_adult20 seed42 | 1966-1985 | 0.243 to 0.246 | 0.108 to 0.106 | 0.643 to 0.644 | 0.615 to 0.615 | 0.005 | 1 | 0.003 | 1 | 0.003 |
| floor with factor_p2_rational_adult20 seed42 | 2010-2025 | 0.226 to 0.230 | 0.102 to 0.100 | 2.058 to 2.073 | 1.124 to 1.126 | 0.007 | 0 | 0.002 | 0 | 0.003 |
| floored departure power1 seed42 | 1951-1965 | 0.146 to 0.201 | 0.064 to 0.076 | 0.032 to 0.024 | 0.432 to 0.461 | 0.072 | 13 | 0.071 | 0 | 0.071 |
| floored departure power1 seed42 | 1966-1985 | 0.181 to 0.222 | 0.080 to 0.111 | 0.779 to 0.611 | 0.629 to 0.694 | 0.079 | 8 | 0.090 | 9 | 0.093 |
| floored departure power1 seed42 | 2010-2025 | 0.195 to 0.201 | 0.087 to 0.120 | 2.307 to 1.918 | 1.246 to 1.543 | 0.297 | 19 | 0.093 | 44 | 0.098 |
| floored curvature Veit seed42 | 1951-1965 | 0.146 to 0.178 | 0.064 to 0.053 | 0.032 to 0.025 | 0.432 to 0.355 | 0.077 | 22 | 0.041 | 0 | 0.041 |
| floored curvature Veit seed42 | 1966-1985 | 0.181 to 0.232 | 0.080 to 0.070 | 0.779 to 0.655 | 0.629 to 0.514 | 0.115 | 11 | 0.032 | 6 | 0.066 |
| floored curvature Veit seed42 | 2010-2025 | 0.195 to 0.248 | 0.087 to 0.075 | 2.307 to 2.027 | 1.246 to 0.933 | 0.314 | 7 | 0.025 | 29 | 0.045 |
| floored curvature rational seed42 | 1951-1965 | 0.201 to 0.202 | 0.076 to 0.086 | 0.024 to 0.026 | 0.461 to 0.450 | 0.074 | 7 | 0.072 | 0 | 0.072 |
| floored curvature rational seed42 | 1966-1985 | 0.222 to 0.246 | 0.111 to 0.106 | 0.611 to 0.644 | 0.694 to 0.615 | 0.121 | 2 | 0.053 | 1 | 0.070 |
| floored curvature rational seed42 | 2010-2025 | 0.201 to 0.230 | 0.120 to 0.100 | 1.918 to 2.073 | 1.543 to 1.126 | 0.418 | 14 | 0.040 | 4 | 0.047 |
| floored departure power2 seed42 | 1951-1965 | 0.178 to 0.202 | 0.053 to 0.086 | 0.025 to 0.026 | 0.355 to 0.450 | 0.096 | 14 | 0.040 | 0 | 0.039 |
| floored departure power2 seed42 | 1966-1985 | 0.232 to 0.246 | 0.070 to 0.106 | 0.655 to 0.644 | 0.514 to 0.615 | 0.103 | 19 | 0.077 | 12 | 0.106 |
| floored departure power2 seed42 | 2010-2025 | 0.248 to 0.230 | 0.075 to 0.100 | 2.027 to 2.073 | 0.933 to 1.126 | 0.193 | 12 | 0.062 | 23 | 0.089 |
| gradual departure veit_adult20 seed42 | 1951-1965 | 0.144 to 0.186 | 0.065 to 0.068 | 0.031 to 0.034 | 0.432 to 0.500 | 0.067 | 15 | 0.043 | 0 | 0.044 |
| gradual departure veit_adult20 seed42 | 1966-1985 | 0.179 to 0.218 | 0.082 to 0.081 | 0.762 to 0.780 | 0.629 to 0.675 | 0.048 | 9 | 0.029 | 3 | 0.034 |
| gradual departure veit_adult20 seed42 | 2010-2025 | 0.194 to 0.226 | 0.090 to 0.086 | 2.264 to 2.227 | 1.260 to 1.217 | 0.066 | 3 | 0.016 | 22 | 0.017 |
| gradual departure rational_shared seed42 | 1951-1965 | 0.144 to 0.162 | 0.065 to 0.075 | 0.031 to 0.024 | 0.432 to 0.424 | 0.068 | 16 | 0.045 | 0 | 0.045 |
| gradual departure rational_shared seed42 | 1966-1985 | 0.179 to 0.186 | 0.082 to 0.110 | 0.762 to 0.640 | 0.629 to 0.660 | 0.067 | 7 | 0.068 | 14 | 0.078 |
| gradual departure rational_shared seed42 | 2010-2025 | 0.194 to 0.173 | 0.090 to 0.116 | 2.264 to 1.959 | 1.260 to 1.544 | 0.285 | 20 | 0.061 | 38 | 0.069 |
| gradual departure rational_adult20 seed42 | 1951-1965 | 0.144 to 0.202 | 0.065 to 0.075 | 0.031 to 0.024 | 0.432 to 0.461 | 0.072 | 11 | 0.072 | 0 | 0.071 |
| gradual departure rational_adult20 seed42 | 1966-1985 | 0.179 to 0.223 | 0.082 to 0.110 | 0.762 to 0.610 | 0.629 to 0.694 | 0.079 | 8 | 0.088 | 10 | 0.093 |
| gradual departure rational_adult20 seed42 | 2010-2025 | 0.194 to 0.202 | 0.090 to 0.119 | 2.264 to 1.915 | 1.260 to 1.544 | 0.285 | 19 | 0.087 | 41 | 0.095 |
| crowding power2 veit seed42 | 1951-1965 | 0.144 to 0.173 | 0.065 to 0.053 | 0.031 to 0.024 | 0.432 to 0.349 | 0.083 | 25 | 0.039 | 0 | 0.039 |
| crowding power2 veit seed42 | 1966-1985 | 0.179 to 0.227 | 0.082 to 0.071 | 0.762 to 0.650 | 0.629 to 0.507 | 0.121 | 13 | 0.033 | 5 | 0.066 |
| crowding power2 veit seed42 | 2010-2025 | 0.194 to 0.245 | 0.090 to 0.077 | 2.264 to 2.014 | 1.260 to 0.949 | 0.311 | 7 | 0.022 | 30 | 0.035 |
| crowding power2 rational_adult20 seed42 | 1951-1965 | 0.202 to 0.199 | 0.075 to 0.087 | 0.024 to 0.026 | 0.461 to 0.450 | 0.076 | 8 | 0.070 | 0 | 0.070 |
| crowding power2 rational_adult20 seed42 | 1966-1985 | 0.223 to 0.243 | 0.110 to 0.108 | 0.610 to 0.643 | 0.694 to 0.615 | 0.124 | 3 | 0.054 | 2 | 0.071 |
| crowding power2 rational_adult20 seed42 | 2010-2025 | 0.202 to 0.226 | 0.119 to 0.102 | 1.915 to 2.058 | 1.544 to 1.124 | 0.422 | 14 | 0.041 | 4 | 0.049 |
| departure under power2 crowding seed42 | 1951-1965 | 0.173 to 0.199 | 0.053 to 0.087 | 0.024 to 0.026 | 0.349 to 0.450 | 0.101 | 14 | 0.039 | 0 | 0.039 |
| departure under power2 crowding seed42 | 1966-1985 | 0.227 to 0.243 | 0.071 to 0.108 | 0.650 to 0.643 | 0.507 to 0.615 | 0.111 | 20 | 0.076 | 13 | 0.105 |
| departure under power2 crowding seed42 | 2010-2025 | 0.245 to 0.226 | 0.077 to 0.102 | 2.014 to 2.058 | 0.949 to 1.124 | 0.174 | 12 | 0.056 | 17 | 0.076 |
| shared overflow seed42 | 1951-1965 | 0.144 to 0.669 | 0.065 to 0.359 | 0.031 to 0.076 | 0.432 to 7.078 | 6.646 | 339 | 0.589 | 9 | 0.588 |
| shared overflow seed42 | 1966-1985 | 0.179 to 0.669 | 0.082 to 0.395 | 0.762 to 0.576 | 0.629 to 7.727 | 7.098 | 306 | 0.771 | 100 | 0.765 |
| shared overflow seed42 | 2010-2025 | 0.194 to 0.503 | 0.090 to 0.204 | 2.264 to 1.384 | 1.260 to 7.136 | 5.877 | 228 | 0.371 | 86 | 0.425 |
| adult_priority overflow seed42 | 1951-1965 | 0.144 to 0.696 | 0.065 to 0.370 | 0.031 to 0.062 | 0.432 to 6.321 | 5.889 | 332 | 0.589 | 3 | 0.584 |
| adult_priority overflow seed42 | 1966-1985 | 0.179 to 0.697 | 0.082 to 0.405 | 0.762 to 0.566 | 0.629 to 6.861 | 6.232 | 297 | 0.746 | 115 | 0.766 |
| adult_priority overflow seed42 | 2010-2025 | 0.194 to 0.535 | 0.090 to 0.235 | 2.264 to 1.188 | 1.260 to 6.604 | 5.344 | 223 | 0.392 | 106 | 0.444 |
| kfloor1p25 seed42 | 1951-1965 | 0.144 to 0.145 | 0.065 to 0.065 | 0.031 to 0.031 | 0.432 to 0.432 | 0.003 | 0 | 0.001 | 0 | 0.001 |
| kfloor1p25 seed42 | 1966-1985 | 0.179 to 0.180 | 0.082 to 0.081 | 0.762 to 0.763 | 0.629 to 0.629 | 0.004 | 0 | 0.007 | 0 | 0.003 |
| kfloor1p25 seed42 | 2010-2025 | 0.194 to 0.195 | 0.090 to 0.089 | 2.264 to 2.277 | 1.260 to 1.261 | 0.007 | 0 | 0.006 | 2 | 0.002 |
| kfloor4p0 seed42 | 1951-1965 | 0.144 to 0.146 | 0.065 to 0.064 | 0.031 to 0.032 | 0.432 to 0.432 | 0.005 | 1 | 0.002 | 0 | 0.002 |
| kfloor4p0 seed42 | 1966-1985 | 0.179 to 0.181 | 0.082 to 0.080 | 0.762 to 0.779 | 0.629 to 0.629 | 0.006 | 0 | 0.009 | 1 | 0.004 |
| kfloor4p0 seed42 | 2010-2025 | 0.194 to 0.195 | 0.090 to 0.087 | 2.264 to 2.307 | 1.260 to 1.246 | 0.016 | 0 | 0.008 | 3 | 0.005 |

## All model versions versus common reference: modern East, all land

| Fit | Potential lambda means | Lambda mean absolute change | Lambda crossing cells | C mean absolute change | C crossing cells | Local replacement mean absolute change |
|---|---|---:|---:|---:|---:|---:|
| step29_yb_native | 1.239 to 2.119 | 0.890 | 15 | 0.051 | 35 | 0.055 |
| step29_yb_native_s42 | 1.239 to 2.110 | 0.881 | 15 | 0.050 | 34 | 0.055 |
| step31_nest | 1.239 to 1.794 | 0.565 | 12 | 0.054 | 24 | 0.057 |
| step31_nest_bloose | 1.239 to 1.923 | 0.689 | 20 | 0.059 | 27 | 0.062 |
| step31_nestonly | 1.239 to 1.806 | 0.576 | 15 | 0.058 | 26 | 0.060 |
| step32_cbc_yb | 1.239 to 1.906 | 0.672 | 18 | 0.101 | 28 | 0.093 |
| step32_cbchi_yb | 1.239 to 1.653 | 0.430 | 5 | 0.095 | 29 | 0.090 |
| step32_cbclo_yb | 1.239 to 1.878 | 0.645 | 18 | 0.104 | 27 | 0.095 |
| step33_nestcbc_natdep_s41 | 1.239 to 1.546 | 0.326 | 11 | 0.075 | 29 | 0.073 |
| step33_nestcbc_natdep_s42 | 1.239 to 1.538 | 0.321 | 11 | 0.074 | 27 | 0.072 |
| step33_nestcbc_s41 | 1.239 to 1.792 | 0.564 | 12 | 0.054 | 24 | 0.058 |
| step33_nestcbc_s42 | 1.239 to 1.787 | 0.559 | 12 | 0.054 | 24 | 0.057 |
| step34_abs_broodhi | 1.239 to 1.376 | 0.151 | 8 | 0.044 | 24 | 0.042 |
| step34_abs_broodlo | 1.239 to 1.250 | 0.041 | 2 | 0.019 | 28 | 0.025 |
| step34_abs_comp_natdep_s41 | 1.239 to 1.463 | 0.237 | 27 | 0.078 | 25 | 0.097 |
| step34_abs_comp_natdep_s42 | 1.239 to 1.281 | 0.085 | 7 | 0.041 | 25 | 0.046 |
| step34_abs_comp_s42 | 1.239 to 1.306 | 0.067 | 3 | 0.010 | 5 | 0.012 |
| step34_abs_foundhi | 1.239 to 1.250 | 0.039 | 1 | 0.013 | 15 | 0.014 |
| step34_abs_foundlo | 1.239 to 1.243 | 0.011 | 2 | 0.003 | 3 | 0.004 |
| step34_abs_hazard_natdep_s41 | 1.239 to 1.451 | 0.225 | 23 | 0.073 | 28 | 0.091 |
| step34_abs_hazard_natdep_s42 | 1.239 to 1.264 | 0.084 | 7 | 0.041 | 26 | 0.047 |
| step34_abs_hazard_s41 | 1.239 to 1.263 | 0.025 | 1 | 0.003 | 2 | 0.004 |
| step34_abs_hazard_s42 | 1.239 to 1.255 | 0.020 | 2 | 0.003 | 1 | 0.004 |
| step34_abs_natdep_s41 | 1.239 to 1.342 | 0.133 | 12 | 0.054 | 27 | 0.061 |
| step34_abs_natdep_s42 | 1.239 to 1.394 | 0.194 | 25 | 0.078 | 32 | 0.095 |
| step34_abs_s42 | 1.239 to 1.249 | 0.015 | 2 | 0.004 | 3 | 0.004 |
| step34_abs_v2_hazard_s41 | 1.239 to 1.172 | 0.091 | 5 | 0.024 | 19 | 0.023 |
| step34_abs_v2_hazard_s42 | 1.239 to 1.164 | 0.099 | 5 | 0.017 | 19 | 0.020 |
| step34_abs_v2_natdep_s41 | 1.239 to 1.238 | 0.095 | 7 | 0.057 | 26 | 0.052 |
| step34_abs_v2_natdep_s42 | 1.239 to 1.188 | 0.101 | 7 | 0.039 | 22 | 0.037 |
| step34_abs_v2_s41 | 1.239 to 1.163 | 0.093 | 5 | 0.023 | 19 | 0.023 |
| step34_abs_v2_s42 | 1.239 to 1.158 | 0.100 | 5 | 0.016 | 22 | 0.020 |
| step34_broodhi | 1.239 to 1.742 | 0.521 | 12 | 0.052 | 27 | 0.052 |
| step34_broodlo | 1.239 to 1.821 | 0.591 | 13 | 0.056 | 29 | 0.063 |
| step34_foundhi | 1.239 to 1.915 | 0.686 | 19 | 0.058 | 23 | 0.061 |
| step34_foundlo | 1.239 to 1.789 | 0.561 | 13 | 0.054 | 24 | 0.058 |
| step34_rel_comp_natdep_s41 | 1.239 to 1.576 | 0.354 | 11 | 0.074 | 30 | 0.073 |
| step34_rel_comp_natdep_s42 | 1.239 to 1.608 | 0.378 | 16 | 0.085 | 33 | 0.085 |
| step34_rel_comp_s41 | 1.239 to 1.803 | 0.573 | 16 | 0.056 | 24 | 0.059 |
| step34_rel_hazard_natdep_s41 | 1.239 to 1.579 | 0.357 | 11 | 0.074 | 31 | 0.073 |
| step34_rel_hazard_natdep_s42 | 1.239 to 1.571 | 0.349 | 11 | 0.076 | 31 | 0.074 |
| step34_rel_hazard_s41 | 1.239 to 1.804 | 0.575 | 15 | 0.055 | 24 | 0.058 |
| step34_rel_hazard_s42 | 1.239 to 1.798 | 0.569 | 12 | 0.055 | 24 | 0.058 |
| step34_rel_v2_hazard_s41 | 1.239 to 1.699 | 0.473 | 11 | 0.064 | 25 | 0.061 |
| step34_rel_v2_hazard_s42 | 1.239 to 1.675 | 0.453 | 12 | 0.055 | 24 | 0.054 |
| step34_rel_v2_natdep_s41 | 1.239 to 1.451 | 0.238 | 10 | 0.082 | 26 | 0.082 |
| step34_rel_v2_natdep_s42 | 1.239 to 1.506 | 0.287 | 11 | 0.082 | 23 | 0.080 |
| step34_rel_v2_s41 | 1.239 to 1.701 | 0.474 | 11 | 0.064 | 25 | 0.061 |
| step35_est1951_ctl_hi_s41 | 1.239 to 1.528 | 0.304 | 12 | 0.052 | 35 | 0.057 |
| step35_est1951_ctl_lo_s41 | 1.239 to 1.408 | 0.177 | 7 | 0.046 | 29 | 0.049 |
| step35_est1951_ctl_mid_s41 | 1.239 to 1.430 | 0.203 | 5 | 0.044 | 32 | 0.049 |
| step35_est1951_ctl_mid_s42 | 1.239 to 1.398 | 0.175 | 3 | 0.044 | 31 | 0.047 |
| step35_est1951_nat_hi_s41 | 1.239 to 1.467 | 0.244 | 16 | 0.042 | 32 | 0.040 |
| step35_est1951_nat_lo_s41 | 1.239 to 1.299 | 0.101 | 4 | 0.050 | 26 | 0.046 |
| step35_est1951_nat_mid_s41 | 1.239 to 1.435 | 0.224 | 6 | 0.035 | 31 | 0.032 |
| step35_est1951_nat_mid_s42 | 1.239 to 1.509 | 0.290 | 13 | 0.039 | 27 | 0.036 |
| step36_est1951_nonest_ctl_s41 | 1.239 to 1.805 | 0.582 | 15 | 0.035 | 31 | 0.032 |
| step36_nonest_ctl_s41 | 1.239 to 1.533 | 0.298 | 10 | 0.037 | 21 | 0.030 |
| step36_nonest_ctl_s42 | 1.239 to 1.521 | 0.287 | 9 | 0.037 | 21 | 0.031 |
| step36_nonest_nat_s41 | 1.239 to 1.528 | 0.295 | 19 | 0.163 | 35 | 0.168 |
| step36_nonest_nat_s42 | 1.239 to 1.435 | 0.206 | 8 | 0.108 | 26 | 0.099 |
| step38_cbc1951_broods_free_s41 | 1.239 to 2.118 | 1.001 | 88 | 0.112 | 56 | 0.120 |
| step38_cbc1951_circle2_free_s41 | 1.239 to 2.118 | 1.030 | 95 | 0.121 | 66 | 0.135 |
| step38_cbc1951_free_s41 | 1.239 to 2.117 | 1.017 | 88 | 0.113 | 64 | 0.120 |
| step38_cbc1951_free_s42 | 1.239 to 2.212 | 1.082 | 90 | 0.122 | 54 | 0.127 |
| step38_cbc1951_lo_s41 | 1.239 to 2.207 | 1.070 | 89 | 0.113 | 57 | 0.120 |
| step38_cbc1951_mid_s41 | 1.239 to 2.321 | 1.148 | 79 | 0.095 | 54 | 0.096 |
| step38_cbc1951_mid_s42 | 1.239 to 2.278 | 1.134 | 87 | 0.104 | 54 | 0.110 |
| step38_cbc1951_noncbc_free_s41 | 1.239 to 1.258 | 0.019 | 1 | 0.001 | 1 | 0.001 |
| step38_cbc1951_noncbc_free_s42 | 1.239 to 1.256 | 0.022 | 2 | 0.004 | 3 | 0.004 |
| step39_adult_priority_fixed_s41 | 1.239 to 6.872 | 5.633 | 208 | 0.400 | 119 | 0.452 |
| step39_adult_priority_fixed_s42 | 1.239 to 6.604 | 5.365 | 221 | 0.393 | 105 | 0.444 |
| step39_kfloor1p25_s41 | 1.239 to 1.260 | 0.026 | 2 | 0.007 | 1 | 0.004 |
| step39_kfloor1p25_s42 | 1.239 to 1.261 | 0.027 | 2 | 0.008 | 1 | 0.004 |
| step39_kfloor4p0_s41 | 1.239 to 1.349 | 0.126 | 4 | 0.045 | 30 | 0.046 |
| step39_kfloor4p0_s42 | 1.239 to 1.246 | 0.021 | 2 | 0.009 | 6 | 0.005 |
| step39_shared_fixed_s41 | 1.239 to 7.480 | 6.241 | 250 | 0.394 | 103 | 0.447 |
| step39_shared_fixed_s42 | 1.239 to 7.136 | 5.897 | 226 | 0.372 | 87 | 0.425 |
| step39_veit_fixed_s41 | 1.239 to 1.257 | 0.023 | 2 | 0.003 | 3 | 0.003 |
| step39_veit_fixed_s42 | 1.239 to 1.260 | 0.025 | 2 | 0.003 | 3 | 0.004 |
| step40_rangev2_historical_s41 | 1.239 to 1.222 | 0.087 | 6 | 0.015 | 8 | 0.018 |
| step40_rangev2_historical_s42 | 1.239 to 1.226 | 0.091 | 6 | 0.016 | 12 | 0.018 |
| step40_rangev2_invasion_s41 | 1.239 to 1.205 | 0.134 | 9 | 0.018 | 18 | 0.024 |
| step40_rangev2_invasion_s42 | 1.239 to 1.211 | 0.134 | 9 | 0.018 | 17 | 0.023 |
| step41_rational_adult20_s41 | 1.239 to 1.550 | 0.311 | 20 | 0.087 | 36 | 0.096 |
| step41_rational_adult20_s42 | 1.239 to 1.544 | 0.305 | 17 | 0.085 | 38 | 0.094 |
| step41_rational_shared_s41 | 1.239 to 1.515 | 0.276 | 16 | 0.065 | 36 | 0.071 |
| step41_rational_shared_s42 | 1.239 to 1.544 | 0.305 | 18 | 0.060 | 35 | 0.068 |
| step41_veit_adult20_s41 | 1.239 to 1.212 | 0.053 | 1 | 0.017 | 23 | 0.019 |
| step41_veit_adult20_s42 | 1.239 to 1.217 | 0.049 | 1 | 0.017 | 25 | 0.018 |
| step42_power2_rational_adult20_s41 | 1.239 to 1.137 | 0.215 | 4 | 0.062 | 40 | 0.055 |
| step42_power2_rational_adult20_s42 | 1.239 to 1.124 | 0.221 | 3 | 0.063 | 40 | 0.054 |
| step42_power2_veit_s41 | 1.239 to 0.947 | 0.294 | 7 | 0.022 | 23 | 0.043 |
| step42_power2_veit_s42 | 1.239 to 0.949 | 0.290 | 9 | 0.021 | 27 | 0.035 |
| step43_factor_p1_rational_adult20_s41 | 1.239 to 1.539 | 0.300 | 17 | 0.086 | 37 | 0.094 |
| step43_factor_p1_rational_adult20_s42 | 1.239 to 1.543 | 0.304 | 17 | 0.085 | 38 | 0.094 |
| step43_factor_p2_rational_adult20_s41 | 1.239 to 1.141 | 0.213 | 5 | 0.063 | 41 | 0.055 |
| step43_factor_p2_rational_adult20_s42 | 1.239 to 1.126 | 0.220 | 3 | 0.063 | 40 | 0.055 |
| step43_factor_p2_veit_s41 | 1.239 to 0.958 | 0.284 | 9 | 0.028 | 25 | 0.045 |
| step43_factor_p2_veit_s42 | 1.239 to 0.933 | 0.306 | 9 | 0.028 | 23 | 0.047 |

## Registered new comparisons across the complete regional belt

The belt exhaustively partitions land at 25–49N: East east of 95W, Plains 100–95W, West west of 100W. These rows use fixed observed cells and report the same measures in every region. Biological C excludes abundance innovations; fitted C includes them. Legacy regional summaries above retain their original masks.

| Contrast | Era | Region | Cells | Mean absolute biological lambda change | Mean absolute full-process lambda change | Mean absolute biological C change | Mean absolute fitted C change | Fitted C reversals |
|---|---|---|---:|---:|---:|---:|---:|---:|

Full JSON includes every contrast, three eras, seven regional supports and both observation supports, including correlations, continuous differences, crossing counts with exact/.01/.05 thresholds and finite-support coverage. No pair is declared stable solely because a correlation is high.

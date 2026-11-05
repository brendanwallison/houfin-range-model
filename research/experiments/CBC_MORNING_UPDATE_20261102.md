# CBC / later-state overnight update — November 2, 2026

Six optimizations completed and passed refinement checks: four joint CBC fits and two matched no-CBC controls. A post-fit execution error hid the GPU before a GPU-only diagnostic. The checkpoints and model fits were intact. CPU support was added to the external readout, the wrappers were corrected, and the six missing diagnostic suites were recovered locally without optimization. Duplicate watcher instances were also removed; collectors now use a lock and atomic status writes. Original logs/failed queue receipts remain as provenance.

**Interpretation:** CBC materially changes the inferred state/demography, but these first joint fits are not improvements across the full objective. CBC means are closer to observed levels than in the initial fixed-population calibration, while BBS scores degrade, adult survival falls, juvenile survival approaches its ceiling, fecundity demand rises and fitted invasion arrives earlier. Treat this as a reproducible observation/process conflict to resolve, not a reason to discard CBC or adopt a joint fit merely because it fits CBC better. The seasonal and reproductive-conversion sensitivities remain important.

All six use 1951 initialization, no release-cell Allee exemption, no old CBC backcast pin and no native departure. Initial multipliers are relative to the model reference density. Nominal bird-equivalents use the earlier conditional conversion convention, not an independently calibrated census or uncertainty interval. Free-abundance seed estimates are not posterior uncertainty.

| Fit | Initial multiplier | Nominal bird-equivalent | Adult survival | Juvenile/adult | Population near juvenile ceiling | Fmax | Fitted arrival error | Frozen-release arrival error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step38_cbc1951_free_s41 | 1.910 | 196.3 | 0.278 | 0.9994 | 76.4% | 11.82 | -6.0 | -31.0 |
| step38_cbc1951_mid_s41 | 5.447 | 560.0 | 0.297 | 0.9999 | 81.6% | 12.28 | -6.0 | -32.0 |
| step38_cbc1951_free_s42 | 1.507 | 154.9 | 0.286 | 0.9997 | 78.8% | 11.82 | -6.0 | -31.0 |
| step38_cbc1951_mid_s42 | 5.447 | 560.0 | 0.289 | 0.9997 | 79.6% | 11.96 | -6.0 | -32.0 |
| step38_cbc1951_noncbc_free_s41 | 1.007 | 103.5 | 0.562 | 0.4766 | 0.0% | 8.31 | -3.0 | -26.0 |
| step38_cbc1951_noncbc_free_s42 | 0.957 | 98.4 | 0.566 | 0.4736 | 0.0% | 8.37 | -3.0 | -25.0 |

Survival/Fmax are modern model-population-weighted medians. Ceiling means Sj/Sa>.95. Arrival error is model minus observation in years; negative is early. These fitted-state summaries do not establish vital-rate identification.

## CBC supervision and prediction

| Fit | Spatial observed/predicted mean | Temporal observed/predicted mean | Temporal observed/predicted detection | Historical training observed/predicted mean |
|---|---|---|---|---|
| step38_cbc1951_free_s41 | 147.6 / 139.5 | 122.8 / 147.0 | 85.4% / 88.5% | 81.1 / 57.1 |
| step38_cbc1951_mid_s41 | 147.6 / 143.8 | 122.8 / 146.9 | 85.4% / 88.5% | 81.1 / 56.6 |
| step38_cbc1951_free_s42 | 147.6 / 126.6 | 122.8 / 147.0 | 85.4% / 88.5% | 81.1 / 56.8 |
| step38_cbc1951_mid_s42 | 147.6 / 133.6 | 122.8 / 148.1 | 85.4% / 88.5% | 81.1 / 56.4 |

Unseen circle effects are integrated; other parameters are conditional on MAP. CBC spatial withholding is not independent BBS spatial transfer. Historical training counts above cover all regions, not just the founder cell. Aggregate mean calibration does not establish correct historical spatial allocation.

## Matched BBS prediction

| Fit | Temporal marginal change (90% interval) | Spatial marginal change (90% interval) |
|---|---|---|
| step38_cbc1951_free_s41 | -0.0886 [-0.1184, -0.0635] | -0.1192 [-0.2057, -0.0365] |
| step38_cbc1951_mid_s41 | -0.0805 [-0.1057, -0.0586] | -0.0999 [-0.1767, -0.0208] |
| step38_cbc1951_free_s42 | -0.0849 [-0.1133, -0.0617] | -0.1144 [-0.2010, -0.0292] |
| step38_cbc1951_mid_s42 | -0.0822 [-0.1083, -0.0598] | -0.1041 [-0.1826, -0.0273] |
| step38_cbc1951_noncbc_free_s41 | -0.0023 [-0.0054, 0.0008] | 0.0022 [-0.0041, 0.0110] |
| step38_cbc1951_noncbc_free_s42 | -0.0020 [-0.0053, 0.0012] | 0.0021 [-0.0047, 0.0118] |

Common reference: no-CBC 1951 fixed-middle control. These exploratory block intervals condition on fitted points; they are not demographic credible intervals.

## East, Great Plains, West — contribution and establishment mechanism

Every regional row here uses equal land-cell weights and latitude 25–49N. East >=80W; primary Plains 100–95W; West <100W. C<.95 measures a greater-than-5% realized contribution deficit. K/Allee-dead selects positive potential growth but no positive equilibrium; it is not all unsuitable habitat.

| Fit | Region | Mean C | Land share C<.95 | K/Allee-dead land share | Median K |
|---|---|---:|---:|---:|---:|
| step38_cbc1951_free_s41 | East | 0.570 | 79.7% | 0.2% | 0.12917 |
| step38_cbc1951_free_s41 | Great Plains | 0.660 | 79.7% | 0.3% | 0.19299 |
| step38_cbc1951_free_s41 | West | 0.750 | 65.1% | 0.3% | 0.13356 |
| step38_cbc1951_mid_s41 | East | 0.555 | 77.9% | 0.0% | 0.13253 |
| step38_cbc1951_mid_s41 | Great Plains | 0.669 | 77.1% | 0.0% | 0.20306 |
| step38_cbc1951_mid_s41 | West | 0.757 | 64.6% | 0.1% | 0.14450 |
| step38_cbc1951_free_s42 | East | 0.578 | 78.3% | 0.2% | 0.13244 |
| step38_cbc1951_free_s42 | Great Plains | 0.676 | 78.4% | 0.0% | 0.19495 |
| step38_cbc1951_free_s42 | West | 0.755 | 64.9% | 0.1% | 0.13192 |
| step38_cbc1951_mid_s42 | East | 0.564 | 78.3% | 0.2% | 0.13073 |
| step38_cbc1951_mid_s42 | Great Plains | 0.679 | 78.7% | 0.3% | 0.20604 |
| step38_cbc1951_mid_s42 | West | 0.763 | 63.2% | 0.1% | 0.14016 |
| step38_cbc1951_noncbc_free_s41 | East | 0.521 | 62.1% | 0.9% | 0.01260 |
| step38_cbc1951_noncbc_free_s41 | Great Plains | 0.674 | 73.9% | 1.8% | 0.02746 |
| step38_cbc1951_noncbc_free_s41 | West | 0.755 | 60.1% | 1.6% | 0.05033 |
| step38_cbc1951_noncbc_free_s42 | East | 0.522 | 61.6% | 0.5% | 0.01168 |
| step38_cbc1951_noncbc_free_s42 | Great Plains | 0.677 | 73.7% | 2.4% | 0.02679 |
| step38_cbc1951_noncbc_free_s42 | West | 0.759 | 59.7% | 2.0% | 0.05060 |

## Matched aggregate-process changes: CBC versus no CBC

Same free-initial-abundance priors, matched seeds and modern equal-land support as the regional table. Similar regional mean C does not imply stable contribution geography. These comparisons show changes in fitted-state processes, not posterior identification.

| Seed | Region | Mean absolute C change | C map correlation | Direct reversals beyond .95/1.05 |
|---|---|---:|---:|---:|
| 41 | East | 0.113 | 0.932 | 6 |
| 41 | Great Plains | 0.137 | 0.821 | 10 |
| 41 | West | 0.111 | 0.893 | 40 |
| 42 | East | 0.123 | 0.914 | 5 |
| 42 | Great Plains | 0.139 | 0.808 | 8 |
| 42 | West | 0.112 | 0.887 | 44 |

The broader [cross-model audit](CROSS_MODEL_PROCESS_AUDIT_20261101.md) now includes the matched CBC/no-CBC and free/fixed-abundance contrasts, with local replacement, potential lambda, component rates and alternative spatial support.

## Full diagnostic coverage and remaining interpretation

| Concern | Evidence/status |
|---|---|
| Historical growth and timing (P1–P2) | Regional early/modern potential-growth values and fitted/frozen fronts are retained in JSON. Later initialization does not clear the historical opening or timing issues. |
| Reproduction/age balance (P3–P5) | Fmax, nest likelihood readouts, adult survival, juvenile ceiling and matched seed/abundance contrasts included. Fixed annual conversion is retained in these first fits; uncertain-conversion runs remain queued. |
| Source/sink, tiny K and establishment (P6–P8) | Actual contribution, retention/export, BIDE closure, isolation, effective survival, K and positive-growth/Allee-dead masks recovered. Regional rows above avoid population weights hiding deficient land cells. Finite-propagule results are in each movement JSON; deterministic tests do not identify stochastic establishment. |
| Initialization (P9) | Free and fixed-middle comparisons complete; low profile running and high profile queued at the initial morning check. No initial-abundance confidence interval established. |
| Distances, redistribution and movement loss (P10–P12) | Permanent-distance/tail and finite-propagule followups run automatically after source download/scoring. Full results or explicit pending status are in JSON. Distances are conditional on successful movers; seasonal CBC redistribution is separate. |
| Projection flexibility, scale and corrections (P13–P15) | Same shared ecological field and structural source; new source-specific CBC nuisances are recorded. No claim that scale confounding or posterior geometry is solved. HMC feasibility of the new joint model is not tested. |

The low-abundance fit was running at the morning check. At 06:41 CST, the already-staged uncertain annual NestWatch conversion, 600-km winter mapping and broader circle heterogeneity tests were moved ahead of further abundance profiles and seed repeats, because the first four CBC fits reproduce the same compensation. Sources/checkpoints were unchanged. No new native-departure branch was added. The remaining A100 allocation had 23h38m at 06:25 CST; no extra allocation was requested. See [live queue status](CBC_ESTABLISHED_STATUS_20261101.md).

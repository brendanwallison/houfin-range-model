# Morning update — 2026-11-01, 05:33 CST

**Scope correction:** this note reported selected results, not a completed evaluation of the full pathology checklist. The [full stored-run evidence audit](FULL_PATHOLOGY_AUDIT_20261101.md) now compares all **63 accepted real-data GRID54 runs** and marks coverage for P1–P15 explicitly. Only 31 have detailed biological replays, 19 have front replays, and four have fixed-point precision replays. All 63 now have parameter summaries and positive-survey-record K/Allee context. Earlier non-GRID54 archives and several mechanistic diagnostics remain unaudited here. The candidate comments below are provisional and do not establish an overall pathology winner.

Live TACC check at 11:32 UTC: 13 second-wave attempts completed successfully, two running, nine pending. All five first-wave configurations also completed and have paired held-out scores. Both allocated A100 workers are busy; the third allocation remains pending. The controller, TACC monitor and local scorer are active. Of the 36 additional GPU-hour budget, about 13.91 hours are measured completed runtime and 22 hours are reservations for the eleven unfinished staged attempts. Further arms can stage as reservations clear, until 10:00 CST. Compounding remains held for the documented feature-equivalence failure.

## Main biological comparison

Modern model-population-weighted medians; Sj is resident/stayer survival. Western sink share is weighted by fitted population, not an independent observation-based estimate.

| Configuration | Adult survival | Juvenile/adult survival ratio | Western population in sinks |
|---|---:|---:|---:|
| CBC + NestWatch, relative Allee | .399–.400 | .995 | 8.5% |
| Same data, absolute Allee, seeds 41/42 | .560–.570 | .465–.479 | 10.3–10.4% |
| Absolute Allee + native departure, seeds 41/42 | .555–.577 | .437–.479 | 1.6–2.2% |

The absolute-law change is the strongest simple demographic improvement so far: it removes ceiling saturation in both seeds and reduces the inferred early-to-modern eastern growth contrast, with little held-out score change. Tiny Plains K persists and is lower than under relative Allee. These remain candidate tradeoffs, not an adopted fit.

Native departure adds one scalar and sharply reduces western immigration dependence. Paired spatial-score gains versus the absolute-law parent are +.0164 [+.0030,+.0286] and +.0096 [-.0115,+.0296] (90% intervals); uncertainty and basin differences remain. Realized annual fecundity at nests rises from about 3.57 to 3.80–4.00. This configuration merits close review as a parsimonious candidate.

Distance hazard alone gives h about .0031–.0032 and negligible spatial-score gain, with demographic values close to the absolute-law parent. Combining hazard with native departure gives substantially different h estimates across seeds (.0195 versus .0017) and different fecundity. Do not interpret a unique dispersal-mortality estimate from these fits.

## Auxiliary-data sensitivity

Under absolute Allee, changing fixed attempts per female from 2.5 to 2.0 gives adult survival .534 and juvenile/adult ratio .641. Changing to 3.125 instead gives adult survival .379, ratio .980, and 68.3% of modeled population above the .95 ratio threshold. Paired spatial-score changes remain small (+.0050 and -.0015, respectively, both intervals include zero). This is direct real-data evidence that uncertain nesting assumptions can restore a biological pathology while counts retain similar predictive performance.

DESK v2 fits are accepted with small positive point-score changes, but differ across seeds: adult survival .515/.566 and ratio .582/.480. Movement × v2 comparisons are still outstanding.

## Remaining concern

Fitted-front median timing at the existing 10% threshold stays about three years early across these candidates. Frozen-habitat release-only replay remains about 36–38 years early. Absolute Allee and native departure improve particular demographic pathologies without establishing that the underlying expansion mechanism is resolved. Compare these diagnostic timings within their stated protocol; do not pool them with earlier experiment thresholds.

Sources: live GRID54 run-local pathology.json/front_speed.json, accepted refinement reports, queue receipts, second_wave_ledger.json, and local verified rescore_step*.json. The continuously updated detailed tables remain in OVERNIGHT_RESULTS_20261031.md.

## Tradeoffs revealed by the wider audit

These are shares of positive modern survey records, not shares weighted by fitted abundance. K-low uses the descriptive threshold K < .05 route-count units; it is not an adoption gate. Allee-dead means positive post-establishment growth but no positive equilibrium in the cached diagnostic; immigration and averaging matter, so observed records there are a flag for investigation rather than a contradiction by themselves.

| Configuration | Whole-domain positive records with very low K | Whole-domain positive records Allee-dead | Western positive records in fundamental sinks |
|---|---:|---:|---:|
| Relative Allee, seed 41 | 0.165% | 0% | See full run audit |
| Absolute Allee, seed 41 | 1.325% | 1.012% | 23.97% |
| Absolute Allee + native departure, seed 41 | 0.410% | 1.854% | 7.03% |
| Absolute Allee + native departure, seed 42 | 0.540% | 4.584% | 3.07% |

This strengthens the evidence that native departure reduces the western sink issue beyond changing model-population weights. It also exposes an omitted tradeoff: more observed-positive records fall in Allee-dead cells, especially in the second native-departure seed. The absolute law's survival improvement therefore cannot be presented as an across-the-board biological improvement. Tiny Plains K must likewise be assessed at observations and geographically, not judged from its minimum or regional lower tail alone.


The integrated mechanism hypotheses, candidate solutions and experiment-validity qualifications are now collected in [UNIFIED_SYNTHESIS_20261101.md](UNIFIED_SYNTHESIS_20261101.md), including the audited synthetic-centering mismatch. Use that synthesis together with the full run audit when interpreting the candidate highlights above.

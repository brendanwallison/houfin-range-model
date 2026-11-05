# Current constraint guards — checked 2026-10-31 around 20:55 CDT

Source: fresh local mirror of the persistent TACC monitor (`checked_utc` approximately 2026-11-01 01:55 UTC), accepted refinement reports/scalars, the existing step31/step30 gates, and completed automatic founder rescoring. This applies existing rules; it is not a new adoption decision or a full independent verification round.

## NestWatch: better hazard recovery, but the full gates FAIL

All synthetic controls/bias arms below have accepted refinements and completed diagnostics. The planted journey-distance hazard is .03; null truth is zero.

| Arm | Fitted h | Interpretation against existing rule |
|---|---:|---|
| No nests, null | .00238 | Small hazard |
| No nests, recovery | .00773 | Fails parameter recovery [.015,.045] |
| Nests, null | .00813 | h passes <=.01, but **full G1 fails** |
| Nests, recovery; attempts pinned | .01511 | Barely inside h range, but **full G2 fails** |
| Nests, recovery; attempts free | .00812 | Fails h range |
| Nests, recovery; attempts +25% wrong | .02049 | Sensitivity arm; a better h alone is not a pass |
| Suburban-output bias | .00354 | h passes its threshold |
| Failed nests underreported | .01068 | Report-only bias arm, not a pre-registered binary h gate |

G1 compares against step29_yb_native truth: founders log -.84786 -> -1.56254 (difference -.71467, beyond +/-.5); alpha_j 1.05342 -> 1.45637 (difference +.40295, beyond +/-.25). Thus the null does not retain the planted demographics even though its hazard estimate is small.

G2 compares fecundity to nestnull: softplus(alpha_f) is approximately 7.503 -> 6.609, a decrease of about 11.9%, outside the allowed 10%. Founders change little between these two fits. Consequently the nest block **has not passed as an identification device**. The improved h point estimate must not be described as successful recovery with the compensators controlled.

Real-data nest, loose-attempts and nest-only arms are still running, around 328–338/400 MAP steps at the snapshot. Their biological interpretation remains gated by the failed G1/G2 rules. Do not cancel or erase the scheduled diagnostic fits, but do not use their scores to promote the block as a validated constraint.

Next discriminating work is to separate an optimizer-basin/recovery problem from weak identification: evaluate the planted truth and accepted synthetic optima under exactly the same likelihood, and test matched initialization/seed robustness before declaring structural non-identification or changing gates. More constraints alone would obscure that distinction.

## Compounding: the current model largely absorbs the planted effect

All compounding synthetic fits are now numerically accepted. Planted lambda .25 recovers as .14013, within its parameter interval [.125,.375], but the current-model comparator is only **5.61 objective units worse** (U 131899.149 vs 131893.540), inside the pre-registered 10-unit threshold. Paired held-out comparison is still needed for the full model-comparison rule; this objective difference alone is not marginal model evidence.

The fecundity cross-leak gives lambda .00846 (passes <=.05), but its fitted fecundity is approximately 8.794 against planted 9.94, about 11.5% low: outside the 10% fecundity condition. Hence the full cross-leak gate fails. A small real-data compounding estimate remains uninformative about biological absence. Realism-based structural choices remain separate from empirical identification.

## Founders: all brackets finished and automatically scored

Paired held-out real-BBS spatial mean log-score differences, each versus its matched base, with 90% block-bootstrap intervals:

- Central CBC pin, yearling: +.0041 [-.0072,+.0147].
- Low bracket, yearling: +.0038 [-.0073,+.0144].
- High bracket, yearling: -.0019 [-.0166,+.0080].
- Central CBC pin, delayed/native: -.0132 [-.0236,-.0039].

All yearling bracket point scores are inside the pre-registered +/-.005 floor, but this does not establish interval-based equivalence. Demographic sensitivity across the brackets remains material: low vs high alpha_f 6.757 -> 7.595 and alpha_j 3.263 -> .618. Predictive stability therefore does not imply demographic identification.

## Execution / monitoring

The local 3090 completed both float32 bias-guard MAP stages and verified transfers at 18:42:53 CDT. TACC completed their float64 refinements; the reserved bundle returned successfully and `local_running` is empty. The local GPU task is now finished, not stalled.

Two A100 workers remain allocated; the third request is pending. At the snapshot, pending task files are empty and only the real NestWatch bundle 0405 is running. The monitor and follow-up dispatcher remain live, with fresh heartbeats and no follow-up errors. The seasonal branch remains a planning document only.

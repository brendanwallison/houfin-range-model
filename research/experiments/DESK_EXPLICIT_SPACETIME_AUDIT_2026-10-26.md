# Explicit space/time inputs and changing-niche experiments

Audited experiment records, current DESK architecture and its available git history, and origin/research/age-model at e714453f14780611b5e96dfb67b7ed8f9677ba30. Negative findings mean no confirmed implemented-and-tested track was found in these sources; they do not prove no uncommitted prototype ever existed.

## Encoder versus downstream model

The current DESK encoder consumes covariate grids and masks, with an optional masked spatial residual from neighboring covariate codes. It has covariate/output-history machinery. This is richer than an isolated instantaneous covariate vector, but is not a calendar-time-conditioned ecological map. The audited architecture has no explicit year/time embedding or time-varying network weights. Historical records explicitly describe DESK as lacking a year input. Changes in fertilizer, irrigation, secondary-land age and other covariates can act as implicit clocks. E041 tested removing those clocks; it did not add a year input.

The available model_arch.py history includes multi-stream generalization, spatial convolution, coordinate-basis provenance and private seasonal heads, but no confirmed direct space/time conditioning experiment. Important distinction: ESK/DESK being supervised across many cell-years, or writing a spacetime cube, does not mean year and coordinates are network input features.

## Confirmed explicit temporal tracks

| Track | What varied | Actual status / result |
|---|---|---|
| DESK E016/E033/E039 and fair rivals | Separate species/community predictors with smooth position features × calendar time; combinations with DESK | Run. These trend predictors matched or beat some temporal readouts. In E033c, an adequate96-feature position×time comparator beat the split readout; adding DESK contributed about.004–.010 correlation. This is a downstream/surrogate result, not a retrained time-conditioned DESK encoder. |
| DESK E040 | Explicit spatial/temporal GP covariance, including slowly drifting cell residuals | Run with repaired fit/baseline checks. These predict abundance using space and time directly, not DESK's environmental geometry. They do not identify niche change separately from population/observation persistence. |
| Age-model E057 P0 | Smooth per-field intercept drift over1966 onward | Built, real/null/recovery runs. Null/recovery guards passed in the record; the real run used substantial temporal drift but retained demographic contortion. The real refinement hit8000 iterations unconverged, so not a final validated solution. |
| Age-model E057 P2 | Raw covariate × smooth time: population density, built-up land, broader environmental panels | Built and run. These explicitly change the proxy-to-vital-rate relationship with time. The final experiment note marks the arms NOT INTERPRETABLE because a route-SD setting was not actually fixed and refinements stalled or capped. These are not clean negative evidence against drift. |
| Age-model E057 P1 | A low-rank drifting direction in DESK coefficient space, beta(t)=beta+a(t)u | Designed. No implemented-and-tested result found in the audited record. Do not conflate it with the implemented P0/P2 tracks. |
| Age-model E067 | Additive regional habitat correction h(x,t)=sum S_b(x)w_b g(t), shared across four fields; consistent path image for movement survival | Built and run. No-floor null/recovery guards passed. Real67c2 changed habitat trajectories, reduced some western sink patterns and retained a plains trough, but did not remove demographic contortion and used weights4.4 prior SDs out. Older comparator/numerical differences limit the cross-run comparison. |
| DESK E073/E074 | Place-dependent slopes and a covariate-driven anchored delta | Tested surrogate/readout tracks. No explicit calendar-time niche change: slopes depend on kind of place. Breeding-period improvements did not transfer to early CBC; reach-decaying the interaction did not rescue it. Split-training architecture remained a proposal, not an established improvement. |

## What this answers

We did test explicit temporal changes in niche/proxy relationships and regional habitat, primarily **after DESK**, in the population model. We have not confirmed a tested DESK encoder z=f(covariates,space,time) that can give the same environmental covariates different representations solely because the calendar period differs.

The E057/E067 time bases were held at their1966 value before1966. They therefore did not learn an independently changing1902–1940 niche relationship. This is particularly important for the current early-century calibration question. Their successes/failures cannot settle whether historical winter supervision could identify a useful time-conditioned community geometry.

Explicit time may represent unmeasured drivers or proxy changes, genuine niche change, population dynamics, or observation trends. The prior experiments do not identify these interpretations from prediction alone. A future encoder-side test needs a nested covariate-only control, low-complexity time conditioning, and held-location/blocked-era tests with winter observation controls; it should not inherit the old 'drift failed' conclusion from invalid P2 fits.

## Sources

- Current encoder: src/community_encoder/train_DESK/model_arch.py, particularly forward(x,mask) and the optional spatial residual; architecture commit1800777.
- research/experiments/E016.md (including skeptic corrections), E033.md (including E033c), E040.md, E041.md, E073.md (final synthesis/corrections), E074.md.
- Captured age-model E057: final_age_handoff_20261025/E057_proxy_drift_reference.md; implementation commita4a2813.
- Captured age-model E067: final_age_handoff_20261025/E067_spacetime_reference.md; implementation commit6608804 and subsequent floor/operator corrections.

This audit changes no model and does not adopt an explicit temporal input or a new calibrated Z metric.

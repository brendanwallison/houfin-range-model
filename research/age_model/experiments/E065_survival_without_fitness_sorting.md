# E065 — habitat-specific survival without survival-informed sorting

Preregistered: 2026-10-20, after reading E064's numerical failures and before fitting
any E065 arm.

## Why this is a distinct experiment

E064 asked whether urban feeding or shelter can compensate for harsh winters by
improving within-cell survival. Its implementation also let birds ideal-free-sort among
urban strata using that survival difference. That is a second behavioral mechanism,
not implied by feeding. It places the survival coefficient inside an implicit
equal-fitness allocation root. E064d/f/g/i refinements ended with gradients from
`3e10` to `3e27`; 64h/j produced non-finite MAP losses, and the null guard 64n became
non-finite. Those are numerical failures, not negative biological results.

E065 retains the same mean-preserving survival tilts and moderators, but population is
allocated among strata using capacity/fecundity only. The survival tilt acts afterward
on adult and juvenile survival. This tests the physiological compensation hypothesis
without asserting that birds anticipate survival and instantaneously redistribute.

## Arms

| Arm | Mechanism | Route effects |
|---|---|---|
| 65d | generic urban survival | none |
| 65f | survival × cold-winter deficit | none |
| 65g | survival × snow | none |
| 65h | survival × combined harsh winter | none |
| 65i | capacity-deficit + harsh-winter survival | none |
| 65j | both interactions | SD fixed at 0.7 |
| 65n | full 65j structure, synthetic null | SD fixed at 0.7 |
| 65r | full 65j structure, injected signal | SD fixed at 0.7 |

Everything else matches E064: E055a baseline, smooth allocation positive part,
`quick90` 400-step cold MAP, float64 refinement, diagnostics, and western-sink audit.
The injected 65r coefficients are generic capacity 0.5, capacity moderator 0.5,
generic survival 0.25, and survival moderator 0.35; 65n sets all four to zero.

## Predictions and gates

- Winter compensation requires a positive, stable survival moderator, with cold/snow
  arms identifying its source and without worsening demographic intercepts, K trend,
  or the western native-core overlap.
- A near-zero or negative moderator in numerically valid fits rejects the named
  mechanism. Better U alone is not support.
- 65n must recover all four effects near zero and 65r must recover their signs and
  rough magnitudes. Failure of either blocks biological interpretation.
- Refinements are invalid on non-finite values or a flat objective with recent median
  gradient above 10,000. This operational gate is now enforced by the runner rather
  than inferred after diagnostics.
- Western sinks are judged by extent, connected components, native-seed overlap, and
  positive-BBS overlap exactly as in E064; their existence alone is not a failure.

## Results so far (read 2026-10-20; 65n/65r running, 65j failed the plateau gate -- median |grad| 2.4e4)
All five no-route arms converge (2,444-3,569 it). Refined:
| arm | U | sigma_s / sigma_s_mod (beta_k / mod) | Sa/Sj/F | lam0 | plains / west / east med lam | Allee-dead W/P/E | K trend z |
|---|---|---|---|---|---|---|---|
| 65d generic | 186,344 | -.045 | .74/.11/11.0 | 1.39 | .99 / 1.00 / 1.22 | .13/.10/.05 | 5.0 |
| 65f x cold | 186,309 | +.337 / +.800 | .65/.18/10.9 | 1.51 | .99 / .98 / 1.18 | .04/.15/.04 | 5.5 |
| 65g x snow | 186,249 | -.146 / -.177 | .75/.13/10.7 | 1.44 | 1.00 / 1.00 / 1.14 | .18/.15/.03 | 3.7 |
| 65h x harsh | 186,204 | -.080 / -.355 | .72/.14/10.3 | 1.45 | 1.00 / 1.00 / 1.17 | .15/.19/.03 | 4.3 |
| 65i both | 186,426 | -.247 / +.603 (+.281 / +.172) | .72/.18/9.6 | 1.52 | 1.00 / 1.00 / 1.21 | .11/.19/.03 | 4.7 |
Reading (provisional until 65n/65r): (1) the winter moderator's SIGN is unstable -- positive under cold (+.80), negative
under snow (-.18) and the combined index (-.36), positive again in 65i -- exactly the preregistered warning that the
index exploits covariance rather than winter severity; (2) the generic urban survival effect is ~0 or negative; (3) none
of it relieves the refine: Sj .11-.18 (juvenile survival at a fraction of adult), F 9.6-11, Allee-dead 10-19%, the
plains trough flattened to the west's level, K trend 3.7-5.5 sds. No support for urban winter compensation as the
missing mechanism. MAP-stage plots for these arms are being backfilled (0310).

## 65n NULL RESULT (2026-10-20): the structure manufactures a negative capacity tilt
Truth: all four sub-cell effects 0 (synthetic counts at 59a's refined point). Refined 211,111 (4,222 it): beta_k -.293,
beta_k_mod -.109, sigma_s -.077, sigma_s_mod -.144. So beta_k ~ -.3 is what the structure returns with NO effect:
E064c's real beta_k -.353 ("urban strata hold less capacity") is indistinguishable from this artifact and is not
evidence. Of the survival moderators only cold (+.80) clearly exceeds the null, and it is one of three indices with
inconsistent signs -- not a coherent winter-compensation signal. Caveat: the null's truth point (59a refined) fitted
route SD 1.29 while 65j's structure fixes .7 -- a mild misspecification that may inflate null estimates; a clean
truth point (64k or the E066 B3 baseline) would remove it. 65r recovery still running.

## 65r RECOVERY RESULT (2026-10-20): FAIL -- the gate blocks every E064/E065 moderator reading
Truth beta_k .5, beta_k_mod .5, sigma_s .25, sigma_s_mod .35 (synthetic counts at 59a's refined point). Refined
278,790 (1,831 it): beta_k -.21, beta_k_mod -1.47, sigma_s -.28, sigma_s_mod -.74 -- every sign wrong, two by large
magnitudes. With 65n returning beta_k ~ -.3 from no effect, the within-cell compensation structure is not identified
as specified (the sub-cell tilts trade off against the other fields). Per the preregistered gate, NO real-data E064/
E065 moderator estimate is interpretable. Caveat (as for 65n): the truth point fitted route SD 1.29 vs the tested .7.
Not pursued further without a reformulation that is identifiable on synthetic data first.

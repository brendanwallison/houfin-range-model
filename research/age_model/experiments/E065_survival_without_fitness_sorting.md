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

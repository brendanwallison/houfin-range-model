# E058-E061 -- single factors on the pinned-Allee base (pre-registered 2026-10-18)

Base run_55a (V&L dispersal, standard reproduction, n50 pinned 0.059, single release, disease as mortality). One factor each:
- 58a subcell_strata: within-cell urbanisation slices -> can urban "stars" persist inside a plains sink?
- 59a route effects, sd PINNED 0.7 (free sd went to 7.5 = off switch): a bounded home for route-scale heterogeneity.
- 60a capacity manifold amplitude 1.02 -> 0.3: K less able to chase route-scale detail (hypothesis 1a).
- 60b manifold coupling .85 -> .5: K's detail stops dragging the growth fields.
- 61a fitness (ifd3-style) dispersers-only settlement.
Read against 55a (same base): MAP vs refined plains/west/east sink share + median lam_f, Allee-dead, vital rates, U.
Combinations only after single factors read, and only informed pairs (e.g. 58a x 59a: towns vs routes through towns).

## Roadmap -- not yet built (state 2026-10-18)
- E057 covariate-effect drift: P0 intercept drift (all fields) -> P2 built-up drift (generic / Plummer / Plummer-shifted)
  -> P1 agnostic rank-1. Null-calibration guard FIRST (zero-drift synthetic refit, then known-drift recovery).
- Front-speed diagnostic: simulated vs observed invasion front speed at MAP (does a lagging front explain the lam0
  inflation every free refine shows?).
- CBC observation model: 1940-1965 first (pre-BBS front), then all years; winter mapping, party-hour effort offset,
  feeder_hrs as effort covariate (recorded ~1975+), own scale and concentration.
- MAP optimisation adequacy: pinned V&L demography beat free at MAP by 7.5k (53b vs 53a) -> quick90 MAP is not a
  reliable optimum; free-vs-pinned MAP losses are not evidence.

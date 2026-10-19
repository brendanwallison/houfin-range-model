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

## E059a RESULT (2026-10-18): bounded route effects remove most of the refine's contortion
Route effects, sd pinned 0.7 (log scale; fitted raw effects sd ~1, i.e. effects ~0.7). MAP clean (189,743 f32 / 189,755 f64).
| | U | Sa/Sj/F | lam0 | plains sink / med lam / med K | west | east | Allee-dead | corr | K trend z |
|---|---|---|---|---|---|---|---|---|---|
| 55a ref (base) | 186,101 | .64/.31/14.7 | 2.06 | .49/1.00/.012 | .52/1.00 | .42/1.20 | .16/.18/.01 | .809 | 5.6 |
| 59a MAP | (189,743) | .57/.31/5.16 | 1.28 | .67/.88/.147 | .45/1.04 | .43/1.10 | 0 | .862 | 7.8 |
| 59a ref | 169,804 (conv. 597 it) | .49/.44/8.44 | 1.62 | .50/1.00/.137 | .42/1.06 | .37/1.39 | 0/0/0 | .898 | 2.8 |
Giving route-scale heterogeneity a bounded home: NO Allee-dead cells after refinement, plains K stays at its MAP
level (.137 vs base .012), F demand roughly halves (8.4 vs 14.7), the K-trend pressure drops (z 2.8), and the refine
converges in 597 iterations. Confirms E054: the plains zeros the base refine bought with K-collapse are route-level
heterogeneity. Remaining distortions: Sj ~ Sa (.44 / .49), F still above 4-6, eastern lam 1.39, plains trough
shallower than at MAP (.50 vs .67 sinks). (U not comparable with the base: different likelihood.)

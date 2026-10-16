# E001 -- profile typical-habitat fecundity (pre-registered 2026-10-16)

Warm start: run_23a refined (single field, centring, Sj = Sa x ratio, firm weak-Allee prior that the
fit overrode: n50 0.28). There, typical-habitat Fmax = 10.9, ratio 0.95, Sa 0.54, lambda0 2.02.
Variants: pin alpha_f so Fmax(typical) = 3, 5, 8 (everything else free, incl. n50, Sa, ratio).
Measure: refined U (vs run_23a's 199,370.9); per-group log-likelihood change (exp/loglik_decompose.py,
region x era x zero/positive); where Sa / ratio / n50 / dispersal move.

Predictions (written before running):
- H7 (only Fmax*Sj identified): ratio and Sa rise to compensate at Fmax 8 (cost small, < ~30 nats).
  At Fmax 3-5 compensation saturates (Sj <= Sa <= 1), lambda0 must fall, cost becomes large.
- H2 (fill-in speed): the cost lands on POSITIVE counts in the years just after colonization --
  east 1966-1979, midwest/plains 1980-1993 -- as under-predicted counts during build-up.
- H6 (disease recovery): the cost lands on east 1994-2007 (post-epizootic recovery too slow).
- H4 (observation model): the cost is spread across positives everywhere, with concentration shifting.
- H1 interplay: if fast growth exists to push fronts through a strong Allee brake, n50 FALLS when
  Fmax is pinned low.

Kill / decision rules:
- Cost at Fmax 3 < 30 nats: the data barely demand fast growth -> H2/H6 weaken; the implausible
  demography is prior/structure-driven (H7) -> Phase 2 fixes it by structure, not data.
- Cost concentrated (> 60% in one region-era block): that block names the mechanism -> Phase 2
  targets it (fill-in -> immigration/dispersal structure; recovery -> disease form).
- Cost diffuse: H4 -> Phase 2 looks at the count model.

Results / interpretation: (to be filled)

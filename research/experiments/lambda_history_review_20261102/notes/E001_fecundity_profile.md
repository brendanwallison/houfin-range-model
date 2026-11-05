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

Results (2026-10-16, worker job 3514078):
| Fmax | dU | lambda0 | Sa | ratio | n50 |
| free 10.9 | 0 | 2.02 | 0.54 | 0.95 | 0.28 |
| 8 | +48 | 1.96 | 0.59 | 0.96 | 0.26 |
| 5 | +252 | 1.84 | 0.67 | 0.97 | 0.22 |
| 3 | +630 | 1.66 | 0.73 | 0.97 | 0.18 |
Interpretation so far: cost at Fmax 3 is far above the 30-nat kill threshold -> the data DEMAND fast
growth; H7 (pure degeneracy) is not sufficient. Compensation runs through adult survival, the
juvenile ratio pinned at its ceiling (~0.97) throughout -> lambda0 held >= 1.66 even at Fmax 3, with
juvenile survival 0.71 (still implausible). n50 falls with Fmax, as predicted if fast growth exists
partly to push fronts through the Allee brake (H1 coupling). 
WHERE the cost lands (f3 vs 23a, dll -887 = zeros -117, positives -770; prior gives back ~257):
- DIFFUSE over positives -> no block > 60%; by km from release: 100-250 -159, 250-500 -194,
  500-1000 -164, 1000-1500 -191, 1500-2500 -102 (i.e. the whole eastern invasion, not one front).
- by years since first positive in cell: [5,10) -169 is the largest bin; [-10,0) -255 (cells before
  their first positive, multi-route cells); [0,5) +7. Not the "just after colonization" H2 shape.
- by years since disease arrival: spread -20..+25 -> H6 (recovery speed) not singled out.
Verdict: decision rule "diffuse -> H4" fires, BUT the H8 baseline check (below/DIGEST) shows the
count model's NB is MORE dispersed than between-route spread, so the diffuse cost is read as the
TEMPORAL SHAPE of the invasion (H10), not as the count model.

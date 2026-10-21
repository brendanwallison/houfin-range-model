# E076 -- fit quality without priors across the leading candidates (2026-10-21)

User question: how do the candidates compare on fit to the observations, ignoring the priors? U mixes likelihood and
prior, the B3 runs carry fitted route effects (lower RMSE almost by construction), and only the 70-series has a holdout.
- Likelihood evaluation (one-off probe task, not committed): per-observation NB log density from the obs site at the
  MAP and refined points of 70z, 70b, 69a, 69c, 67c2 (conditional on route effects where present), split fitted/held,
  by region and era; RMSE/corr recomputed (67c2's viz RMSE rose .50 -> .88 while U fell ~15,700 -- check).
- 76a / 76b: 69c and 67c2 refitted on the 70z holdout mask, so route-effect models are compared on held-out routes and
  years (the fair test of B3 vs B1).

## Likelihood-only RESULT (probe 0322g, 2026-10-21)
Per-observation NB log density from the obs site (no priors; conditional on route effects in B3), fitted obs only.
| run | n | MAP mean | refined mean | refined sum | west / plains / east (refined mean) |
| 69a (B1, all data) | 192,123 | -1.026 | -.966 | -185,492 | -2.246 / -.656 / -.771 |
| 69c (B3) | 192,123 | -.897 | -.852 | -163,727 | -1.966 / -.572 / -.686 |
| 67c2 (B3 + H) | 192,123 | -.939 | -.850 | -163,338 | -1.960 / -.571 / -.685 |
| 70z (B1, holdout) | 157,371 | -.928 | -.872 | -137,257 | -2.256 / -.635 / -.658 |
| 70b (+ noise) | 157,371 | -.914 | -.867 | -136,355 | -2.239 / -.629 / -.655 |
- 67c2's diagnostics RMSE .88 was a DIAGNOSTICS ERROR: recomputed here, refined RMSE .447 / corr .902 (MAP .501 / .868).
  The viz reconstruction of 67c2's refined point is wrong somewhere; its figures and metrics need re-checking.
- Route effects (69c vs 69a): +21,765 nats conditional on the fitted route effects -- not a fair gain (76a will score
  B3 on held-out routes). The habitat correction adds only +389 over 69c (refined).
- Process noise (70b vs 70z): +902 nats in-sample, slightly worse held out (E070).
- The WEST dominates the misfit: ~15% of observations but ~33% of the negative log-likelihood; its mean count
  (11.2/route) is under-predicted at every point (mu 8.2-10.7). Per-observation density is not comparable across
  regions with different means, but the under-prediction is a bias. Error grows with era (-0.5 in 1966-79 to
  -1.6..-1.9 after 1995) in every run.
- Held-out entries in this probe read 0: masked observations score 0 inside the trace (numpyro mask); the valid
  held-out densities are E070's (metrics.json holdout).

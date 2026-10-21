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

## Probe 2 RESULT: residual structure (2026-10-21; cell-decade means, cells with >= 5 obs)
log(obs/pred) by observed-density quintile (low -> high), refined points:
| run | west | plains | east |
| 68b CONTROL (correctly specified) | -.30 -.25 -.17 -.10 +.21 (spread .51) | -.08 .. +.05 | -.02 .. +.05 |
| 69a (B1) | -.79 -.84 -.48 +.06 +.39 (1.18) | -.21 .. +.29 | -.08 .. -.03 |
| 70z (B1 holdout) | -.84 -.89 -.52 +.03 +.45 (1.29) | -.20 .. +.24 | -.07 .. -.12 |
| 69c (B3 route effects) | -.36 -.37 -.10 +.01 +.11 (.47) | -.10 .. +.15 | -.05 .. -.00 |
- WEST SMOOTHING CONFIRMED for B1: the density-bin spread is 2.3-2.5x the selection artifact -- sparse western cells
  over-predicted, dense ones under-predicted. Route effects (B3) absorb it to the control's level: the heterogeneity
  is fine-scale (sub-cell / route), consistent with dispersal smoothing or sub-27 km patchiness.
- Habitat-edge dependence is WEAK (B1 west by edge quartile -.32 -.25 -.38 -.41 vs control flat -.12): the error is
  not concentrated at sharp habitat edges, so it is not simply the kernel blurring an edge the habitat field draws.
- Era x region: per-observation log density tracks count scale -- the post-1995 rise in misfit is matched by the
  control (east 1995-2009: 69a -2.009 vs 68b -2.001), so it is NOT evidence against the disease term. But the MODERN
  EAST is over-predicted in B1 (2010-25: 69a mu 3.96 vs y 3.36; 70z 3.30 vs 2.62, +26%) -- the continued eastern
  decline is under-represented; 69c much less (3.61 vs 3.36).
- West mean level under-predicted 7-18% in every era in B1 (69a 1966-79: y 20.2 vs mu 16.6).

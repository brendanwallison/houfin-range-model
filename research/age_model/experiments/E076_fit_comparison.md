# E076 -- fit quality without priors across the leading candidates (2026-10-21)

User question: how do the candidates compare on fit to the observations, ignoring the priors? U mixes likelihood and
prior, the B3 runs carry fitted route effects (lower RMSE almost by construction), and only the 70-series has a holdout.
- Likelihood evaluation (one-off probe task, not committed): per-observation NB log density from the obs site at the
  MAP and refined points of 70z, 70b, 69a, 69c, 67c2 (conditional on route effects where present), split fitted/held,
  by region and era; RMSE/corr recomputed (67c2's viz RMSE rose .50 -> .88 while U fell ~15,700 -- check).
- 76a / 76b: 69c and 67c2 refitted on the 70z holdout mask, so route-effect models are compared on held-out routes and
  years (the fair test of B3 vs B1).

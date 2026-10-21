# E072 -- route-level zero inflation (observation process) (pre-registered 2026-10-21)

Why: the shared demand of every E070 refine is persistent near-emptiness of habitat the model rates viable, i.e. many
zero routes where lam_f >= 1 (earlier: plains predicted P(0) .85 vs observed .93). A route samples a patch of its
27 km cell; in sparse cells most patches are empty. If the observation model cannot say "this cell holds birds but this
route's patch does not", the only way to fit those zeros is to empty the cell (K collapse, Allee-dead).
Form (observation.route_occupancy): pi = exp(-mu / m) (route patch empty; Poisson patches of mean size m route
counts), occupied routes NB with mean mu / (1 - pi) -> E[y] = mu unchanged. log m ~ N(log .3, 1). Exact gate logits.
Arms (base 70z): 72a real.  Guards: 72n NULL (counts at 69a MAP without ZI; fit with ZI): pass if demography within
the truth's band and the fitted gate is small where counts are informative (m pulled >= 1 prior sd below .3, or mean
gate on observations with mu > .5 below .1). 72r RECOVERY (counts at 69a MAP with m = .3): m recovered within 30%,
demography within the truth's band.
Read with E070's gates/ranking; key outputs: m, mean gate by region, plains/west K vs 70z, held-out density (scored
with the ZINB) vs 70z refined.

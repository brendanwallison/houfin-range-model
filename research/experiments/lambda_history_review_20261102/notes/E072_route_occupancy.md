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

## RESULTS (2026-10-21)
- 72n NULL: PASS. m pulled from the .3 prior to .013 (3.1 log-sds below); demography .57/.28/4.86 vs truth .57/.31/5.22.
  (Mean gate over ALL observations .42 is dominated by near-empty cells where any gate is 1; not the criterion.)
- 72r RECOVERY: PASS. m .286 vs .3 injected; demography .61/.24/5.07 (Sj slightly low).
- 72a REAL: the data use the zero inflation (m 2.1 at MAP, .66 refined -- larger patches than the prior median).
  MAP .57/.32/5.25 with the best MAP holdout of any arm (-1.882 / -1.229 vs 70z MAP -1.917 / -1.273). Refined
  (3,365 it): .58/.44/13.6, plains .95 < west .99 < east 1.19, Allee-dead .09/.08/.01, plains K .029 -- the usual
  contortion; holdout -1.824 / -1.189 (no better than 70z refined).
Reading: route-level zero inflation is real in the data and helps the plausible MAP predict, but does not remove the
refine's demand; the refine's gain over the MAP is something else.

# E020 -- disease as mortality on the new base (pre-registered 2026-10-16)

Base (fixed by E008-E019): block-3 ideal-free settlement, free departure curve (sparse-cell exodus),
single release, firm weak-Allee prior, juvenile-ratio prior -- overlay map_run30 / E019 sr.
Disease (disease_prior.target = survival, overlay map_run31): added mortality on adults, juvenile
stayers and arrivers, m = 0.8 * modifier(x) * gate(t - onset) * (1 - r(t - onset)) * hill(N_t);
r = host resistance (rec, tau_rec); hill on CURRENT local density (transmission). K disease-free,
c from disease-free rates, so endemic mortality lowers the equilibrium below K.
Warm start: E019 sr (U 197307.3; K-cut disease: median 29%, east at the 50% ceiling, rec 1%).
Variants: dm (free demography), dm_f3 (Fmax 3), dm_corner (Fmax 3, Sj/Sa .5, n50 .10).
Predictions: dU < 0 vs sr (crash speed no longer bound to adult mortality); fitted east peak
mortality well above the old 50% K cut; rec > 0 (resistance) with tau_rec of a decade or two; Sa
free to rise toward its prior / fecundity to fall; the plausible corner cheaper than sr's +596.
Read also: plains corridor table, NE decline/recovery trajectory vs the Hochachka prevalence data.

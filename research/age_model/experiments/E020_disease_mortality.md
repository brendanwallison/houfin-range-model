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

Results (2026-10-16):
| variant | U | dU vs sr | Sa | Sj | F | lam0 | n50 | peak added mortality (dense) | resistance |
| sr (K-cut disease) | 197307.3 | 0 | .66 | .52 | 10.4 | 2.24 | .23 | -- | -- |
| dm | 195911.8 | -1395 | .54 | .43 | 9.9 | 1.80 | .14 | ~0.33 (0.8*sigmoid(-.33)) | rec .01 (none) |
| dm_f3 (4000 its, unconverged) | 196009.3 | -1298 (+98 vs dm) | .80 | .56 | 3.0 | 1.63 | .11 | ~0.28 | .02 |
- Disease as mortality: -1395 nats, second-largest gain of the project. Fecundity 3 now costs +98
  (was +304 under the K-cut disease). n50 falls (.23 -> .14) on its own. lambda0 still ~1.8 free.
- Host resistance not found (rec ~0.01-0.02): with Hill on current density, mortality easing as
  populations thin may already carry what resistance did -- or resistance is unidentified here.
- New base for the next round: dm. Rebased onto it: band weights (E023 dm_band*), block sizes
  (dm_b4/5/7), plus dm_n010 and dm_corner.
E021 sr_crowd: crowding beta stays at its start (1.00), dU -4: crowding pulses irrelevant under settlement.

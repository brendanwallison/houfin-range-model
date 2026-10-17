# E019 -- single historical release (user: 90 founder coefficients is an order of magnitude too many)

invasion_pulse_prior.form = single: ONE parameter, total founders T ~ LogNormal(log 1, 1) in q50
fractions (1 unit ~ 40 birds by a rough route-sampling conversion: a route samples ~25 km2 of a 729 km2
cell at detection ~0.5, so 1 route count ~ 60 birds/cell; q50 = 0.71 route counts). All at the valid
cell nearest Jones Beach, 1940; Allee exemption (10 y) only there. Old prior median ~9 units (~360
birds of sustained inflow); R1b's free founders chose 2.2 (~90 birds) at Fmax 3.
Variants (block-3 settlement, warm ifd3; T starts at its prior mean): sr (free demography), sr_f3,
sr_f3j05 (+ Sj/Sa 0.5), sr_f3j05n010 (+ n50 0.10).
Read: the cost of the plausible corner relative to sr; fitted T in birds; n50 / exodus / lambda0.

Results (2026-10-16):
| variant | U | dU | Sa | Sj | F | lam0 | n50 | T (birds, rough) |
| ifd3 (90-coef grid) | 197642.1 | -- | .52 | .45 | 10.6 | 1.85 | .18 | ~270 |
| sr | 197307.3 | -335 vs ifd3 | .66 | .52 | 10.4 | 2.24 | .23 | 74 |
| sr_f3 | 197611.2 | +304 vs sr | .76 | .73 | 3.0 | 1.73 | .14 | 63 |
| sr_f3j05n010 | 197903.2 | +596 vs sr | .85 | .42 | 3.0 | 1.54 | .10 | 52 |
E018 (free 90-coef founders, Fmax 3, Sj/Sa .5): n50 pinned 0.10 costs +0.5 nats (198117.0 vs 198117.5).
Reading: the historical single release fits BETTER than 90 free coefficients and infers ~50-75 birds.
The strong Allee is no longer demanded at plausible fecundity. Fast growth remains: lambda0 >= 1.5 in
every plausible-fecundity fit, now carried by adult survival .76-.85 (implausible) -- the residual
problem is lambda0 itself, not its split.

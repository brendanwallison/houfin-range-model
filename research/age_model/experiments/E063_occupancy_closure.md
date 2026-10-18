# E063 -- occupancy closure for the low-density regime (pre-registered 2026-10-18)

Why: E062 showed the eastern front is hypersensitive (1e-9..1e-7 perturbations -> decade-scale timing) and no land
cell is ever 0 (fat juvenile kernel tail). The deterministic Allee threshold acting on sub-bird EXPECTED densities is
ill-conditioned, and continuous mate-finding lets fractional birds breed. User (2026-10-18): test extensively; sources
and sinks must stay explicit (they do: lam_fund per cell is unchanged; a realised, occupied-conditional lam is added).

Model (population_model.occupancy): state per cell = expected densities N_a, N_j + psi = P(>= 1 bird) across
realisations, birds Poisson at b = pop_scalar x birds_per_route_count per density unit.
- Arrivals A land in every realisation: psi <- psi + (1 - psi)(1 - e^{-bA}); expected density unchanged.
- Departure, disease (Hill) and reproduction act on m = N / psi; then N = psi m.
- Pairing stochasticity: fecundity x phi(lam) = 1 - e^{-lam}(I0(lam) + I1(lam)), lam from the zero-truncated
  Poisson mean b m (a lone bird -> 0; ~1 - sqrt(2/(pi lam)) for groups). Parameter-free.
- Chance extinction: psi <- psi (1 - e^{-b m}); expected density unchanged (empty realisations hold no birds).
- b -> infinity recovers the mean-field model exactly (tested). Observation: counts from N (option a).
Found while testing: with INDEPENDENT (Poisson) arrivals, occupancy + pairing makes sparse establishment harder than
mean-field, not easier -- lucky founders need CLUMPED arrivals (flock dispersal; NB arrivals = a later extension).
Runs (base 62b = 55a + numerics; settlement off by design):
- 0039a forward check: 50c MAP params under 63f (50c + numerics + occupancy), f32 vs f64. Pass: eastern trajectories
  agree within ~2x across precisions (vs 14x-80x without occupancy).
- 63a/63b/63c: 58 / 29 / 116 birds per route count (detection .5 / 1 / .25), cold MAP -> refine.
Read: f32 MAP loss vs f64 U agreement; plains/west/east sinks and median lam (fundamental) + occupied-conditional
realised lam; Allee-dead; psi maps (plains speckle?); does the refine still need lam0 ~2 and plains K -> .01?

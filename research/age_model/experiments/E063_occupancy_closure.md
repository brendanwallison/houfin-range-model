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

## Forward check RESULT (2026-10-18): occupancy removes the precision hypersensitivity
Eastern-third population, 50c MAP params, f32 vs f64: numerics-only ratio f64/f32 = 1.76 (1966), 3.6 (1974), 17.9
(1982), 61.7 (1990), 18.4 (1998); under occupancy (63f) the ratio is 1.00-1.01 in EVERY year. The front is no longer
ill-conditioned. But at these (mean-field-fitted) params the eastern invasion never takes off under occupancy (east
.076 in 1966 -> .15 peak; legacy f32 reached 585): with pairing stochasticity and independent arrivals, a sparse
front cannot establish at mean-field parameters. No land cell is below psi 1e-6 (the fat kernel tail gives every cell
a tiny occupancy). The refits 63a-c ask whether ANY parameters reproduce the eastern expansion with honest low-density
dynamics; if not, the candidates are clumped (flock) arrivals and the release size/clustering.
CORRECTION (user, 2026-10-18): the 1.00 f32/f64 agreement above is NOT evidence that occupancy cures the precision
sensitivity -- at those params nothing invades, and a population far from the invasion threshold is insensitive in
any formulation. The sensitivity lives at the knife edge. Proper test queued (0034h): fecundity x1..x6 sweep through
the eastern-invasion threshold, f32 vs f64 at each value, numerics-only (62a) vs occupancy (63f). Pass: f32/f64
agree AT the transition under occupancy while the numerics-only model diverges there.

## Knife-edge sweep RESULT (coarse, 2026-10-18; 50c params, fecundity x1..x6, f32 vs f64 eastern population)
Numerics-only: transition near x1-1.25 -- x1: 1982 .86 vs 15.4 (18x), 1998 23 vs 420; x1.25: 1966 183 vs 320
(1.75x); >= x1.5 agree within ~1%. Occupancy: threshold moves up (lone birds cannot breed) -- x1, x1.25 no invasion;
x1.5: 1998 .77 vs 1.29 (1.7x), 2025 244 vs 340 (1.4x); >= x2 within ~1-3.5%. Both are sensitive only at their own
threshold; worst sampled ratio ~18x vs ~1.7x, but the coarse grid samples each transition at different depths.
Some sensitivity is unavoidable at a deterministic bifurcation. Fine sweep queued (10 points across each model's own
window) to compare the WIDTH and WORST-CASE ratio of the sensitive window like for like.

## FINE knife-edge sweep RESULT (2026-10-18) -- occupancy does NOT improve conditioning at the threshold
Worst f64/f32 eastern-population ratio across each model's own transition window (10 F values each):
numerics-only F x.9-1.35: 6.4x (.9), 2.3x (.95), 18x (1.0), 2.9x (1.05), 1.7x (1.1), ~1.1x (>= 1.3);
occupancy F x1.4-2.0: 10x (1.47), 2.5x (1.53), 1.9x (1.6), 2.0x (1.73), 5.6x (1.8), ~1.1x (>= 1.93).
Window widths ("never" -> "established before 1966") are equal: ~x1.5 in F for both.
Reading: the float32 effect is SYSTEMATIC -- f32 always invades later, matching f64 at ~2-3% lower fecundity in BOTH
models. The sensitivity is a steep but smooth timing response (~20 y per 5% of F), not chaos. In parameter terms the
f32/f64 gap is a ~2% effective offset: it explains the MAP f32-vs-f64 U mismatches, not the refine's F 5 -> 14.
CORRECTS both earlier readings ("cures it" -- off threshold; "~10x better" -- coarse grid). Occupancy's case now
rests on biology (lone birds cannot breed; stochastic establishment) and on whether 63a-c fit better, NOT on
numerical conditioning. For HMC: expect a narrow curved ridge in F (and its trade-offs), tractable with a good metric.

## 63a-c RESULT (read 2026-10-20): occupancy does not prevent the refine contortion
Base 62b (55a + numerics) + occupancy at 58 / 29 / 116 birds per route count. All refines converge (grad 70-280).
| run | U ref | Sa/Sj/F | lam0 | plains / west / east med lam | Allee-dead (frac suitable) |
|---|---|---|---|---|---|
| 63a MAP | | | | .93 / 1.07 / 1.16 | 0 |
| 63a ref (58) | 186,283 | .45/.45/10.3 | 1.70 | .76 / .72 / 1.01 | .105 |
| 63b ref (29) | 186,139 | .64/.43/10.5 | 2.06 | 1.06 / .99 / 1.24 | .085 |
| 63c ref (116) | 186,149 | .66/.31/15.2 | 2.13 | 1.01 / 1.00 / 1.26 | .178 |
The familiar distortions return (F 10-15, Sj = Sa in 63a, Allee-dead, a western sink in 63a); U equals the base's
(~186.1k). Occupancy is not the missing piece for the refine's pull; its biological case stands but it is not a fix.

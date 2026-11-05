# E002 -- profile the Allee half-density n50 (pre-registered 2026-10-16)

Warm start: run_23a refined (fitted n50 = 0.28 route counts, 9.5 prior sds above a firm weak prior).
Variants: pin n50 = 0.03, 0.10 route counts (demography free).
Measure: refined U; per-group log-likelihood change; where lambda0 / Fmax / dispersal move; the
Allee-dead fraction is NOT computed here (needs the viz job) -- only likelihood and parameters.

Predictions:
- H1 (front pinning): the cost lands on ZERO counts beyond fronts -- plains and the west-east gap,
  especially 1966-1993 -- as over-predicted occupancy (the weak Allee lets birds leak ahead).
- If H1 holds, lambda0 should FALL with a weaker Allee (less brake to push through) -- the coupled
  demand the 2x2 design was built to separate.
- If the cost is on positives in the east instead, the strong Allee is shaping core dynamics
  (density-dependent growth), not fronts.

Kill / decision rules:
- Cost at n50 = 0.03 < 30 nats: strong Allee is not demanded; run_23a's n50 is a soft optimum.
- Cost concentrated in plains/edge zeros: H1 supported -> Phase 2 asks whether something else
  (habitat at the Plains, or dispersal into hostile cells) stalls the front without an Allee brake.

Results (2026-10-16, worker job 3514078):
| n50 | dU | dll | lambda0 | dispersal intercept |
| free 0.28 | 0 | 0 | 2.02 | -0.58 (prior 2 +- 1) |
| 0.10 | +241 | -326 | 1.69 | -0.86 |
| 0.03 | +757 | -876 | 1.84 | -2.49 |
WHERE (n003): zeros +181 net but east 1966-79 zeros -476; positives -1056. By years since first
positive: [-10,-5) -189, [-5,0) -261, [0,5) -278 -> the cost is AT THE FRONT (+-5 y of arrival).
By km from release: 100-250 -359, 1500-2500 -337 (plains), 2500-5000 +244 (native west gains).
Interpretation: far above the 30-nat rule -> strong Allee IS demanded. Not plains-edge zeros (the
H1 prediction) but the EARLY EASTERN FRONT: with a weak Allee birds leak ahead of the 1966-79
front, and the fit can only slow them by shutting dispersal down (intercept -2.49, 4.5 prior sds).
lambda0 does NOT fall monotonically with a weaker Allee (1.69 at 0.10, 1.84 at 0.03): the front
brake moves from Allee to dispersal, and fast growth is then needed to fill behind the slow front.

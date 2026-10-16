# E014 -- what should settlement follow? capacity vs realized fitness vs nothing (pre-registered 2026-10-16)

E008 viz (ifd3 vs 23a): K-weighted settlement flattened vital rates (survival habitat sensitivity
.34 -> .07; fundamental lambda > 1 in 99% of cells vs ~64%), moved all range structure into K
(median K ~2x) and Allee (Allee-dead 22% -> 31% of suitable habitat, mostly boreal Canada reclassified
from unsuitable); Great Plains flipped from ~37% true sinks to ~1% (fold-criterion sources 36% -> 73%),
against the SDM benchmark (plains mean lambda 0.90) and while still over-predicting plains counts.
Reading: K-weighting gives K a second job (where birds go), so K became the only spatial lever.

Variants (warm start ifd3; settlement.weight):
- i3fit = fitness: equal realized lambda (model's BH fecundity, Allee excluded) across occupied cells;
  sinks fill only by spillover. Nests capacity under equal vital rates. No new parameters.
- i3uni = uniform: habitat-blind block mixing -- the control for "lossless local mixing" vs
  "habitat-directed settlement".
Predictions:
- fitness: dU <= ~0 vs ifd3 (it nests it); vital-rate habitat sensitivities RECOVER; fundamental
  lambda < 1 returns in a share of the plains; Allee-dead area shrinks.
- uniform: if dU is close to ifd3, E008's gain was mixing, not habitat choice; if much worse, habitat
  direction matters. Then viz + departure readout on the winner.

Results (2026-10-16; habitat sensitivities w = [adult survival, fecundity, capacity, juvenile survival]):
| variant | U | dU vs ifd3 | Sa | ratio | F | lam0 | n50 | w |
| ifd3 (capacity) | 197642.1 | 0 | .52 | .87 | 10.6 | 1.85 | .18 | .07 2.96 2.15 1.25 |
| i3fit (fitness) | 197969.1 | +327 | .55 | .97 | 10.7 | 2.07 | .30 | .00 .02 4.11 .11 |
| i3uni (uniform) | 199467.5 | +1825 | .48 | .95 | 13.7 | 1.99 | .18 | .13 4.14 2.38 .19 |
Reading (prediction for fitness FAILED -- the opposite happened):
- Uniform mixing loses the whole E008 gain: the gain was habitat DIRECTION, not lossless mixing.
- Fitness-IFD drove vital rates completely FLAT (all three vital-rate sensitivities -> ~0) and put
  everything into K (capacity sensitivity 2.15 -> 4.11), i.e. it collapsed onto the special case
  where it equals capacity weighting -- and still fit 327 nats worse than ifd3, whose fecundity varies
  in space under K-weighting (a model the fitness family does NOT contain).
- Why: perfect fitness sorting empties every cell below the block's w* (sinks hold birds only by
  spillover). The data have birds in lower-quality cells, so any vital-rate contrast is punished;
  flattening rates is the only escape. Perfect-information IFD is too strong a sorting rule.
- Next option (needs the user): imperfect sorting, weight = K * lambda(N)^tau with tau >= 0 learned
  (tau = 0 is ifd3) -- one parameter for how strongly settlement follows fitness.

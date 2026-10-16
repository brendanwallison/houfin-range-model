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

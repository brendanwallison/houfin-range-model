# E013 -- crowding departure arm (pre-registered 2026-10-16)

Form: logit p = int + slope (x - 0.8) + beta_hi softplus(5 (x - 1)) / 5, x = N/K, beta_hi ~ HalfNormal(4).
The free-sign slope stays the low-density arm; beta_hi = 0 is exactly the previous model (exact warm
start). kappa = 5: the arm starts acting ~0.2 below K (leak 0.14 beta_hi logits at x = 1).
Variants: i3crowd (on ifd3, warm ifd3) -- primary; c23crowd (on 23a, no settlement, warm 23a) -- does
shaping departure alone fix 23a's dumping of birds into low-K cells (99th pct N/K 4-17)?
Predictions: beta_hi clearly > 0 (crowded cells, esp. east 1994-2007, send pulses); dU < 0; the
slope stays negative (front exodus is a separate signal). If beta_hi -> 0: crowding pulses are not
demanded given settlement (settlement already relieves over-K cells locally).
Readout after each: exp/departure_readout.py.

Results (2026-10-16, worker 3514323):
| variant | U | dU | beta_hi | int / slope | p(0) p(1) p(2) p(4) | departing/yr |
| i3crowd (settlement) | 197640.7 | -1.4 vs ifd3 | 0.19 | -4.53 / -7.99 | .87 .002 .000 .000 | ~1% |
| c23crowd (no settlement) | 198345.7 | -1025 vs 23a (+704 vs ifd3) | 6.98 | -1.95 / -4.26 | .81 .14 .48 .995 | ~19% |
Reading:
- With settlement, the crowding arm is NOT demanded (beta -> 0, ~1 nat): settlement already relieves
  over-K cells locally (N/K 99th pct only 1.3-1.6), so pulses add nothing.
- Without settlement, the data demand a clear U-shape: sparse-cell exodus AND an over-K pulse
  (p 14% at K -> 48% at 2K). It recovers 60% of settlement's gain, a parametric stand-in for it.
  41% of juvenile departures come from cells above 1.1 K.
- So "relief of crowded cells" is a real signal, expressible either as local K-directed settlement
  or as crowding-driven long-range departure; the sparse-cell exodus is present in both fits.

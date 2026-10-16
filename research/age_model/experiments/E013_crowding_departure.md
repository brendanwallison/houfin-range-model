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

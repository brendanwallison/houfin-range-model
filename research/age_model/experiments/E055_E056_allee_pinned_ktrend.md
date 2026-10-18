# E055 / E056 -- Allee pinned to literature; continental K trend prior loosened (pre-registered 2026-10-18)

E055 (user, 2026-10-18: Allee effects are unidentifiable from counts -> pin to literature like dispersal, sensitivity around it).
V&L 1996 p. 263: r = 0.15 birds/km2 at K* ~3.8/km2 (4% unmated at K*/2), chosen from r 0.05/0.15/0.25 by spread rate.
Half of breeders mated at N = r/2 -> n50/K* = 0.0197; mapped through our capacity level (median 3.0 route counts) -> 0.059.
- 55a: 51a (V&L dispersal, standard repro) + n50 = 0.059.     - 55b: 53a (V&L dispersal + V&L repro) + n50 = 0.059.
- 55c / 55d: 51a + n50 x1/3 (0.020) / x5/3 (0.098) -- V&L's own r sensitivity.
- 55e: 51a + RELATIVE Allee pinned at n50/K = 0.0197 (no unit bridge; K alone cannot make a cell Allee-dead).
E056 (user: see the K trend un-suppressed). The 'safety valve' is a near-zero prior (budget .05, 3 cosines), not a cap; it is
pushed 11-13 prior sds at every MAP (K x0.3-1.5 over the century). 56a/56b = 55a/55b with budget .5.
Read: MAP vs refined plains/west/east sinks + median lam_f, Allee-dead, Sa/Sj/F, the K-trend shape, and E054's per-obs gain.
Prediction: pinned n50 removes Allee-dead at both MAP and refine; if the refine still contorts it relocates into vital rates
or K level (then the pressure is elsewhere -- E054). Loosened K trend: if the contortion shrinks, the missing mechanism is
temporal and E057 (covariate-effect drift) is the regional generalisation.

## Results so far (2026-10-18)
| run | MAP f32 / f64 | U ref | Sa/Sj/F | lam0 | plains sink / med lam / med K | west | east | Allee-dead W/P/E | corr |
|---|---|---|---|---|---|---|---|---|---|
| 55a MAP | 197,774 / 197,778 (clean) | | .57/.31/5.20 | 1.28 | .67 / .87 / .169 | .49 / 1.01 | .45 / 1.07 | 0/0/0 | .757 |
| 55a ref | | 186,101 (conv. 2689 it) | .64/.31/14.67 | 2.06 | .49 / 1.00 / .012 | .52 / 1.00 | .42 / 1.20 | .16/.18/.01 | .809 |
| 55b MAP (V&L repro) | 201,694 / pending | | .51/.24/4.51 | 1.60 | .73 / .72 / .077 | .16 / 1.56 | .40 / 1.21 | .10/.09/.09 | .730 |
| 55c MAP (n50 x1/3) | 198,243 / pending | | .57/.30/5.16 | 1.26 | .70 / .84 / .161 | .46 / 1.03 | .48 / 1.03 | 0/0/0 | .755 |
Pinning n50 at the V&L value (.059) changes ~nothing: free 51a fitted ~.06 anyway. The refine still makes Allee-dead
cells (.16-.18 W/P) -- by dropping K 14x (plains .169 -> .012) instead of raising n50. As predicted, the absence
switch relocates into K; 55e (relative Allee, where K alone cannot kill a cell) is the decisive arm. K trend still
13 prior sds at MAP.

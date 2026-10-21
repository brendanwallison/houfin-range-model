# E069 -- FFT floor variants of base B1 (2026-10-20)

The hard floor (numerics.fft_floor_ulps: zero convolution outputs below 10 ulps x max) makes U discontinuous (66b
stall: +.45-nat jumps along -grad at steps >= 1e-8; smooth with the floor off) and the refine exploits it (~100
nats). Its purpose was to stop the float32 FFT round-off floor acting as phantom immigration
([[float32-phantom-floor]]). Three legs, same structure (62b = 55a + numerics):
- 62b hard floor; 69a no floor (zero pad kept); 69b SMOOTH floor, x^3/(x^2+tol^2) with the same tol (continuous,
  finite gradient, sub-resolution values -> ~0; tests/test_fft_floor_smooth.py).
Probe 0321ja: U along -grad at the 66b stall point with the smooth floor (as 0321bx).

## 69a RESULT (no floor)
MAP: Sa/Sj/F .57/.31/5.22, lam0 1.29; plains lam_f .87 (sink .67) vs west 1.01, east 1.07; no Allee-dead; k_trend 12.7
sds. Refined 186,066 (3,041 it, tol_du stop; 62b 186,171): Sa/Sj/F .65/.29/15.2, lam0 2.03; plains 1.00 = west 1.00,
east 1.20; plains K .011; Allee-dead west .16 / plains .19 / east .01 (62b .08/.10/.01); west sink .55 -> .58.
The floor's discontinuity caused stalls (66b/66c) but does NOT drive the contortion: without it the refine lands at
the same F ~15, flattened trough and K collapse, with somewhat more Allee-dead cells.
Consequence: E068's "refine stays at the plausible truth" was obtained WITH the hard floor -> rerun as 68b.

## 0321ja probe: the smooth floor does NOT remove the 66b jump (2026-10-20)
At the 66b stall point the smooth floor gives the hard floor's numbers to 6 digits (U 184,940.1157; dU +.108 at 1e-8,
+.454 at 1e-7..1e-5); no floor: smooth, linear, U +100. So the discontinuity is not the threshold's step: smoothing it
changes nothing measurable. What both floors share and no-floor lacks is that negative round-off becomes EXACTLY 0;
something downstream reacts discontinuously to that. Probe 0321jbz decomposes dlogp by site / observation / density
cell at step 1e-7 (smooth vs no floor). 69b (smooth floor, running) is therefore expected to behave like 62b.

## 0321jbz RESULT: the jump is an Allee establishment switch, not the floor's arithmetic (2026-10-20)
At the 66b stall, step 1e-7 along -grad (smooth floor; hard floor identical): dU +.4536, ALL of it in the obs site
(priors ~1e-6). Density: max relative change .986 -- one cell (row 62, col 103; central plains) in 1987 goes .135 ->
.0019, and 55,996 cell-years move by > .1% (its spread); the affected observations are 1991-1996 routes nearby
(mu .21 -> .14, 1.97 -> 1.46, ...). No floor, same step: max relative change .0003, no cell-year above .1%, dU linear.
Reading: with an Allee factor, a thin founding population sits on a separatrix -- above it the cell establishes,
below it the population dies out. The floor makes dying out REAL (sub-resolution densities are exactly ~0, so FFT
round-off can no longer rescue them), which is what it was built for (no phantom immigration). The refine then parks
founding populations ON the separatrix, where an infinitesimal parameter change flips establishment: a jump of
~.45 nats over a step of ~1e-8. Smoothing the threshold cannot help -- the dynamics are continuous but bistable, and
over decades the switch is effectively discontinuous. Without the floor, round-off immigration (~1e-16 x max in
float64) keeps every cell marginally seeded and blurs the separatrix (and in float32 it was phantom immigration).
Consequences: (1) 69b (smooth floor) will behave as 62b; the floor question is really "should extinction be exact",
and with the floor the deterministic Allee model is a switch the optimiser exploits. (2) This is the same absence
switch seen as "Allee-dead" cells -- now located mechanistically. (3) Process noise (E070) does not remove a
deterministic separatrix at MAP; stochastic establishment (E063 occupancy, which blurs the threshold over Poisson
realisations) is the structural remedy. E070 runs on no-floor bases, so it is not exposed to the switch.

## 69c RESULT (B3 no floor; the clean baseline for 67c2)
MAP 180,428: .58/.33/5.15, lam0 1.32; plains .93 < west 1.04 < east 1.17; west sink .51. Refined 172,157 (948 it):
.49/.45/8.69, lam0 1.65; plains .96, west 1.02, east 1.49; Allee-dead .01/.02/0; west sink .54 -- the same point as
64k (hard floor: .50/.45/8.8, 172,035): on B3 the floor changes nothing.
vs 67c2 (+ st_correction): -319 nats; west sink .54 -> .22 and the plains stay lowest, but F RISES 8.7 -> 10.7 and Sa
falls .49 -> .44 -- the persistent correction worsens the demographic contortion while fixing the spatial one.

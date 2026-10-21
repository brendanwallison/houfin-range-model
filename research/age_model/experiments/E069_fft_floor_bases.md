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

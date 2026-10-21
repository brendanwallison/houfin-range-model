# E067 -- smooth spatiotemporal habitat correction (pre-registered 2026-10-20)

Restores, as a general misspecification term, the original (March) st_basis correction to the habitat fields that was
repurposed into the K/disease term in July and then removed. Motivation: every refine converts the plausible MAP into
higher reproductive turnover (F ~9-15) to match TIMING the model cannot otherwise represent (E057: a real temporal
signal; E063: invasion timing ~20 y per 5% of F); DESK's change is spatially coherent but poorly aimed, and a no-DESK
position x time placebo matched the split readout's gain. A smooth REGIONAL time-varying habitat term is the direct
way to let habitat, not vital rates, carry regional timing.
Form (population_model.st_correction): h(cell, t) = sum_b S_b(cell) w_b g(t); S = 15 land-centred cosines (n_freq 3);
g linear over 1966-2023, held before, observation-centred; w ~ N(0, .1). h is SHARED by all four habitat fields (a
habitat-quality correction, as originally) and enters on H -- not inside K's link (the failure of the July version).
Path consistency: journey survival gets the correction's path image sum_b (A S_b) w_b g(t), A = the land-normalised
path operator, applied once to each basis function (src/data/preprocess/st_correction_basis.py; the builder checks it
reproduces the stored Z_disp). Zero weights reproduce the base model exactly (test).
Arms (cold quick90 MAP -> MAP plots -> float64 refine -> plots + sink audit):
- 67a B1 (62b) + correction, real.   - 67c B3 (62b + route SD .7) + correction, real.
- 67n NULL: B1 + correction on synthetic counts at 62b's MAP (plausible truth, no correction). Pass: |w| within ~2
  prior sds, land-RMS h small vs 67a's.
- 67r RECOVERY: truth = 62b MAP + an east-west correction growing over 1966-2023 (basis cos(pi x), w .5). Pass: the
  injected basis weight recovered within ~30%, others near 0.
Read (only if 67n/67r pass): does the refine keep MAP-like demography (F, Sj) and the plains trough; where and when h
is large (regional timing it took over); K-trend pressure; west-sink audit. Compare with E068 (same truth, no
correction) to separate "the correction absorbs real misspecification" from "the refine contorts regardless".

Execution note (2026-10-20): the basis build's operator guard first failed (shape order: integrate_paths returns
(..., M, K), stored Z_disp is (N, K, M) -- fixed for both the check and S_disp), then on max error (.47). The operator
diagnostic (0321ca) showed rebuilt vs stored Z_disp corr .99999, slope 1.000, p99 rel err ~1e-7; only 11-16 of 17,209
cells (islands; the land-weight normaliser at its 1e-10 cut-off) differ. Local and path habitat are consistent. The
guard is now robust (p99.9 < 1e-3 and < .2% outlier cells).

## v2 (2026-10-20): restarted WITHOUT the hard FFT floor
The floor test at the 66b stall (0321bx) showed numerics.fft_floor_ulps makes U discontinuous: with it, U jumps +.45
along -grad at every step >= 1e-8 (|grad| 1.45e6); without it U is smooth and matches the linear prediction exactly
(|grad| 7.4e4), and U differs by ~100 nats at the same point -- the refine was exploiting the floor. Every 62b-based run
(E063, E066, E067 v1, E068) carries it. E067 v1 arms were stopped at the MAP stage. v2 = same design on no-floor bases
(zero padding kept): 67a2 (B1), 67c2 (B3), 67n2 / 67r2 (null / recovery; truth 62b MAP), plus the no-floor B1 baseline
69a. Converged 62b-based results (66a, 66m, 68a, 63a-c) need re-checking without the floor before they stand.

## 67n2 / 67r2 GUARDS (no-floor base; read 2026-10-20): BOTH PASS
- 67n2 NULL (truth 62b MAP, w = 0): refined in 1808 it (tol_du stop, |grad| 43). Max |w| .115 = 1.15 prior sds
  (MAP 1.33); land-RMS h <= .07. No spurious correction from invasion/disease dynamics.
- 67r2 RECOVERY (w[0] = .5 on cos(pi x)): refined in 802 it (|grad| 32). w[0] = .38 (76%, inside the 30% band, right
  sign, the largest weight); other 14 weights <= .117, i.e. the null's noise level. Shrinkage toward 0 under the
  N(0, .1) prior is expected and is why recovery is low rather than high.
The real-data arms 67a2 (B1) and 67c2 (B3) are readable against 69a (no-floor baseline) when they land.

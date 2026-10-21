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

## 67c2 RESULT (B3 no-floor + correction, real; 2026-10-20)
Spatial pattern below = sum_b S_b w_b; h(t) = pattern x g(t), g rising ~-.55 (1966) -> ~+.45 (2023), so a positive
pattern means habitat IMPROVING over the record.
- MAP 187,538: Sa/Sj/F .56/.30/4.98, lam0 1.25; plains lam_f .92 (sink .63) < east 1.07 < west 1.18; no Allee-dead;
  west sink .27 (64k MAP .50). |w| 2.8 prior sds. Pattern: west +.25, plains +.15, east -.21.
- Refined 171,838 (1,185 it, tol_du stop; 64k hard-floor B3 172,035): Sa/Sj/F .44/.41/10.7, lam0 1.63; plains 1.07 <
  west 1.29 < east 1.50; Allee-dead .05/.05/0; west sink .22 (64k .54); plains K .067. |w| 4.4 sds, land-RMS h .38
  (1966) / .27 (2023). Pattern sd .67: west +.48 (p90 +1.05), plains +.19, east -.16 (p10 -1.03).
Read: the correction does real work -- the western sink (54% of the west in 64k, already at MAP) largely disappears,
and the plains stay the lowest region after refinement (64k/69a: plains = west). It does so by letting western habitat
IMPROVE over 1966-2023 and parts of the east decline: regional timing carried by habitat, as intended. But the
demographic contortion is only reduced, not removed (F 10.7 vs 8.8 in 64k, Sj -> Sa, east lam 1.50), and the weights
sit 4.4 prior sds out -- the data want more correction than the N(0, .1) budget allows. The 64k comparison mixes the
floor change; 69c (B3 no floor) queued as the clean baseline.

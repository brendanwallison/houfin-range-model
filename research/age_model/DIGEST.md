# DIGEST (newest first)

## 2026-10-16 evening: the release was a real distortion (E016 R1, E018, E019)
- Single historical release (1 parameter) fits 335 nats BETTER than the 90-coefficient grid and infers
  ~50-75 founders. Under it (or free founders) the strong Allee is no longer demanded (n50 0.10: +0.5).
- Residual problem: lambda0 >= 1.5 at plausible fecundity, carried by adult survival .76-.85.
- Within-cell BUI clustering (E017) not wanted: tilts -> 0 from both sides.

## 2026-10-16 afternoon: what settlement + departure are doing (E010-E015)
- Habitat-directed (K) settlement is the whole E008 gain (uniform mixing +1825). Settlement does NOT
  want to follow vital rates: perfect fitness-IFD flattens them (+327), learned selectivity tau -> 0.19 (-4).
- The negative density-dependent departure (sparse-cell exodus) is worth ~1260 nats under settlement
  and is catastrophic without it (+11288): exodus + K-regathering act as a pair.
- It is not adult site fidelity: an adult offset gives -88 with the shared slope unchanged (-8).
- Crowding pulses are demanded only WITHOUT settlement (U-shape, -1025 vs 23a).
- Plausible juvenile survival is cheap under settlement (+82); fecundity stays ~10 everywhere.

## 2026-10-16 E008 ideal-free settlement: -1729 nats (largest gain to date)
- Block-3 settlement in proportion to K, zero parameters: U 199,371 -> 197,642; west and the early
  eastern front fit better. Dispersal turns strongly NEGATIVELY density-dependent
  (departure 87% at N/K=0 -> 1% at 0.8; 23a flat ~35%): sparse cells empty, settlement regathers. n50 .28 -> .18, lambda0 2.02 -> 1.85, F unchanged ~10.6.
- Caveat: front speed is now partly a structural constant of the block size; profiles of n50/lambda0
  must be redone ON the settlement model (E010 queued) before Phase 2 reads them.

## 2026-10-16 E001/E002 read + baseline checks (lead: H10 invasion-curve shape)
- Both implausible demands are REAL and large: Fmax 3 costs +630 nats, n50 0.03 costs +757.
- Fecundity cost is diffuse over the whole eastern invasion's positives; Allee cost sits AT the early
  eastern front (+-5 y of arrival, east 1966-79 zeros), not the plains edge.
- Fit is ahead of the east in 1966-79 (obs/fit 0.41), behind in 1980-93 (1.32; midwest 1.62): the data's
  curve is more sigmoid than the model's. Strong Allee + suppressed dispersal + lambda0 ~2 sharpen it.
- H8 (route heterogeneity) weakened: NB is already more dispersed than route spread.
- H9 holds over the whole record in the west (fit rising with the covariate lambda trend while obs
  flat/declining) -> H3 territory, deferred to DESK feature work.
- Queued E009 (Fmax 3 + n50 0.03 jointly: additive?) and E004 (dispersal intercept at prior mean).

## 2026-10-16 Phase 1 opened
Diagnostic phase in the ~12 h SSH window, <= 25 SU (DEB23008 only: 381 SU left at start; overall research cap 100 SU). Tools: per-observation likelihood decomposition
(exp/loglik_decompose.py); warm-started profiles from run_23a's refined point. First measurements:
profile typical-habitat fecundity and Allee n50; decompose where each step's likelihood moves.

# DIGEST (newest first)

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

# E080 -- is the refine over-fitting temporal habitat change? (pre-registered 2026-10-23)

Evidence: land-mean fundamental lambda rises ~.17-.36 over 1902-2025 and suitable land roughly doubles with ~no losses,
already in the plausible synthetic truth (DESK's own trajectory) and amplified by every refine (78r: 18% -> 50%); the
rise is concentrated after 1940, i.e. during the invasion it competes with. The central-plains "bridge" (34-40N:
33-53% source at MAP -> 75-85% refined) is created by the refine. In split-readout refines the deviation/level
amplitude ratio SHRINKS for fecundity (.2-.5) but INFLATES for adult and juvenile survival (1-7x), overriding a
DESK-centred .14 prior every time -- DESK's change direction is poorly aimed (dcos .21), so large temporal channels are
not credible. User: the temporal change is over-fit unrealistically; a thin bridge is not implausible (abundance maps hint
at it).
Arms (75c base: split readout r32, holdout mask, no floor; K trend OFF in all three so only the readout carries change):
- 80a STATIC: within_ratio_fixed 0 -- habitat varies in space only (disease the only temporal mechanism).
- 80b CALIBRATED: within_ratio_fixed .14 -- temporal change at DESK's measured credible size, not free.
- 80c FREE control: deviation amplitudes free (as 75c), K trend off.
Read: held-out density (temporal 2021-25, spatial blocks) and the likelihood-only probe vs 75c/78s/70z; demography;
the central-plains corridor at MAP vs refined and by era (1902-15 / 1940-55 / 2010-25); counterfactual release share.
Decision rule: if 80a or 80b predicts withheld data within ~.01 nats/obs of 80c, the free temporal channel is not
earning its keep -> adopt the calibrated (or static) form; if 80c is clearly better on the TEMPORAL holdout, the data need
habitat change and we look at where it acts.

## First pass (2026-10-23): 80a read, 80b/80c STALLED
- 80a STATIC (8,000 it = cap): .54/.18/11.7; plains .74 < west 1.06, east 1.07 -- deep trough, NO bridge; lambda flat
  .88 1902-2025 by construction; Allee-dead .11 W; holdout -1.822 / -1.194 vs free 75c -1.833/-1.183 and 78s
  -1.813/-1.186: temporal about equal, spatial ~.01 worse. The temporal channel buys little prediction for a lot of
  habitat change; the F demand does not come from temporal habitat.
- 80b (.14) stalled |grad| 3.4e6; 80c (free) stalled 5.1e4 -- the absolute-Allee establishment switch on the 75c base.
RERUN on the 79b base -- smooth aggregated-encounter Allee k = 3 (converged in 410 it; keeps mate limitation, which the
user requires -- the near-zero-Allee 79d base was cancelled): 80d static, 80e calibrated .14, 80f free (K trend off in all).
Note: building 80a-c on the absolute-Allee base after E078/E079 had located the stall was an avoidable error.

## Full ladder (2026-10-23; all on the smooth k = 3 base, K trend off)
| arm | temporal habitat change |
| 80d | static (deviation amplitude 0) |
| 80g | TIED: beta_dev = .14 x beta_level -- scaled space-for-time, nothing temporal fitted |
| 80h | CORRELATED: amplitude .14 x level, direction r u_level + sqrt(1-r^2) u_own, r ~ Beta(8,2) per field |
| 80e | calibrated amplitude .14, INDEPENDENT direction (asserts no spatial-temporal relationship -- user: a strong claim) |
| 80f | free amplitude, independent direction |
Tests: tied with c = 1 reproduces the unsplit readout exactly; correlated with r = 1 reproduces tied.
Read: holdout (temporal first), demography, plains corridor at MAP vs refined and by era, fitted r (80h), deviation ratios.

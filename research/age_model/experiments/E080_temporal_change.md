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

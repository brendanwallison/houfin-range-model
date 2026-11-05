# E039 -- diagnostic upweighting of the plains / sparse areas (pre-registered 2026-10-17)

DIAGNOSTIC ONLY (weighted likelihood is not a posterior): population_model.observation_weighting.
- w_plains3: CONUS Great Plains corridor observations x3.
- w_sparse5: clip(5 * (1/(m+1)), 1, 5), m = the cell's mean OBSERVED count since 1966 (data-derived).
Base: dsk_route (E038: route effects, no kappa) so kappa's own discounting does not cancel the weights.
Read (re-scored UNWEIGHTED via the per-region decomposition vs dsk_route): plains nats gained vs west/east
nats lost; plains zero/positive calibration, obs/fit by era; plains sinks (viz); demography.
Trade-off (plains fixable at small cost elsewhere) vs structural inability (plains won't fit / others
degrade a lot).

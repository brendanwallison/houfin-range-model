# E078 -- informed combinations of the overnight leads (pre-registered 2026-10-22)

Leads (all on the 70z holdout base): 75c split readout rank 32 = the first real-data refine with plausible demography
(.55/.37/6.5) but a flattened trough; 77a sedentary native range (r .65, guards pass) = the first refine without plains K
collapse, but demography contorts; 74c dispersers' settlement (K x R0^tau) kept plains K .21 and gives the best MAPs.
Arms: 78a = 75c + 77a; 78b = 77a + 74c; 78c = 75c + 77a + 74c; 78s = 75c with MAP seed 7 (is 75c's basin a lucky start?
G5 was marginal); 78o = occupancy, 78f = fitness settlement, both rerun with the float32-only floor after their MAPs
went NaN on float32 negative round-off (75a, 74e).
Read with E070's gates; 78a/78c succeed if they keep 75c's demography AND 77a's plains K, with the plains lowest and
holdout >= 70z (-1.817/-1.178). 78s: same basin (+-.05 on Sa/Sj, F within 20%) or a different one.
Also (dev probe): jump localisation at the 76b refine stall (clean regime, |grad| 1e5) -- 73h0, 75d, 76b all stalled
without the floor; and float64 diagnostics for 73h (its spatial holdout -1.79 came from float32 diagnostics).
- Added (user, 2026-10-22): 78r = 75c + route effects (SD pinned .7; B3's observation model, best temporal holdout in
  76a and absorbs the fine-scale western heterogeneity of E076 probe 2); 78q = 78r + sedentary native range.

## Stall probe (0326f, 76b refine stall, no floor; 2026-10-22)
U along -grad: step 1e-9 linear; 1e-8 +.0018 (linear -.0009); 1e-7 +.167; 1e-6 +.68; 1e-5 +4.86 -- all in the obs
site (priors ~1e-6). One cell (row 56, col 108; central plains) in 1987 drops .0109 -> .0037 (-66%); 28,617
cell-years move by > .1%. The Allee establishment switch (E069) persists WITHOUT the floor in float64 -- smoother
(no exact extinction) but still near-discontinuous on the scale L-BFGS probes: a thin founding population on the Allee
separatrix. This explains the clean-regime stalls (73h0, 75d, 76b). Structural remedies: stochastic establishment
(occupancy, 78o) or a weaker/absent Allee term (71b, Allee off, never stalled).
73h float64 diagnostics: holdout -1.825 / -1.171 (best spatial so far); the earlier -1.79 was the float32 artifact.
- 78t (added 2026-10-22): rank-64 twin of 78r (75d's r64 fit 25 of 37.5 GiB; 1,700 nats below 75c at its stall).

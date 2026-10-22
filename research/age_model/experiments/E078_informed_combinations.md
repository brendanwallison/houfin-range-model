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

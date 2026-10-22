# E079 -- the mate-finding Allee form (pre-registered 2026-10-22; QUEUED ONLY AFTER 78r READS)

Why: every clean-regime stall (71a, 73h0, 75d, 76b) used the absolute Poisson-encounter Allee; the stall probe located
the cause as the Allee establishment separatrix -- a thin founder flipping between establishing and failing -- which
survives without the FFT floor. The planned HMC needs a smooth posterior, so optimiser workarounds are out (user).
Every mate-finding form that -> 0 at zero density is a strong Allee effect (a threshold exists); forms differ in how
sharp the transition is. The count -> bird mapping is unknown (n50 is in BBS route-count units), so a smooth or
scale-free form is defensible on its own terms (user).
Form (allee_prior.aggregation_k): random encounter averaged over gamma-distributed within-cell local densities
(negative-binomial aggregation): 1 - (1 + gamma N / k)^(-k); k -> inf = the current Poisson form, k = 1 = hyperbolic
(Michaelis-Menten) exactly. Same low-density slope (still strong); broader, shallower transition. Default (k unset)
is bit-identical to the original expression. All Allee uses share one helper (forward step, equilibrium quantities,
viability / Allee-dead classification, the old viz).
Arms on 78r (split readout r32 + route effects; plan-A base per user -- the sedentary native range is a side arm):
79a k = 1; 79b k = 3; 79c relative (N/K, scale-free); 79d Allee off (float32-only floor at MAP).
Read: stalls (does the refine converge?), E070 gates, holdout vs 78r and 70z, Allee-dead, plains trough; plus an
HMC-readiness probe at each refined point (U along random directions at small steps; kinks / jumps counted).
The floor ("paired on arrival") arm waits for a literature check on House Finch pair-formation timing.
- 79e (added 2026-10-22): 79a at rank 64 (78t base + k = 1) -- does the smoother Allee let rank 64 converge?

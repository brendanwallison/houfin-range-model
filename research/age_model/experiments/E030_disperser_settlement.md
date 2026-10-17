# E030 -- settle the DISPERSERS, not everyone; weight by K or by R0 (pre-registered 2026-10-17)

Problem (user + analysis): settlement rearranged EVERY bird in each 3x3 block every year, re-equalising
N/K -- residents reshuffled annually, local vital-rate signatures erased below block scale, sinks
invisible, K the only spatial lever (vital-rate sensitivities collapsed when settlement went in).
settlement.apply_to = dispersers: only this year's adult + juvenile ARRIVALS are allocated within the
block; residents keep their cells (site fidelity). Weights (user: R0 is the simpler fitness signal):
- ct_ds: K (as before, movers only)
- ct_ds_r0: K * R0^tau, R0 = Fmax*Sj/(1-Sa), tau ~ HalfNormal(2) learned (starts at its prior mean)
- ct_ds_pr0: R0 alone
Base cand_tight (Sa .62 Sj .43 F 6.8 lambda0 1.70, single field coupling 1).
Predictions: vital-rate habitat sensitivities RECOVER; plains true-sink share rises; the exodus may
weaken (residents no longer need to be regathered). Fit vs cand_tight (195465.2) is the cost/benefit.

ct_ds (K weight, warm in the exodus regime): U 203094.6 (+7629) -- dispersers-only settlement BREAKS the
sparse-cell exodus (it relies on regathering residents). Not a fair test: retest ct_ds_pos with dispersal
reset to the cold-start mode (int .8 / slope +1.2).

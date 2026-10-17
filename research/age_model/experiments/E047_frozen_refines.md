# E047 -- which parameter blocks must move for the refinement gain? (pre-registered 2026-10-17)

E044: run_40 MAP (realistic: plains 70% sinks, median lam .85) is 12,813 nats short of the refined fit; ~5,600 of
the gain is eastern positives 1980-2025; refining raised growth broadly and erased the plains trough.
Block-frozen L-BFGS refines from the run_40 MAP (HOUFIN_REFINE_FREEZE holds sites at MAP values):
- F1 freeze habitat-field pattern: manifold_factor, manifold_idio, manifold_coupling_logit, w_scale_raw
- F2 freeze demographic levels: alpha_a, alpha_j, alpha_f, n50_raw
- F3 freeze both
Free in all: K level, k_trend, dispersal, settlement, disease, observation (kappa, phi0, fledgling share), release.
Read: share of the 12,813 recovered; plains trough kept (viz); where the remaining gap sits (decomposition).
If F3 recovers most of it with the trough intact, the contortion is opportunistic -> a modest constraint suffices.

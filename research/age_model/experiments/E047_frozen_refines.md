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

F1 RESULT (habitat-field pattern frozen at the run_40 MAP): U 198,699.9 -> 192,266.5 (~51% of the 12,650 MAP->refined
gap). Source/sink KEPT REALISTIC: W .31/.13/1.17, P .67/.44/.85, E .41/.13/1.16 (sink / K-weighted sink / median lam_f)
vs full refine W .37/.55/1.08, P .42/.45/1.05.
F1 - full (zeros/positives): east positives 1980-93 -379, 1994-07 -841, 2008-25 -1021; plains positives 1994-07 -328,
2008-25 -457; west spread. => The contortion lives in RESHAPING THE HABITAT FIELD, and what it buys is (1) eastern
abundance 1980+ (~2.2k nats) and (2) modern plains positives (~0.8k): an east that outperforms its covariates and a
trough that softens recently (user: urbanisation speckle). Needed: a time-varying, spatially local channel (urban /
anthropogenic) reaching the east and the modern plains WITHOUT moving the static pattern -- higher DESK rank (E048)
or a direct term.

F3 RESULT (habitat pattern AND demographic levels frozen): U 198,699.9 -> 192,497.0 (~49% of the gap) with source/sink
fields identical to the MAP. F1 adds only ~230 nats by also freeing demography. => about HALF the refinement gain is
non-contorting (K level, dispersal, settlement, disease, observation, release); the other half comes entirely from
reshaping the habitat field (east 1980+ and modern plains positives).

F2 RESULT (demographic levels frozen, habitat FREE): U 198,699.9 -> 187,669.0 (87% of the gap) but every region becomes
mostly high-K SINKS: W .66/.69/.97, P .81/.76/.85, E .61/.49/.91. Confirms the habitat field is the contortion engine:
free to reshape, it builds an implausible immigration-dependent geography.
Summary: frozen habitat (F1, F3) keeps realism with ~50% of the gain; free habitat (F2, full) gains the rest by distorting
the source/sink map.

**CONDITIONAL (2026-10-17):** run on the run_40 base whose learned departure clamped dispersal to ~13-19% (E043). Not a viable model; conclusions about refinement contortion may be downstream of the clamp. Superseded by E050.

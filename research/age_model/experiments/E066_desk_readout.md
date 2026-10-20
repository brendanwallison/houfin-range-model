# E066 -- DESK readout: split level/deviation x rank, on three bases (pre-registered 2026-10-20)

Why: the DESK experiments (branch research/desk-temporal) recommend, as implications not decisions: (1) a SPLIT
readout -- the cell's long-run level z-bar (1996-2025 mean) and the within-cell deviation z(t) - z-bar get separate
iid amplitudes (E059e: one amplitude is fitted to spatial variation, so the temporal coefficients ARE the spatial ones;
split raises out-of-community change skill .089 -> .22 and fixes the scale; amplitudes between >> within ~50x;
E012: a recent reference window erases the gain); (2) rank 64 (or ~32) over 24 (+.02-.04); (3) a seed ensemble
(later; needs an input build). Here E057 showed a REAL temporal signal the age model lacks, and every refine
contorts demography -- plausibly fighting DESK's poorly aimed temporal deltas read with spatial coefficients.
Caveats carried over: the split's temporal gain may be partly regional trend (a no-DESK placebo deviation block
matched it); DESK's backcast change is still 3-5x too large on early CBC lists even with the split.
Implementation: population_model.readout {split, ref_window [1996, 2025], rank}; within block = an independent
exchangeable draw (same coupling rho) with its own amplitudes, same prior as the level block; centring on the level
only; path features split the same way; rank slices the leading features of numpyro_input_vl_r64 (built here).
Nests the single-amplitude model exactly when within weights = level weights (float64 test, 1e-9).

Bases (chosen ahead of results; user 2026-10-20): B1 = 62b (V&L dispersal, n50 pinned, numerics); B2 = B1 + E057 P0
intercept drift (calibrated); B3 = B1 + route effects with SD truly fixed .7.
| readout | B1 | B2 | B3 |
|---|---|---|---|
| single amplitude r24 | 62b (have) | 66e | 66i |
| split r24 | 66a | 66f | 66j |
| split r32 | 66b | 66g | 66k |
| split r64 | 66c | 66h | 66l |
| single amplitude r64 (rank control) | 66d | | |
Every arm: cold quick90 MAP -> MAP diagnostics + west-sink audit -> float64 refine (plateau gate) -> diagnostics + audit.
Read (pre-registered): (i) fitted within/between amplitude ratio per field (DESK expectation << 1); (ii) does the
refine keep the MAP's plains trough and plausible demography (Sa .55-.65, Sj .3-.45, F 4-6, no broad Allee-dead or
western sink) -- the primary criterion; (iii) K-trend pressure; (iv) in B2, does the P0 drift SHRINK once the readout
is split (drift was compensating for the readout) or persist (a temporal signal DESK lacks); (v) 66d vs 66c separates
rank from split. U is compared only within a base. Budget ~90 SU (Phase 1 ~450 + ~30 committed + 90 = ~570 of 600).

Amendment (2026-10-20, before any E066 fit; user question on interactions with the weak temporal corrections):
the deviation block (z(t) - z-bar, zero-mean per cell only WITHIN 1996-2025) can carry a continental temporal
trend -- especially in the 1966-1995 backcast -- that competes with P0's intercept drift and the K trend; the priors
(tight drift, free deviation amplitude) decide which absorbs it. Added readout for B2 (and all arms): the land-mean
fitted deviation term per field over time, beside the P0 drift, so "drift shrank because the readout explained it"
is distinguishable from "the deviation block took the continental trend". P2 (covariate-effect drift) is collinear
with the deviation block wherever a covariate trends; it is only interpretable on top of a split readout.
Candidate E067 (not yet registered; after E066): split readout + (a) a REGIONAL trend correction -- low-rank smooth
spatial basis x linear time, time-centred, strong prior, bounded multiplicative (cf. the generic spatiotemporal basis
that annihilated eastern K) -- motivated by DESK's finding that change is spatially coherent but not environmentally
determined and that a no-DESK position x time placebo matched the split's gain; (b) a separate, smaller amplitude for
pre-1996 (backcast) deviations, motivated by DESK's backcast change being 3-5x too large on early CBC lists. Each
with null + recovery guards.

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

## 66a RESULT (split r24 on B1, 2026-10-20): best MAP, among the worst refines
MAP: plains .76 sinks med .81 vs west 1.08, east 1.03 (deepest trough yet); Sa/Sj/F .56/.28/4.92; within/level
amplitude ratios .82-.90. Refined (converged, 1,183 it; 185,061 = -1.1k vs 62b): Sa/Sj/F .57/.52/15.2, lam0 2.41;
plains 1.00 / west .93 / east 1.07 -- trough INVERTED; Allee-dead 3-5%; within/level .96/.25/1.16/.42 (survival,
fecundity, capacity, juv survival) -- NOT << 1 as the DESK work expected (~1/50). The refine leans on DESK's temporal
deviations and contorts harder: more weight on poorly aimed DESK change needs more compensation. Extra DESK temporal
freedom is not the missing signal (cf. E068: the data do demand one).

Amendment (2026-10-20, after 66a, before any 66m fit; user-approved "route 3"): 66a gave the deviation block the level
block's prior and the refine pushed deviation/level to .25-1.2 while contorting. 66m = 66a with the deviation
amplitude tied to the level: w_scale_within = w_scale x ratio, ratio ~ LogNormal(log .14, .7) per field -- centred on
DESK's measured split (E059e: kernel variances ~50x apart -> sd ratio sqrt(1/50) ~ .14), 95% prior mass ~.035-.55.
Read: fitted ratios (do the data insist beyond the prior?), and whether the refine keeps 66a's MAP trough and plausible
demography. If 66m holds where 66a contorted, the over-free deviation block drove 66a; if it contorts alike, the
pressure is independent of DESK's temporal channel (pointing to E067). Later, the principled long-run version is an
empirical-Bayes Mercer-space prior from DESK's per-direction out-of-community errors (route 1).

## 66b/66c (split r32 / r64, B1) -- refines STALLED (median |grad| 2.1e6 / 1.5e5); 66g/h/k/l on hold
Gradient anatomy at the 66b stall (U 184,940 -- lower than 66a's 185,061): |grad| 1.45e6 spread over alpha_a (9.8e5),
both manifold blocks (level 7.2e5/4.7e5, within 3.5e5/2.3e5), w_k_trend (3.3e5), alpha_k, alpha_j. Along -grad, U RISES
by ~+.45 at BOTH 1e-7 and 1e-5 -- a JUMP (discontinuity), not a corner. 66a (r24) at its optimum: |grad| 639, small
non-descent wobble only. Suspect: the hard FFT floor in numerics (fft_floor_ulps zeroes sub-threshold convolution output,
so values crossing the threshold make U jump); B1 carries it. Test queued (0321bx: same point, floor on vs off).

## 66m RESULT (2026-10-20): calibrated deviation prior is overruled; contortion unchanged
MAP 203,102: plains .82 sinks med .77 vs west 1.07, east 1.00; Sa/Sj/F .55/.28/4.89; deviation/level .22-.38.
Refined (converged, 5,620 it; 184,639): Sa/Sj/F .51/.46/16.2, lam0 2.21; plains .99 / west .94 / east 1.11; west sink
.41 -> .63; deviation/level 1.7-3.3 -- ~3 log-sds ABOVE the DESK-centred prior (.14). The data push DESK's temporal
deviations to outweigh its spatial level, the opposite of DESK's own measured split, and the contortion is the same as
66a's. Constraining DESK's temporal channel does not remove the pressure; the refine overrides the prior to get it.

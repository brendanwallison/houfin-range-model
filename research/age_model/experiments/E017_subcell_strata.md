# E017 -- sub-cell urbanisation strata (user's coarse-dispersal / fine-IFD idea) (pre-registered 2026-10-16)

Each 27 km cell = 6 within-cell built-up-intensity slices (HISDAC-US BUI quantiles 5/25/50/75/90/99,
area weights .15/.225/.25/.2/.12/.055; CONUS only -- uncovered cells homogeneous). Every year after
dispersal (and block settlement, if on) the cell's birds are redistributed among slices by the fitness
ideal-free rule; Allee + Beverton-Holt act on each slice's LOCAL density. Mean-preserving tilts keep
DESK's cell K and Fmax exactly: subcell_beta_k (K) ~ N(0,1), subcell_delta_f (Fmax) ~ N(0,0.5), per sd
of log BUI. Both 0 = the previous model (tested).

Variants:
- s_ifd3: strata + block-3 settlement, warm ifd3.
- s_only: strata, NO block settlement, warm 23a -- was block settlement standing in for this?
- s_ifd3_ex0: s_ifd3 with the release Allee exemption OFF -- does clustering replace the patch?
Predictions (H_cluster): beta_k and/or delta_f clearly > 0 (towns favoured); n50 FALLS (clustered
colonists clear mate-finding locally) and the sparse-cell exodus slope relaxes toward 0; lambda0
falls (town slices fill fast, cell totals slower); gain concentrated in the eastern transient
(+-10 y of first positive, 250-1500 km from release). Null: tilts ~0, nothing moves -> within-cell
urbanisation is not the missing structure (then F2 detection lag is the main bet).

First runs (2026-10-16) -- VOID as tests of the idea: new sites took init_to_value's random-uniform
fallback, so the refines began at U ~2.3e6 and stopped in worse basins than the models they nest:
s_ifd3 U 197729.0 (+87 vs ifd3; beta_k -1.37, delta_f 0.14), s_only 199338.6 (-32 vs 23a; beta_k -0.10,
delta_f 0.10). Fixed: missing warm-start sites now start at their prior mean (hmc_common.
init_values_else_mean) -- 0 for both tilts, i.e. exactly the nested model. Re-run as s_ifd3b / s_onlyb.
Caveat carried to E012/E013/E015 (adult offset, crowding beta, tau): same random init, but those
ended near their nested values with small gains, so they are less suspect; re-run if they matter.

Reruns from the nested point (s_ifd3b, s_onlyb): tilts stay EXACTLY 0 (21 iterations, dU ~0). Not a
verdict either: with mean-preserving tilts every first-order effect cancels in the area-weighted sum,
so the gradient at beta = delta = 0 is identically zero -- the nested model is a stationary point and
the question is second order. Third attempt: start off zero via HOUFIN_REFINE_INIT --
s_ifd3c (beta_k 0.5, delta_f 0.3), s_ifd3d (beta_k -0.5, delta_f 0.3), s_onlyc (0.5, 0.3),
s_ifd3_ex0 (0.5, 0.3). Compare U against ifd3 197642.1 / 23a 199370.9.

Off-zero starts (2026-10-16):
| variant | start (bk, df) | U | dU vs ifd3 | end bk | end df |
| s_ifd3c | (0.5, 0.3) | 197674.9 | +33 | 0.40 | 0.00 |
| s_ifd3d | (-0.5, 0.3) | 197637.5 | -5 | -0.04 | 0.00 |
Reading: from both sides the fecundity tilt goes to 0 and the K tilt to ~0 (the +0.4 start stalls in a
worse basin). Within-cell BUI clustering is NOT wanted: the fit keeps the cell-mean Allee brake for the
early eastern front, and clustering (which relieves mate-finding) works against that brake.
H_cluster (as built: BUI slices, fitness IFD, mean-preserving tilts) -> not supported.

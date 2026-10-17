# E022 -- learned juvenile kernel tail (pre-registered 2026-10-16)

dispersal_kernel.band_weights: log-weights of the 155-483 km and >483 km juvenile bands relative to
0-155 km, N(0,1), mass-preserving (age_forward.juvenile_band_weights); 0 = current kernel. Each band
keeps its precomputed journey survival Q, so no new path features.
Variants on the single-release base (warm sr): sr_band (start 0), sr_band_t (start outer +1.0),
sr_band_corner (Fmax 3, Sj/Sa .5, n50 .10, start outer +1.0).
Prediction (H10 / Kot et al.): outer band weight UP (satellite seeding), lambda0 DOWN, the plausible
corner cheaper than +596; the sparse-cell exodus may relax (long jumps replace it as the front driver).

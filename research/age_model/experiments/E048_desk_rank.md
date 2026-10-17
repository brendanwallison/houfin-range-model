# E048 -- DESK rank 48 / 64 (pre-registered 2026-10-17)

User: higher-rank / more robust representations gave marginal abundance gains in DESK experiments; more
flexibility might let the model capture e.g. an urbanisation signal that rank 24 drops. The model uses the top 24
of DESK's 64 latent features. Inputs rebuilt at latent_dim 48 and 64 into SEPARATE dirs (numpyro_input_r48/_r64;
shared rank-24 inputs untouched); cold MAP (quick90) -> refine -> viz of BOTH the MAP and refined points, run_40
structure. Rank 64 runs with XLA mem fraction .95 (Z_disp ~13 GB float64).
Read vs rank-24 cold run_40 (MAP: plains 70% sinks, lam .85; refined U 186,050.7, corr .808): fit; MAP realism;
whether refinement still erases the plains trough (more flexibility could capture urban signal OR give refinement
more room to contort); demography; east under-prediction at the MAP.
Budget: Phase 1 cap raised to 150 SU (DEB23008 at 309 SU on 2026-10-17 afternoon).

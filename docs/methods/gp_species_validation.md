# GP species validation — design

Status: built 2026-10-06, not yet run on TACC. Code: `train_DESK/gp_kernels.py`,
`train_DESK/validate_gp_species.py`, `train_DESK/gp_species_analysis.py`; tests in
`tests/test_gp_species.py`; job `scripts/tacc/submit_gp_species.sh`, overlay
`config/overlays/gp_species_base.json`. Outputs land in `<run>/gp_species/`.

Temporal holdouts (added after the first run, which covered space only): on a `desk_tempho_*`
checkpoint the withheld years are excluded from every fit, and rows are scored in three groups
(space, time, space_time) with matching change sets. The first version would have leaked the
withheld years into training; `row_splits` now owns that exclusion and a test checks it.

Built differently from the plan below, with reasons in the code:
- **Baseline shapes are shared across species.** The spacetime and covariate GPs fit ONE set of
  lengthscales for all evaluation species, plus a per-species amplitude and noise. That is the
  same structure DESK has, so the comparison is between kernel shapes with equal per-species
  freedom. The shapes are fitted by the envelope gradient, never by differentiating an
  eigendecomposition. Each fit reports whether it converged.
- **Thinning removes training rows** (cell-years) shared by every species, not each species'
  detections separately. The curve is read against each species' remaining detections.
- **The regression response** is `(err_desk^2 - err_base^2) / MSE_base`, per (cell, decade,
  species) for level and per (change cell, species) for change. Errors are two-way clustered by
  species and held-out block, and the response is winsorized at 0.5%/99.5%.

## Question

The similarity suite asks whether `z(x)·z(x')` reproduces observed community similarity at
held-out cell-years. This suite asks the question the kernel is actually used for: **as a GP
kernel, does it predict an individual species' abundance in held-out blocks better than the
alternatives?** Same blocks, same years, same point aggregation, and the same metrics wherever
they translate.

A GP whose kernel is `z·z'` (rank 64) is exactly Bayesian linear regression on `z`. Inference is
therefore exact and cheap (Woodbury, O(n·64²) per species), and the existing ridge readout
(`validate_baselines.fit_species_readout`) is its posterior mean at one fixed hyperparameter. What
the GP framing adds is hyperparameters learned per species, a predictive variance, and the
probabilistic scores that variance makes possible.

## Checkpoint

The production run (`config/overlays/production.json`, `encoder/desk`, 2026-09-14) trained with
`holdout_frac: 0`, so it has no held-out blocks to grade on. The suite grades its holdout
predecessor, **`sweeps/desk_hp/sweep_t0_f100_base`** (2026-08-26):

- It is the `base` configuration production was retrained from: 532 held-out cells, half-life
  10.47 y (production's is 10.72 y).
- The newest holdout run by timestamp is `nest_t0_f100_nest_probe`, but that is a NestedLoRA
  probe, an arm that has since been closed, not the production recipe.
- Two differences from production: `base` stopped at its selected best epoch (107) instead of a
  fixed 200, and that epoch was chosen by a held-out kernel metric on these same blocks. That is a
  mild optimistic bias in DESK's favour. It is weaker for out-of-community species than for the
  similarity metric, but it is not zero.
- The basis is asserted at load time (`check_basis_matches` against `esk_balanced/spacetime`).

The path, its `holdout_cells.npy`/`buffer_cells.npy`, and its `ema_half_life` are recorded in the
report header.

**Rule:** results from this suite never select epochs, configurations or species. Otherwise the
held-out blocks turn into tuning data.

## Evaluation species

- Every BBS species that crosswalks one-to-one to the eBird taxonomy (666 of 764 entries; the
  unid./hybrid/slash/race rows drop out at the join), **minus the reference community and minus
  House Finch** (AOU 5190 / `houfin`, checked by AOU code). That leaves about 570 species.
- House Finch never appears in this suite: grading the kernel on the deployment species would be
  double-dipping.
- No filter by taxon or by data volume. A species is dropped from a **metric**, never from the
  evaluation, only when that metric is undefined for it (for example, zero held-out variance makes
  RMSE skill 0/0). Every metric reports how many species it dropped and the reason. Species with
  zero training detections stay in, because that regime is where DESK's claim lives.

## Data

An all-species cell-year matrix, built by the same chain as `validate_bbs_routes.load_observed`:
`load_usca_observations` → crosswalk → `build_community_matrix` (route `SpeciesTotal` summed,
divided by the number of QC-passing route-years) → `densify_community` (a surveyed absence is a
real zero) → `log1p`. The only change is the species list. Columns are guarded by an
`assert_same_layout` analog against a saved species order, because a misaligned column is the
failure this pipeline has already had once.

Known undercount: race-level BBS rows are dropped instead of being added into their species.

## Task (A only for now): block extrapolation

- **Conditioning set:** training-block cells (not held-out, not buffer), all years.
- **Targets:** held-out-block cells, all surveyed years, restricted to `common_holdout_years`
  where the run defines them.
- **Likelihood:** Gaussian on log1p abundance with a per-species constant mean. If a result is
  ambiguous, the fallback is a count likelihood (negative binomial) fitted by MCMC.
- Task B (reveal the held-out cell's modern window, predict its past) is deferred.

## Predictors

All predictors share the mean function and noise model. Only the kernel differs.

| Name | Kernel | Hyperparameters per species |
|---|---|---|
| `desk` | `s² · z_ema(x)·z_ema(x')` | `s², σ², μ` |
| `no_change` | as `desk`, with each cell's modern-window `z` copied to every year | as `desk` |
| `spacetime` | Matérn-3/2(space) × exponential(time) | lengthscales, `s², σ², μ` |
| `covariate` | ARD-RBF on DESK's covariates, output-EMA'd at DESK's learned half-life | ARD lengthscales, `s², σ², μ` |
| `ceiling` | `s² · z_esk·z_esk'` from the observed community (split-half where possible) | as `desk` |

- **Hyperparameters from the stat model.** The amplitude of the stat model's `w_env` prior
  cannot carry over, because it is in vital-rate units and this suite works in log1p abundance.
  What carries over is the **shape**: an isotropic prior on coefficients, meaning a fixed `z·z'`
  with no per-dimension weights. So the primary `desk` learns only the scalar `s², σ², μ`.
  - Diagnostic only: an ARD variant with per-dimension weights, which shows how much the fixed
    isotropy costs.
- **Fitting.** Hyperparameters are fitted by marginal likelihood on a training subsample
  stratified by year-window × region, the same sampler as the route suite.
- **Conditioning on all training rows.** `desk`/`no_change`/`ceiling` condition on every training
  row exactly, through the low-rank form. `spacetime` and `covariate` use exact local GPs over the
  nearest K training rows in each kernel's own metric, with sensitivity reported at two K. A
  common subsample would handicap the spacetime GP specifically, because its information is in
  near neighbours.
- **Known asymmetry, not corrected:** DESK sees a 5×5 neighbourhood through its convolution,
  while the covariate GP sees the cell's own covariates only.

## Metrics

**Primary (fixed before any results):** per-species RMSE skill on held-out same-cell change
(early epoch → modern epoch, `epoch_gate` windows):

    skill = 1 − rmse(Δ_pred, Δ_obs) / rmse(Δ_no_change, Δ_obs)

The skill is reported against every other predictor too. Pooling is the median across species
plus the share of species where `desk` beats each baseline, both with bootstrap CIs that resample
species and held-out blocks.

**Secondary:**
- RMSE skill on held-out **level**;
- held-out log predictive density skill and CRPS;
- 50%/90% interval coverage;
- per-species direction skill and rank τ: `species_change_agreement` turned the other way, so
  that within a species it compares the direction of change across cells, keeping the
  majority-direction correction and the abstain-on-zero rule;
- `resolving_room` (the gap from null to ceiling) per species;
- the posterior share of variance each kernel component explains.

## Regressions: what predicts performance

- **Unit:** species × extrapolation stratum. **Response:** RMSE skill of `desk` against each
  baseline in turn, weighted by the number of rows.
- **Model:** mixed model with a random intercept per species.
- **Species covariates:**
  - phylogenetic and AVONET trait distance to the community, computed for **all** species (the
    saved pool covers only 389);
  - co-occurrence similarity to the community, on training blocks only;
  - urban tolerance where it exists;
  - migration class;
  - training detections, held-out detections, prevalence;
  - optionally, USGS trend credibility flags.
- **Extrapolation degree, per row:**
  - distance from the cell to the nearest training cell;
  - covariate novelty (Mahalanobis distance to the training-block covariate distribution);
  - years from the modern window.
- **Interactions:** extrapolation degree × similarity, and extrapolation degree × data
  availability. These answer whether DESK's advantage grows with harder extrapolation and whether
  that depends on the data.
- **Collinearity:** density and similarity are correlated, because the community was picked after
  data-volume gates. VIFs are reported, and the effects are read as partial effects, not
  marginals.

**Data-poor arm:** thin each species' training detections to 100/30/10/3% with the test set fixed,
then compare skill-vs-n curves across predictors. This is the direct test of "better in data-poor
regimes", and it is sharper than the cross-sectional regression on detection counts.

## Known leakage, recorded rather than fixed

`run_spacetime_esk` fitted the ESK landmarks and eigenpairs on every `bbs_points` row, including
held-out blocks. For out-of-community species the leak is only indirect: the basis geometry saw the
held-out blocks' community composition, never the evaluated species. `desk`'s `z` at held-out
cells still comes from covariates alone.

## Reuse plan

- Reused unchanged:
  - `validate_bbs_routes`: `load_observed` chain, `desk_z_ema`, `stratified_sample`,
    `epoch_gate`, `noise_floor`, `bootstrap_skill_ci`, `resolving_room`, `assert_complete`;
  - `desk_training.apply_output_ema` for the covariate GP's inputs;
  - `covariate_io` normalisation, which is fitted on training pixels only.
- New:
  - `train_DESK/gp_kernels.py`: the kernels above, exact low-rank and local inference, and
    marginal-likelihood fitting;
  - `train_DESK/validate_gp_species.py`: the runner, metrics and regressions;
  - an all-species matrix builder;
  - tests with planted cases where the right answer is known (for example, a species generated
    from `z` that `desk` must win, and one generated from a spatial field that `spacetime` must
    win).
- Runs on TACC: the covariate tiers are only available there.

# Executed code for the LA1 HMC target

This file records what actually ran. It guards against describing dead code.
Tags: **[E]** measured or quoted, **[L]** inferred from code, **[U]** unknown.

## 1. How this was measured

- **Driver [E].** The driver is `ms_driver.py`, a session scratchpad script (`...\scratchpad\ms_spec\ms_driver.py`). It runs the main-repo `research/exp/invasion_sandbox/laplace_fit.py` (`dee075d`) with the exact HMC command-line data flags (RESOLVED_CONFIG.json `cli`). The overlay is `J200VW_ALL_local.json`, the init is LA1 `best_params.npz`, and the model code comes from `TAKEOVER_CODE_ROOT` = worktree `0f73b31`. It ran through `research/bin/local_takeover_run.sh` on CPU, in float64, on cores 8-15.
- **The evaluation [E].** `laplace_fit.run_hmc` was replaced by an evaluator. It receives the same `parts`, `coords` and `x0` that `run_hmc` would. It computes the HMC potential and its gradient once (jit `value_and_grad`), then `parts(x0)` once:
  - `potential(x) = parts(x)[0] - coords.log_jacobian(x[:nc]) - x[nc]` (copied from `laplace_fit.py:40-42`);
  - `parts(x0)`.
- **Coverage [E].** coverage.py 7.6.1 with branch coverage was unpacked privately and is not in the shared venv. It was limited to `/mnt/c/Dev/houfin-range-model/*`. The raw data is `executed_coverage.json`.
- **Validation [E].** The local `parts(x0)[0]` is 201,574.842184440 and TACC `result.json` reports 201,574.842184438 (difference 2e-9). Every component in `parts_info` matches `result.json`, so the measured path is the TACC path.
  - Potential: 201,703.579154.
  - log|J| = -128.9158; log σ = 0.17879.
  - Gradient: max |g| = 908.0, ‖g‖₂ = 2477.3, both at `manifold_factor`. The HMC init is not a stationary point.
  - Wall time: 61 s on 8 CPU threads, including compilation.
- **Import resolution [E].** `laplace_fit`, `harness`, `cbc_obs` and `specimen_lik` were imported from the **main repo** `research/exp/invasion_sandbox/`, because the script's own directory comes first on `sys.path`. `harness.py` is byte-identical to the worktree copy. `local_map_coordinates` and all of `src/` come from the **worktree**.
- **What "executed" means here [L].** JAX traces Python once per `jit`, so coverage shows the code traced into the computation graph. Branches on *traced values* are written with `jnp.where`/`clip`, so both arms are in the graph and coverage cannot tell them apart.
  - Where the spec depends on such a branch (Veit soft cap, `relu`-type maxima, gate clipping), MODEL_SPEC states the formula, not which arm is active.
  - Python-level `if` on config flags *is* resolved by coverage. Those decisions are tabulated in §3.
- **Not exercised [E].** The NUTS machinery in `run_hmc` (Hessian mass matrix, whitening, sampling) was patched out. Only the target density was evaluated.

## 2. Objective components at LA1 (measured, nats; `parts_info`)

| term | value | code |
|---|---|---|
| model log density excluding BBS obs, `-(lp - obs_model)` | (part of total) | `laplace_fit.py:673-678,764` |
| BBS route-marginal NB2 NLL `marginal_obs` (Laplace over 19,127 route effects) | 165,500.338 | `laplace_fit.py:638-659,686` |
| (BBS NB2 at route effect 0 removed from lp, `obs_model_eps0`) | 206,856.330 | `laplace_fit.py:674-678` |
| (BBS conditional at e*, `conditional_obs`) | 150,759.562 | info only |
| NestWatch `-nest_loglik` (inside lp) | 1,881.220 | `age_priors.py:2046` |
| CBC count NLL (Laplace over 1,823 circle effects) | 9,212.400 (+ its 4-parameter prior) | `laplace_fit.py:397-424,728-729` |
| BBL m-array NLL | 4,666.491 (+ the reporting-logit prior) | `laplace_fit.py:509-532,738-739` |
| specimen count NLL | 2,415.276 (+ nuisance priors) | `laplace_fit.py:752-762` |
| range-absence cost | 23.148 | `laplace_fit.py:711-714` |
| native_viable penalty | 2.341 | `laplace_fit.py:589-598,763` |
| σ prior `+½σ²` | 0.715 | `laplace_fit.py:616,690` |
| coupling_coef_prior (factor) | 30.565 (in lp) | trace site `coupling_coef_prior` |
| coupling_map_prior (factor) | 0.000 | trace site `coupling_map_prior` |
| local_age_structure_regularizer (factor) | −0.076 (in lp, i.e. +0.076 log density) | trace site |
| observer_route_eps ZeroSumNormal log prob at e = 0 (constant) | 17,575.618 (in lp; constant) | trace site `observer_route_eps` |
| presence / range presence / establishment / trough / natal / CBC-rate / depression terms | 0 (flags absent) | |

The full numpyro log density at x0, including the BBS `obs` site at e = 0, is −226,603.354 [E].

## 3. Python-level switches: what the trace resolved [E]

Config flags as resolved at import (`age_priors` module globals). Sources: `eval_result.json:ap_flags` and the coverage untaken-arm lists below.

| switch (config key = value) | executed branch | code |
|---|---|---|
| `manifold_prior.form = capacity_demography` | `_capacity_demography_w_env` (capacity factor sampled, `capacity_correlation_logit` sampled) | `age_priors.py:931-932, 772-809` |
| `readout.split = true`, `within_shared = "demographic"` | `w_env_within = w_scale * w_env[:,0]/w_scale[0]` | `age_priors.py:950-962` |
| `readout.rank = 64` | Z truncated to 64 columns (no change: input is 64) | `data_loading.py:457-474` |
| `habitat_centering = occupied` | `z_center = habitat_center_occupied` | `age_priors.py:1515-1519` |
| `juvenile_survival.form = ratio` | `Sj = Sa*sigmoid(.)`; `Q = sigmoid(alpha_a+.)*sigmoid(alpha_j+.)` | `age_fields.py:325-326, 400-404` |
| `capacity_level_prior.link = softplus`, `form = field` | `K_base = softplus(arg)` | `age_fields.py:359` |
| `k_trend.enabled = false` | `w_k_trend` deterministic zeros | `age_priors.py:1040-1043` |
| `disease_prior.target = survival` | K disease-free; `disease_mortality_amp` deterministic; added mortality in forward | `age_fields.py:367-368`; `age_priors.py:1716-1720`; `age_forward.py:744-751` |
| `disease_prior.onset_gate.form = compact_start` | smoothstep gate; `disease_tau` constant 1.0 (no `disease_tau_raw` site) | `disease_onset.py:63-77`; `age_priors.py:1168-1178` |
| `disease_prior.recovery.form = exponential` | `rec*(1-exp(-age/tau_rec))` | `disease_onset.py:54-57` |
| `allee_prior.form = absolute`, `aggregation_k = 3`, `per_capita = true` | `allee_gamma = gamma_raw*pop_scalar`; mated fraction `1-(1+x/3)^-4` | `age_priors.py:1506, 361-363`; `age_forward.py:1114-1117` |
| `pinned_sites = {n50_raw, log_inv_total}` | `n50_raw` conditioned (observed site); `log_inv_total` deterministic 0 | `age_priors.py:1412-1413, 1477-1478` |
| `invasion_pulse_prior.form = single`, `n_sites = 1`, `n_years = 1` | one release cell (27,89), year index `inv_timestep` = 49 (1951) | `age_priors.py:1465-1485` |
| `dispersal_departure.form = veit` | Veit departure overrides the logistic; `dispersal_logit_*` excluded from coordinates | `age_forward.py:333-339`; `laplace_fit.py:269` |
| `native_departure.enabled = true` | `departure_scale = 1 - r*w_nat` | `age_priors.py:1845-1849` |
| `reproduction = standard, yearling_breeding, crowding_power 2` | `N_j,new = N_a,new*F_actual`; crowding `(N/K)^2` | `age_forward.py:1213-1216`; `density_regulation.py:9-16` |
| `adult_survival_dd.enabled = true` | `dd_survival_share` sampled; `survival_dd_c` passed to forward | `age_priors.py:1797-1803`; `age_forward.py:1189-1190` |
| `spinup.enabled = true (40 y, habitat_years 5)` | two forward calls: spin-up (time 40, `inv_t` = 1040) then main (time 124) | `age_priors.py:1970-1987`; `forward_calls.json` |
| `observation.form = breeding` | `obs_fledgling_weight` sampled; breeders = `densities[t-1]` | `age_priors.py:2061-2066` |
| `observation.observer_effect.enabled = true, sigma_mode = fixed` | `observer_first_year_eta` sampled; `observer_route_eps` ZeroSumNormal site (fixed at 0 by laplace_fit) | `age_priors.py:2090-2093, 230-286` |
| `observation.conc_density_power = true` | `obs_conc_kappa` sampled; `phi = conc*q^tier*max(mu,1e-3)^kappa` | `age_priors.py:2139-2142` |
| `n_obs_quality_tiers = 2` (data) | `quality_conc_mult` sampled | `age_priors.py:2130-2132` |
| `observation_mask` = only a `_pinned_comment` key | `obs_weight` = 1 for all 191,895 rows (`MaskedDistribution` with an all-true mask) | `data_loading.py:705-719` |
| `nest_likelihood.enabled = true`, `broods.form = pinned` | nest factor in lp; `nest_mean_delta` / `nest_surv_delta` / `nest_dsr_logit` / `nest_log_conc` sampled | `age_priors.py:2015-2046` |
| `numerics = {fft_pad 0, fft_floor_ulps 64, fft_floor_smooth true}` | smooth FFT floor active in float64 (`fft_floor_float32_only` absent → false) | `age_priors.py:351-360`; `age_forward.py:109-129` |
| `retention_overflow.mode = veit` | `OVERFLOW_ENABLED` false | `age_forward.py:18-19` |
| `cbc_likelihood.enabled = false` | model-internal CBC block skipped (CBC enters via `laplace_fit --cbc_counts`) | `age_priors.py:2166-2169` |
| laplace_fit flags | `ccnt` (latent, BBL source, `npar` 4), `bbl` (until 2025, disease), `absn`, `spc` (counts), `nviab` active; `pres`, `estab`, `crate`, `dd`, `saa`, `lpc`, `rprs`, `nat`, `trough`, `ywg`, `route_skew`, `route_mean` None/False | `eval_result.json:args`, `pres_is_None` |

## 4. Config keys present but INACTIVE for LA1 [E unless marked]

| key = value | why inactive | evidence |
|---|---|---|
| `population_model.capacity_floor_n50_multiple` = 8.0 (local) / 20.0 (TACC) | not read by any `src/` module or laplace_fit; the two overlays differ only here (plus `input_dir`) and the objectives agree | grep `src/`; overlay diff |
| `allee_prior.n50_raw_loc` = −3.4915, `n50_raw_scale` = .25 | `n50_raw` pinned → its log prob is a constant (−0.0931) | trace `n50_raw` observed |
| `invasion_pulse_prior.total_median`, `total_log_sd`, `median_fraction_of_global_q50`, `log_budget` | `log_inv_total` pinned; form single | `age_priors.py:1477-1478` |
| `release_allee_exemption_years` = 0 | 0 → no exemption | forward arg `allee_exempt_years` = 0 |
| `dispersal_target_capacity_fraction` = .8 (metadata) and the `dispersal_logit_intercept` / `dispersal_logit_slope` sites (held at 2.0 / 4.0) | Veit departure overrides the logistic `p_total` / `p_adult` | `age_forward.py:333-339` |
| `dispersal_random_enabled` = false | zeros | `age_priors.py:1383-1387` |
| `manifold_prior.coupling_target`, `coupling_logit_scale`, `target_corr_*`, `communality_logit_scale`, `angle_scale`, `fixed_coupling` = 1.0 | used only by the rank2 / exchangeable forms | `age_priors.py:867-934` |
| `disease_prior.severity_ceiling` = .5 | target = survival uses `mortality_ceiling` .8 | `age_priors.py:162-163` |
| `disease_prior.recovery.delay_*`, `ceiling_prior` (absent) | exponential recovery, fixed ceiling | |
| `observation.observer_effect.sigma_u` = .97, `sigma_mode` = fixed, `sigma_prior_scale` = 1.0 | laplace_fit zeroes `observer_route_eps` and integrates its own route effects with free σ; `sigma_u` only rescales the saved `observer_route_eps`. `sigma_prior_scale` IS used, as σ's HalfNormal scale | `laplace_fit.py:242-243, 264, 616, 973` |
| `k_range.max_fold_deviation` = 26 | reporting only (not read in src/model) | grep |
| `k_trend.n_basis`, `budget` | `enabled` false | |
| `trend_routing.*` | `enabled` false | |
| `settlement.form` = none, `dispersal_kernel.band_weights` = false | off | forward args `settle_block` 0, `juvenile_kernel_weights` None |
| `nest_likelihood.broods.log_sd` = .3 | broods pinned | |
| `readout.within_ratio_fixed`, `within_corr`, `within_tied`, `within_ratio_prior`, `feature_columns` = null/false | `within_shared` branch taken first | `age_priors.py:950` |
| `cbc_likelihood.*` (`file`, `circle_sd`, …) | `enabled` false | |
| `map.*`, `run_names.*`, `runtime.input_residency`, `path_features.*`, `path_diagnostics_dir`, `community_bridge.*`, `ebird_dir`, `results_dir`, `bbs_npz`, `ocean_mask`, `disease_arrival_map`, `raw_z_dir`, `pseudo_zero_subsample`, `invasion_lat` / `invasion_lon` / `invasion_sites` / `invasion_window_years` | not read at evaluation time. Ingest-time keys are already baked into `metadata.pkl` (e.g. `inv_window` 10, `inv_locations` 4×2); `runtime.input_residency` is superseded by env `HOUFIN_MODEL_INPUT_RESIDENCY` | `data_loading.py:814`; DESK_PROVENANCE.md |
| `latent_dim` / `source_latent_dim` = 64, `kernel_contract` | validated, not a switch (`validate_model_feature_contract`) | `age_priors.py:609-616` |
| all `_..._comment`, `_pathology_candidate`, `_later_established_state`, `_cbc_established`, `_overflow_early`, `_crowding_curvature_experiment`, `_floor_combination`, `_final`, `_leader`, … | comment/record keys, never read | |
| laplace_fit `--presence_k` default .43724 | **active** (absence cost k); listed here because it is a default, not a flag on the command line | `laplace_fit.py:160, 714` |

**Switch set but with no effect [E].** Three cases:
1. `capacity_floor_n50_multiple` differs between the local (8) and TACC (20) overlays and changes nothing.
2. `observer_effect.sigma_u` (.97) looks like the route-effect scale but does not enter the objective.
3. `fixed_coupling` 1.0 is ignored by the capacity_demography form.

## 5. Per-module coverage (generated from `executed_coverage.json`)

Function `name:line (executed body lines / measurable body lines)`. "Branch points with an untaken arm" lists source lines where one arm of a Python `if` never ran. Most are the inactive switches above.

| module | lines covered/statements | branches covered/total |
|---|---|---|
| `research/exp/invasion_sandbox/cbc_obs.py` | 44/48 | 11/14 |
| `research/exp/invasion_sandbox/harness.py` | 72/296 | 9/120 |
| `research/exp/invasion_sandbox/laplace_fit.py` | 401/826 | 92/272 |
| `research/exp/invasion_sandbox/specimen_lik.py` | 22/74 | 0/18 |
| `.worktrees/route-variance/research/exp/local_map_coordinates.py` | 34/38 | 13/16 |
| `.worktrees/route-variance/src/__init__.py` | 0/0 | 0/0 |
| `.worktrees/route-variance/src/community_encoder/__init__.py` | 0/0 | 0/0 |
| `.worktrees/route-variance/src/community_encoder/cube_contract.py` | 30/109 | 11/64 |
| `.worktrees/route-variance/src/config_utils.py` | 66/91 | 16/38 |
| `.worktrees/route-variance/src/model/__init__.py` | 0/0 | 0/0 |
| `.worktrees/route-variance/src/model/age_fields.py` | 82/142 | 19/42 |
| `.worktrees/route-variance/src/model/age_forward.py` | 217/548 | 53/154 |
| `.worktrees/route-variance/src/model/age_priors.py` | 532/903 | 126/340 |
| `.worktrees/route-variance/src/model/checkpoints.py` | 19/30 | 8/14 |
| `.worktrees/route-variance/src/model/data_loading.py` | 198/594 | 62/303 |
| `.worktrees/route-variance/src/model/density_regulation.py` | 8/11 | 3/6 |
| `.worktrees/route-variance/src/model/disease_onset.py` | 29/46 | 8/24 |
| `.worktrees/route-variance/src/model/dispersal_spec.py` | 30/70 | 16/47 |
| `.worktrees/route-variance/src/model/growth_kernel.py` | 34/161 | 5/38 |
| `.worktrees/route-variance/src/model/habitat_geometry.py` | 10/14 | 0/0 |
| `.worktrees/route-variance/src/model/hmc_common.py` | 68/266 | 2/82 |
| `.worktrees/route-variance/src/model/priority_overflow.py` | 7/25 | 0/4 |

### `research/exp/invasion_sandbox/cbc_obs.py`  (lines 44/48, branches 11/14)

Executed: `hav`:26 (3/3), `kappa_from_bearings`:32 (4/6), `movement_matrix`:42 (18/18), `build_source_matrix`:64 (11/12)

Not executed: `cbc_source_abundance`:81

Branch points with an untaken arm (3): 35, 37, 67

### `research/exp/invasion_sandbox/harness.py`  (lines 72/296, branches 9/120)

Executed: `Sandbox.__init__`:222 (40/42)

Not executed: `capture`:41, `capture.record`:43, `edit_habitat`:58, `edit_args`:67, `region_masks`:133, `time_profile`:138, `band_ramp_weights`:152, `install_floor`:170, `install_floor.wrapped`:175, `c_of`:185, `potential_lambda`:192, `check_c_formula`:198, `to_land`:205, `Sandbox.observe`:271, `Sandbox.run`:295, `Sandbox.run.era_mean`:385, `main`:399

Branch points with an untaken arm (57): 51, 53, 61, 69, 72, 74, 79, 81, 83, 85, 90, 92, 94, 98, 101, 104, 105, 107, 110, 111, 113, 116, 117, 120, 122, 124, 142, 145, 147, 163, 178, 182, 200, 208, 211, 213, 227, 247, 277, 303, 305, 310, 311, 316, 327, 331, 350, 351, 360, 361, 378, 380, 383, 394, 406, 413, 419

### `research/exp/invasion_sandbox/laplace_fit.py`  (lines 401/826, branches 92/272)

Executed: `main`:117 (395/759), `main.cbc_count_cost`:397 (22/23), `main.cbc_count_cost.ll`:410 (2/2), `main.cbc_count_cost.step`:417 (2/2), `main._land_sa`:505 (2/2), `main.bbl_cost`:509 (21/23), `main.native_viable_cost`:589 (9/9), `main.native_viable_cost.L`:590 (2/2), `main.row_ll`:618 (3/3), `main.route_prior`:624 (4/6), `main.laplace`:638 (16/18), `main.laplace.step`:646 (4/4), `main.parts`:661 (64/99), `main._resume`:783 (3/5)

Not executed: `run_hmc`:24, `run_hmc.potential`:40, `main.wiggle`:311, `main.cbc_rate_cost`:426, `main.cbc_rate_cost.ll`:432, `main.cbc_rate_cost.step`:438, `main.depression_cost`:477, `main.late_cell_cost`:541, `main.trough_cost`:608, `main.trough_cost.L`:609, `main.save`:949, `main.fun`:994, `main.callback`:1003

Branch points with an untaken arm (124): 35, 45, 62, 64, 65, 68, 75, 78, 80, 82, 90, 110, 245, 248, 260, 279, 283, 299, 317, 327, 329, 335, 343, 355, 358, 377, 386, 394, 400, 435, 436, 447, 471, 482, 484, 490, 499, 511, 513, 520, 524, 535, 543, 546, 551, 557, 564, 572, 583, 601, 629, 633, 654, 676, 681, 687, 692, 698, 712, 715, 718, 722, 728, 730, 733, 735, 738, 740, 745, 753, 756, 774, 775, 777, 780, 781, 790, 792, 794, 803, 807, 817, 821, 824, 825, 830, 837, 838, 840, 844, 848, 852, 853, 870, 873, 880, 890, 891, 896, 910, 917, 920, 924, 926, 933, 935, 952, 955, 959, 962, 965, 976, 978, 980, 982, 984, 986, 989, 996, 999, 1005, 1007, 1015, 1034

### `research/exp/invasion_sandbox/specimen_lik.py`  (lines 22/74, branches 0/18)

Executed: `g_detect`:96 (1/1), `specimen_nll_counts`:100 (9/9)

Not executed: `cell_index_of_points`:39, `build_cache`:55, `specimen_nll`:116

Branch points with an untaken arm (9): 46, 67, 70, 72, 73, 74, 83, 87, 89

### `.worktrees/route-variance/research/exp/local_map_coordinates.py`  (lines 34/38, branches 13/16)

Executed: `MapCoordinates.__init__`:14 (21/24), `MapCoordinates.constrain`:41 (2/2), `MapCoordinates.log_jacobian`:45 (2/2)

Not executed: `MapCoordinates.describe`:50

Branch points with an untaken arm (3): 20, 32, 36

### `.worktrees/route-variance/src/__init__.py`  (lines 0/0, branches 0/0)

Executed: (module level only)

Not executed: -

### `.worktrees/route-variance/src/community_encoder/__init__.py`  (lines 0/0, branches 0/0)

Executed: (module level only)

Not executed: -

### `.worktrees/route-variance/src/community_encoder/cube_contract.py`  (lines 30/109, branches 11/64)

Executed: `validate_cube_contract`:63 (14/19), `validate_model_feature_contract`:129 (6/18)

Not executed: `artifact_identity`:7, `add_basis_identities`:16, `export_contract`:32, `compatible_contract`:89, `select_block_prefixes`:97

Branch points with an untaken arm (30): 10, 11, 19, 23, 25, 26, 38, 40, 43, 52, 66, 69, 72, 74, 77, 84, 101, 104, 108, 111, 112, 114, 120, 132, 135, 137, 139, 140, 145, 147

### `.worktrees/route-variance/src/config_utils.py`  (lines 66/91, branches 16/38)

Executed: `resolve_repo_root`:26 (1/1), `config_dir`:31 (1/1), `_ensure_roots`:44 (3/3), `_expandvars`:51 (7/7), `_load_json`:66 (2/2), `_load_overlay`:71 (8/18), `load_config`:98 (9/9), `load_age_model_config`:126 (1/1), `_deep_merge`:133 (6/6), `load_data_config`:144 (1/1)

Not executed: `load_secrets`:158, `get_secret`:179, `target_points_dir`:197

Branch points with an untaken arm (13): 80, 81, 86, 90, 92, 118, 120, 168, 170, 173, 175, 190, 192

### `.worktrees/route-variance/src/model/__init__.py`  (lines 0/0, branches 0/0)

Executed: (module level only)

Not executed: -

### `.worktrees/route-variance/src/model/age_fields.py`  (lines 82/142, branches 19/42)

Executed: `disease_severity`:38 (4/4), `disease_mortality_amplitude`:100 (6/6), `disease_onset_timestep`:123 (1/1), `project_and_scatter_age_structured`:158 (58/112), `project_and_scatter_age_structured.process_year`:233 (42/94)

Not executed: `disease_k_fraction`:137

Branch points with an untaken arm (20): 213, 221, 222, 237, 245, 264, 295, 300, 303, 304, 313, 325, 335, 353, 367, 389, 400, 405, 416, 423

### `.worktrees/route-variance/src/model/age_forward.py`  (lines 217/548, branches 53/154)

Executed: `clip_safe`:24 (1/1), `rightpad`:29 (3/3), `rightpad_convolution`:109 (10/11), `juvenile_dispersal_vectorized`:203 (12/30), `juvenile_dispersal_vectorized.single_kernel_prop`:229 (3/3), `departure_logit`:271 (3/4), `dispersal_step_age_structured`:280 (30/38), `forward_sim_age_structured`:383 (88/209), `forward_sim_age_structured.scatter_t`:592 (7/7), `forward_sim_age_structured.step`:605 (51/158), `allee_mated_fraction`:1105 (3/4), `reproduction_age_structured`:1147 (21/34)

Not executed: `overflow_probabilities`:20, `_positive_part`:60, `occupied_density`:78, `pairing_fraction`:87, `juvenile_band_weights`:134, `land_cross_correlation`:152, `attraction_weights`:161, `attraction_ratio`:170, `directional_choice_weights`:182, `forward_sim_age_structured._um_rho`:566, `ideal_free_settle`:886, `_to_blocks`:920, `_from_blocks`:926, `fitness_settle`:931, `fitness_settle.density`:967, `fitness_settle.bisect`:980, `subcell_reproduction`:1000, `subcell_reproduction.tilt`:1027, `subcell_reproduction.alloc`:1060, `subcell_reproduction.bisect`:1073, `relative_density`:1126

Branch points with an untaken arm (74): 62, 101, 119, 121, 234, 236, 242, 244, 248, 250, 267, 275, 333, 340, 342, 344, 350, 354, 539, 550, 553, 558, 559, 580, 607, 618, 621, 652, 671, 683, 688, 693, 696, 698, 701, 705, 712, 721, 730, 732, 744, 752, 756, 768, 773, 775, 797, 799, 818, 822, 824, 827, 836, 846, 856, 864, 882, 917, 971, 988, 997, 1036, 1038, 1064, 1081, 1115, 1177, 1183, 1186, 1189, 1196, 1206, 1209, 1213

### `.worktrees/route-variance/src/model/age_priors.py`  (lines 532/903, branches 126/340)

Executed: `observer_effect_spec`:213 (5/7), `observer_log_offset`:230 (16/22), `_softplus_loc_for_median`:531 (3/4), `counts_to_relative`:554 (1/1), `k_link`:580 (1/1), `validate_environment_kernel_contract`:609 (5/6), `age_structure_log_prior`:619 (2/2), `nest_log_likelihood`:649 (10/10), `_nest_site`:669 (2/3), `equilibrium_age_quantities`:683 (13/30), `_capacity_demography_w_env`:772 (17/22), `sample_priors`:812 (95/156), `build_model_2d`:1392 (191/383), `build_model_2d._run_forward`:1922 (2/2)

Not executed: `_trend_eta`:62, `_kdisc_D`:83, `_hat_basis`:202, `route_occupancy_parts`:289, `crowding_capacity`:585, `_exchangeable_w_env`:725

Branch points with an untaken arm (161): 67, 73, 75, 91, 116, 123, 132, 141, 159, 223, 225, 261, 268, 273, 274, 280, 282, 309, 325, 339, 346, 352, 420, 448, 516, 518, 520, 549, 571, 573, 613, 632, 635, 638, 640, 643, 645, 670, 677, 691, 699, 702, 710, 713, 720, 744, 755, 757, 758, 781, 782, 787, 790, 867, 919, 920, 921, 923, 931, 950, 963, 966, 968, 972, 980, 985, 999, 1001, 1005, 1040, 1149, 1193, 1219, 1328, 1339, 1351, 1383, 1465, 1477, 1515, 1522, 1529, 1534, 1541, 1545, 1551, 1557, 1560, 1561, 1571, 1577, 1578, 1586, 1587, 1595, 1598, 1603, 1607, 1609, 1611, 1618, 1620, 1666, 1669, 1678, 1681, 1682, 1689, 1693, 1698, 1702, 1707, 1716, 1725, 1731, 1736, 1737, 1797, 1798, 1814, 1827, 1830, 1834, 1839, 1845, 1850, 1854, 1855, 1869, 1870, 1888, 1898, 1901, 1906, 1916, 1970, 1971, 1981, 1982, 1983, 1989, 1991, 1995, 2003, 2011, 2015, 2020, 2027, 2061, 2073, 2080, 2091, 2094, 2097, 2101, 2128, 2130, 2139, 2149, 2158, 2167

### `.worktrees/route-variance/src/model/checkpoints.py`  (lines 19/30, branches 8/14)

Executed: `load_map_params`:18 (7/10), `auto_delta_params_to_latents`:35 (6/7)

Not executed: `save_pickle_atomic`:8

Branch points with an untaken arm (5): 11, 23, 28, 30, 40

### `.worktrees/route-variance/src/model/data_loading.py`  (lines 198/594, branches 62/303)

Executed: `occupied_habitat_center`:21 (9/11), `_habitat_shrink_enabled`:44 (3/3), `_route_effects_enabled`:50 (3/3), `_observer_effect_enabled`:56 (3/3), `readout_spec`:65 (2/2), `covariate_drift_spec`:71 (3/3), `_extra_pseudo_zeros_enabled`:113 (3/3), `_subcell_enabled`:125 (3/3), `_observation_weighting_spec`:172 (3/3), `_observation_mask_spec`:217 (3/3), `nest_covariate_matrix`:366 (5/6), `nest_arrays`:378 (14/16), `load_data`:415 (111/347), `load_data._to_device`:831 (8/9)

Not executed: `covariate_drift_basis`:78, `covariate_drift_ramp`:101, `_extra_pseudo_zeros_file`:119, `_subcell_moderator`:131, `observation_weights`:178, `holdout_observation_mask`:224, `process_noise_basis`:262, `state_anchor_arrays`:285, `process_noise_groups`:309, `process_noise_detrend`:324, `transient_observation_weight`:337, `load_data_to_gpu`:870

Branch points with an untaken arm (144): 32, 39, 91, 95, 142, 146, 147, 150, 152, 153, 157, 162, 164, 167, 190, 195, 197, 202, 239, 242, 245, 247, 250, 277, 280, 281, 291, 293, 314, 316, 372, 389, 408, 448, 451, 457, 462, 464, 470, 472, 475, 482, 488, 490, 494, 495, 496, 501, 504, 505, 508, 510, 513, 520, 522, 524, 525, 527, 528, 529, 536, 540, 544, 550, 554, 556, 560, 564, 567, 571, 578, 583, 585, 589, 592, 601, 605, 609, 613, 616, 619, 622, 625, 628, 633, 635, 640, 644, 650, 652, 656, 659, 667, 670, 673, 682, 688, 689, 690, 691, 696, 697, 698, 701, 706, 708, 711, 713, 717, 721, 724, 728, 733, 735, 739, 742, 746, 747, 751, 757, 758, 760, 762, 763, 768, 774, 782, 793, 801, 804, 807, 808, 809, 812, 815, 818, 832, 846, 848, 851, 855, 858, 859, 863

### `.worktrees/route-variance/src/model/density_regulation.py`  (lines 8/11, branches 3/6)

Executed: `crowding_argument`:9 (4/6)

Not executed: -

Branch points with an untaken arm (3): 5, 12, 14

### `.worktrees/route-variance/src/model/disease_onset.py`  (lines 29/46, branches 8/24)

Executed: `resolve_onset_spec`:8 (9/16), `resolve_recovery_spec`:27 (6/11), `recovery_fraction`:54 (2/5), `onset_gate`:63 (5/7)

Not executed: -

Branch points with an untaken arm (12): 11, 14, 16, 19, 21, 41, 44, 48, 49, 56, 71, 74

### `.worktrees/route-variance/src/model/dispersal_spec.py`  (lines 30/70, branches 16/47)

Executed: `_validate_splits`:50 (4/7), `resolve_radial_splits`:61 (2/14), `dispersal_spec`:140 (10/14), `require_inputs_match_config`:184 (4/7)

Not executed: `get_gamma_scale`:22, `get_dispersal_quantiles`:32, `kernel_discretization`:110

Branch points with an untaken arm (19): 44, 52, 54, 56, 84, 87, 89, 91, 94, 129, 131, 135, 157, 164, 168, 171, 196, 200, 201

### `.worktrees/route-variance/src/model/growth_kernel.py`  (lines 34/161, branches 5/38)

Executed: `spec_from_config`:73 (10/13)

Not executed: `n_density_terms`:89, `n_relative_terms`:93, `_knots`:97, `bspline_basis`:103, `density_basis`:125, `relative_basis`:134, `softcap`:140, `rates_at`:145, `year_slice`:159, `sample_growth_kernel`:165, `coefficient_fields`:193, `coefficient_fields.year`:207, `low_density_rates`:227, `lambda_on_ray`:237, `equilibrium_readout`:252, `equilibrium_readout.interp`:270

Branch points with an untaken arm (18): 80, 82, 84, 110, 112, 115, 117, 128, 129, 152, 171, 182, 198, 203, 210, 222, 243, 247

### `.worktrees/route-variance/src/model/habitat_geometry.py`  (lines 10/14, branches 0/0)

Executed: `capacity_demography_loadings`:9 (4/4), `capacity_demography_coefficients`:27 (2/2)

Not executed: `survival_breeding_loadings`:17

### `.worktrees/route-variance/src/model/hmc_common.py`  (lines 68/266, branches 2/82)

Executed: `split_data`:80 (3/3), `array_model`:95 (3/3), `array_model.wrapped`:101 (1/1)

Not executed: `load_posterior_config`:60, `setting`:64, `results_dir`:70, `model_kwargs`:75, `file_identity`:108, `fingerprint`:114, `source_identities`:119, `versions`:125, `check_model_drift`:129, `map_dir`:159, `posterior_dir`:185, `laplace_dir`:198, `cast_latents`:206, `load_map`:218, `hide_deterministics`:227, `hide_deterministics.blocked`:235, `init_values_else_mean`:241, `potential_with_args`:258, `potential_with_args.potential`:272, `potential_with_args.postprocess`:275, `unconstrained_setup`:280, `flatten_sorted`:298, `flatten_sorted.unravel`:308, `dense_hessian`:313, `dense_hessian.f`:331, `dense_hessian.hvp_batch`:337, `laplace_inverse_mass`:368, `load_laplace`:398, `jitter_unconstrained`:405, `_host`:430, `_grab`:434, `run_chunked_mcmc`:440, `_log_extra`:512, `archive_if_fresh`:523, `load_chain`:535, `chain_index`:549

Branch points with an untaken arm (40): 26, 32, 143, 145, 147, 150, 169, 175, 180, 193, 201, 236, 251, 253, 306, 337, 338, 342, 344, 349, 352, 357, 361, 400, 412, 416, 431, 454, 464, 465, 467, 489, 498, 525, 527, 538, 541, 542, 544, 546

### `.worktrees/route-variance/src/model/priority_overflow.py`  (lines 7/25, branches 0/4)

Executed: (module level only)

Not executed: `smooth_retained`:11, `priority_overflow`:20, `overflow_probability`:31, `departure_probabilities`:46

Branch points with an untaken arm (2): 48, 51

# Proposed docstring and code-comment patches (for the lead to apply)

**Scope.** No Python source was edited. Each row gives the location at the audited commits, an
excerpt of the current text, and a proposed replacement.
- Commits: worktree `route-variance` @ `0f73b31`, and main repo `laplace_fit.py` / `specimen_lik.py`
  @ `dee075d`. Line numbers were re-checked against the working copies on 2026-11-08.
- Rows A1–A25 follow `DOC_DISCREPANCIES.md` §5. X1 is an extra stale docstring found while
  checking T3.
- Appendix B covers comment keys that live only in the experiment overlay (out of scope here).
- Appendix C covers the data-config timeline comment.

**Proposed text and symbols.** The proposed text uses the symbols of `NOTATION.md` with the
renames of `METHODS_DRAFT.md` §13. In code comments, plain ASCII names are fine (`sigmoid` for
expit).

**What is correct for which config.** Several docstrings describe the code's *defaults* correctly
(rank2, target = capacity, k = None, a logistic departure). They mislead only for the candidate's
configuration. The proposals keep the default description and add the switch that changes it,
rather than describing only the candidate.

**Path abbreviations.** `AF` = `src/model/age_fields.py`, `AP` = `src/model/age_priors.py`,
`AFW` = `src/model/age_forward.py` (worktree), `LF` = main-repo
`research/exp/invasion_sandbox/laplace_fit.py`, `SL` = main-repo
`research/exp/invasion_sandbox/specimen_lik.py`.

---

## A. Executed functions

### A1 — AF:9-26 (module docstring)
**Current (excerpt).**
- "K_base = softplus(alpha_k + gamma_k*H_k + k_trend_t)"
- "measured 2.6183 route counts"
- "Each rate reads its OWN manifold … The four manifolds share a RANK-2 prior … Corr(F_max, K) ~ 0.85 sits above Corr(F_max, survival) ~ 0.70"

**Proposed.**
> K uses a SOFTPLUS link in DENSITY space: `K_base = softplus(alpha_k + H_k [+ k_trend_t])`, then `K = K_base + capacity_floor_internal` when the floor is set. The slope gamma_k is fixed at 1. ``alpha_k`` is solved so the prior MEDIAN of K_base equals `capacity_level_prior.target_level_median_route_counts` (2.6183 in the base config; 2.582 on the 54 km grid). The coefficient structure depends on `manifold_prior.form`:
> - `rank2` / `exchangeable`: each rate reads its own coefficient vector under a correlated prior.
> - `capacity_demography` (current candidate): S_a, F_max and S_j share ONE direction f (correlation exactly 1), and `beta_K = w_K*(r_K*f + sqrt(1-r_K^2)*g)`.
>
> See ``age_priors.sample_priors`` and docs/PRIORS.md#manifold-prior.

### A2 — AF:190-203 (`project_and_scatter_age_structured` docstring)
**Current (excerpt).**
- "Q … reuses the juvenile intercept/slope on … H_sj_disp"
- "``Kbase_flat`` is capacity BEFORE the disease effect"
- "Timesteps before ``disease_timestep`` get K_base unmodified … which is what pins ``k_level``"

**Proposed.**
> Project Z → (S_a, S_j, F_max, K, Q) for every year, on the land cells. Under `juvenile_survival.form = ratio`:
> - `S_j = S_a * sigmoid(alpha_j + H_sj)`;
> - `Q_k = sigmoid(alpha_a + H_Sa^disp,k) * sigmoid(alpha_j + H_Sj^disp,k)` on the path features.
>
> ``Kbase_flat`` is capacity before the additive floor `capacity_floor_internal`; under `disease_target = capacity` it is also before the disease effect. Under `disease_target = capacity`, timesteps before ``disease_timestep`` get K_base undepressed. Under `disease_target = survival`, K has no disease term at any step, and the epizootic acts as added mortality in the forward simulation. K adds the floor at every step.

### A3 — AF:252-256 (comment)
**Current (excerpt).** "now through a PRIOR on their correlation (rank-2 manifold prior, target 0.85) rather than by identity."

**Proposed.**
> Juvenile survival has its own coefficient column. Under `rank2` / `exchangeable` it is prior-coupled to adult survival (target 0.85). Under `capacity_demography` (current candidate) its coefficients equal adult survival's up to amplitude (correlation exactly 1).

### A4 — AF:339-343 (comment)
**Current (excerpt).** "K_base = softplus(alpha_k + gamma_k*H_k + trend) … the measured 2.6183 route counts"

**Proposed.**
> Baseline K with a SOFTPLUS link in DENSITY space: `K_base = softplus(alpha_k + H_k [+ trend])`, with gamma_k fixed at 1. alpha_k's location is solved so the prior MEDIAN of K_base is `target_level_median_route_counts` (2.6183 at 27 km, 2.582 at 54 km). The floor `capacity_floor_internal` is added below, after any disease term.

### A5 — AF:38-81 (`disease_severity` docstring)
**Current (excerpt).**
- "Per-cell PEAK fraction of carrying capacity the epizootic removes … ``K = K_base * (1 - severity(K_base))``"
- "n~3 a sharp invasion threshold"
- "Severity keys off ``K_base`` … N does not exist yet"

**Proposed.** Prepend:
> Used by `disease_target = capacity` (K depression). Under `disease_target = survival` (current candidate) this function feeds only the diagnostic `disease_severity_map`. The applied added mortality is `disease_mortality_amplitude(...) * hill(N')`, with the Hill term on CURRENT post-dispersal density N', `mortality_ceiling` .8, and n bounded to (0.5, 2.55) (`age_forward.py:744-751`).

In the n-steepness sentence, replace "n~3 a sharp invasion threshold" with:
> larger n gives a sharper threshold; the attainable range depends on the ceiling's bound on n.

### A6 — AP:816-823 (`sample_priors` docstring)
**Current (excerpt).** "via a rank-2 angular prior … the continental time trend on K, the structured mycoplasmal-conjunctivitis effect on K"

**Proposed.**
> Covers the habitat coefficients of the four fields (adult survival, reproduction, capacity, juvenile survival) under `manifold_prior.form`: rank2, exchangeable, capacity_demography or survival_breeding. It also covers:
> - the optional continental time trend on K (`k_trend.enabled`);
> - the structured mycoplasmal-conjunctivitis term, which acts on K (target = capacity) or as added mortality (target = survival);
> - the dispersal and demography rate parameters;
> - the Allee term.
>
> ``prior_scale`` multiplies the scale parameters for continuation fitting.

### A7 — AP:1050-1064 (comments)
**Current (excerpt).**
- "MYCOPLASMAL-CONJUNCTIVITIS EFFECT ON K: K = K_base * (1 - …)"
- "median severity of 0.8 * 0.5 = 40%"

**Proposed.**
> MYCOPLASMAL-CONJUNCTIVITIS TERM. With target = capacity (default): `K = K_base * (1 - severity(x) * gate(x,t) * (1 - recovery))`, with severity ceiling `severity_ceiling` (.5), giving a prior-median severity of .8 * .5 = 40%. With target = survival (current candidate): added mortality `m = mortality_ceiling * sigmoid(mu + ...) * gate * (1 - recovery) * hill(N')` on overwinter survival, with prior-median peak .8 * .8 = 64% at saturating density. K is then disease-free.

### A8 — AP:1305-1314 (comment)
**Current (excerpt).** "At loc -2.0 (median n50 0.127 counts) the retained fecundity is 42% at 0.1 counts, 94% at 0.5, and 99.6% at one bird"

**Proposed.**
> With aggregation_k None (default) the factor is `1 - 2^(-C/n50)` (½ at C = n50); at loc -2.0 (median n50 .127 counts) retention is 42% at .1 counts, 94% at .5 and 99.6% at 1. With aggregation_k = 3 and per_capita (current candidate) it is `1 - (1 + ln2*C/(3*n50))^-4`, which is ½ at C = .819 n50. The candidate pins n50_raw = -3.756179 (n50 = .0231 counts), giving retention .32 at .01 counts, .80 at .05, .94 at .1 and .993 at .25. See docs/PRIORS.md#allee-effect.

### A9 — AP:1392-1399 (`build_model_2d` docstring)
**Current (excerpt).** "runs the age-structured forward simulation from the invasion year, and scores BBS counts with a negative-binomial (NB2) likelihood"

**Proposed.**
> runs the age-structured forward simulation from 1902 (t = 0), after an optional spin-up (`spinup.enabled`: 40 years at the 1902–06 mean rates, with no disease and no release); the release enters at ``inv_timestep``. It scores BBS counts with an NB2 likelihood, and adds the NestWatch log-likelihood (`nest_likelihood.enabled`) and the factors on derived quantities (`coupling_coef_prior`, `coupling_map_prior`, `local_age_structure_regularizer`).

The fitting driver `laplace_fit.py` adds further data terms; see docs/model_spec/MODEL_SPEC.md §5.

### A10 — AP:1450-1471 (comments)
**Current (excerpt).**
- "The 1940 release(s)"
- "One coefficient per (candidate site, year)"
- "(default 1 = 1940)"

**Proposed.** Replace "The 1940 release(s)" with:
> The release (calendar year = timeline invasion_year: 1940 in the base config, 1951 in the established1951 inputs).

Prefix the per-(site, year) paragraph with:
> form = grid (default):

In the form-single comment, replace "(default 1 = 1940)" with:
> (default 1 = the release year)

### A11 — AP:1770-1776 (comment)
**Current (excerpt).** "Since dispersal_target_capacity_fraction is 0.8, that puts the seed at N/K - 0.8 = +0.2 on the emigration logit, i.e. the native range is a NET EXPORTER from year one."

**Proposed.**
> Under the logistic departure (default), dispersal_target_capacity_fraction .8 puts the seed at +.2 on the emigration logit (a NET EXPORTER). Under `dispersal_departure.form = veit` (current candidate) that logit has no effect. Departure is then `min(N/K, 1)` (smoothed) × (1 - r_nat * w_nat), with r_nat ~ Beta(1.5, 1.5) when `native_departure.enabled`, so a near-1 r_nat makes the native core nearly sedentary. With `spinup.enabled`, this seed only starts the spin-up.

### A12 — AP:1794-1796 (comment)
**Current (excerpt).** "Its local linearized matrix is [[Sa, Sj], [F*Sa, 0]]. Thus fecundity at lambda=1 is (1-Sa)/(Sa*Sj)."

**Proposed.**
> With `reproduction.yearling_breeding` (current candidate) all winter survivors breed, so lambda = Sa + Sj*F and fecundity at lambda = 1 is (1 - Sa)/Sj. With adult_survival_dd, `Sa_K = Sa/(1 + c_s)` and `c_s = theta * max(Sa/clip(1 - Sj*Fmax, .05, 1) - 1, 0)`. Without yearling breeding the matrix is [[Sa, Sj], [F*Sa, 0]], with F = (1-Sa)/(Sa*Sj).

### A13 — AP:1784-1789 (comment on `disease_severity_map`)
**Current (excerpt).** "Per-cell peak severity at MODERN capacity -- the model's falsifiable claim about the epizootic."

**Proposed.**
> DIAGNOSTIC: `ceiling * sigmoid(...) * hill(K_base[last year])`. Under target = capacity it is the per-cell peak K depression. Under target = survival (current candidate) it is NOT the applied mortality, which uses hill(N') of the current post-dispersal density, times the onset gate and (1 - recovery).

### A14 — AP:683-690 (`equilibrium_age_quantities` docstring and comment)
**Current (excerpt).**
- "surviving adults reproduce, so the local projection matrix is ``[[Sa,Sj],[F*Sa,0]]``"
- "# default: 1 - exp(-x), as before"

**Proposed.** Docstring:
> Algebra matches :func:`reproduction_age_structured`. With yearling_breeding, lambda = Sa + Sj*F_at_K; otherwise the projection matrix is ``[[Sa,Sj],[F*Sa,0]]``.

Comment:
> # mated fraction: 1 - exp(-x) (aggregation_k None), else 1 - (1 + x/k)^-(k [+1 if per_capita])

### A15 — AP:236-250 (`observer_log_offset` docstring)
**Current (excerpt).**
- "u = sigma_u * eps, eps ~ ZeroSumNormal(1)"
- "sigma FIXED by default"
- "Zero-sum also removes the common-shift direction of u"

**Proposed.** Append:
> NOTE: the route-marginal driver (`research/exp/invasion_sandbox/laplace_fit.py`, used for the current candidate) holds this site at 0. It integrates iid N(0, 1) route effects e_r per route by a Laplace approximation, with sigma FREE under HalfNormal(`sigma_prior_scale`). `sigma_u` and the zero-sum constraint then play no role, and the common shift is held only by the N(0, 1) prior.

### A16 — AP:369-372 (config comment, `readout`)
**Current (excerpt).** "the cell's long-run DESK level z-bar and its within-cell deviation z(t) - z-bar get separate manifold weights and amplitudes"

**Proposed.** Append:
> `within_shared = "demographic"` overrides this: every field's deviation reads the adult-survival (demographic) direction f with that field's own amplitude (`w_scale_within = w_scale`). No separate within-cell weights or amplitudes are sampled.

### A17 — AFW:1105-1112 (`allee_mated_fraction` docstring)
**Current (excerpt).**
- "k > 0: … E[1 - exp(-gamma X)] = 1 - (1 + x/k)^(-k)"
- "birds do not experience the 27 km cell mean"

**Proposed.** Add:
> With ``ALLEE_PER_CAPITA`` (config `allee_prior.per_capita`) the exponent is -(k+1): the bird-weighted mated fraction. At k = 3 it is ½ at x = 3(2^{1/4} - 1) = .568, i.e. at C = .819 n50.

Replace "27 km" with:
> grid

### A18 — AFW:1157-1161 (`reproduction_age_structured` docstring)
**Current (excerpt).**
- "multiplied by a mate-finding Allee factor ``1 - exp(-allee_gamma * N)``"
- "the surviving adults produce the next juvenile cohort"

**Proposed.**
> Fecundity is density-dependent, `F_eff = F_max / (1 + c * (N/K)^crowding_power) * A(N)`, where A is ``allee_mated_fraction(allee_gamma * N)`` (see that function for the clumped / per-capita forms). Here N is ``N_total_post`` = adults after dispersal + juvenile stayers + juvenile arrivers already multiplied by (1 - m) in `forward_sim_age_structured` (age_forward.py:751); the disease Hill term instead uses the pre-disease total. Whether this asymmetry is intended is an open question for the user (METHODS_DRAFT §4, §12). With yearling_breeding, all winter survivors (surviving adults + surviving juvenile stayers + Q-surviving arrivers) breed: `N_j_new = N_a_new * F_eff`. Without it, only surviving adults reproduce, and the matrix is ``[[S_a, S_j], [F*S_a, 0]]``.

### A19 — AFW:311-317 (`dispersal_step_age_structured` docstring)
**Current (excerpt).** "Departure probability rises logistically with local density relative to K (``dispersal_logit_*``, ``target_fraction``)"

**Proposed.** Append:
> With ``veit_departure`` (current candidate) the logistic is replaced by the smoothed `p = clip(x - softplus(10(x - 1))/10, 1e-6, 1)`, `x = N/K`, multiplied by ``departure_scale`` (1 - r_nat * w_nat). The logit arguments then have no effect.

### A20 — AFW:465-492 (`forward_sim_age_structured` docstring)
**Current (excerpt).**
- "several nearby candidate cells and a per-(site, year) pulse"
- "The 1940 release … into a 27 km cell"

**Proposed.** Add after the candidate-site paragraph:
> With `invasion_pulse_prior.form = single` (current candidate) ``inv_pop`` has one site and one year: the release cell (27, 89) in 1951 (``inv_timestep`` 49), with no Allee exemption (``allee_exempt_years`` = 0).

Replace "27 km cell" with "grid cell", and "The 1940 release" with "The release".

### A21 — LF:715-716 (comment in `parts`)
**Current.**
```
if 'obs' in trace:   # the old pseudo-zeros are replaced: remove their contribution (rows with no route id)
    pass
```

**Proposed** (neutral while the user decision is pending).
```
# NOTE: no-op. The 1902-39 pseudo-zero BBS rows (no route id; 68,704 in the candidate) stay in the
# route-marginal NB2 term at route effect 0, IN ADDITION to this range-absence cost, unless
# --drop_pseudozeros gives them zero training weight. Whether to drop them is a PENDING user decision
# (docs/model_spec/METHODS_DRAFT.md §12).
```
If the decision is "drop", make `--drop_pseudozeros` the default in the HMC slot flags rather than editing this block.

### A22 — LF:14 (module docstring)
**Current.** "Everything else is Codex's posterior objective (newton_fit.py / local_joint_fit.py --objective posterior)."

**Proposed.** Append:
> Optional data terms added here (each behind a flag): CBC counts (`--cbc_counts`: latent effort, BBL migrant source, linear drift; circle effects Laplace-integrated), the BBL m-array (`--bbl`, `--bbl_disease`), museum-specimen counts (`--specimens --specimen_counts`), range absences (`--range_absence`, cost k_P*log(1 + sN/k_P)) and the native-viability penalty (`--native_viable`). The current candidate's objective uses all of these (docs/model_spec/MODEL_SPEC.md §5, §8).

### A23 — LF:194-195 (`--range_absence` help)
**Current.** "(full pre-1940 footprint + 1940-65 outside the native / introduced envelopes; rangev2 restoration)"

**Proposed.**
> npz of (t, land cell) absence constraints, t <= 1965 (e.g. `range_absence_halo400_nativeMX.npz`: 400 km halo, native-Mexico hull, release 1951; 74,306 cell-years). The cost is k_P*log(1 + sN/k_P) = -log NB2(0 | sN, k_P).

[L] The builder was not re-read; confirm the halo rule before applying.

### A24 — LF:209-210 (`--specimens` help)
**Current.** "effort-aware museum-specimen occupancy likelihood 1880-1940"

**Proposed.**
> effort-aware museum-specimen likelihood 1880-1940: occupancy (default) or NB2 counts per cell-decade with `--specimen_counts`.

### A25 — SL:10-19 (module docstring)
**Current (excerpt).**
- "MODEL … P(y_cd > 0) = 1 - exp(...)"
- "OCCUPANCY (detection) form, not a count"
- "One nuisance, prior Beta(2, 2) on beta"

**Proposed.** Add a section before MODEL:
> COUNTS VARIANT (``specimen_nll_counts``; used by the current candidate via `--specimen_counts`): `y_cd ~ NB2(lambda_cd, phi_S)` with `lambda_cd = max(exp(log s_e + beta*log E_cd) * g(N_cd), 1e-12)`. This adds a fourth nuisance, log phi_S ~ N(0, 2). In `laplace_fit.py` the prior on beta is Beta(2, 2): the MAP objective carries the Beta(2, 2) density on the logit coordinate without a Jacobian (MAP convention), and since the lead's 2026-11-08 fix HMC mode adds the logit Jacobian, so the HMC target is Beta(2, 2) on beta. (Before that fix the HMC target was uniform on beta.)

The occupancy rationale below argues for the binary form; it should be kept, but marked as the default and not the form the candidate runs.

---

## X. Extra stale docstring (not in DOC_DISCREPANCIES)

### X1 — `src/temporal.py:116-123` (`disease_timestep` docstring)
**Current (excerpt).** "The K-depression term (see ``src/model/age_fields.py``) is identically zero before this index"

**Proposed.**
> The epizootic term is identically zero before this index: the K depression under `disease_prior.target = capacity`, or the added-mortality amplitude under `target = survival` (current candidate; `age_priors.py`).

---

## Appendix B. Overlay-only comment keys (overlay `J200VW_ALL_local.json` and its builders)

The overlay is a session-scratchpad file and was not edited (ground rule). The resolved config of
the candidate takes every `_`-key from the overlay, so these texts are what a reader of
`RESOLVED_CONFIG.json` sees. Apply the following in the overlay builders (`mk_lc_ov.py` and the
`gpuq*/` task scripts) so that future overlays carry them.

| row | key | proposed text |
|---|---|---|
| C1–C9, C14–C16 | the overlay copies of the base keys | use the base-config texts as revised in `config/age_model_config.json`, then delete the "this file's default" half where the overlay sets the candidate value |
| C10 | `manifold_prior._form_comment` | "capacity_demography: Sa, F and Sj share ONE direction f; beta_K = w_K*(r_K*f + sqrt(1-r_K^2)*g), r_K = sigmoid(N(logit .85, .15)). coupling_target, target_corr_*, communality_logit_scale, angle_scale and fixed_coupling are not read by this form." |
| C11 | `_grid54_step2_comment` | append "(superseded: the capacity_demography form ignores fixed_coupling; K departs through r_K and its own capacity_factor)" |
| C12 | top-level `_comment` | "One shared space-for-time direction f for Sa, F and Sj; K's spatial level uses r_K f + sqrt(1-r_K^2) g, and its time change uses f (readout.within_shared = demographic)." |
| C13 | `kernel_contract._comment` | "retains all 64 eigenfeatures (latent_dim = readout.rank = 64; truncation none)" |
| C17 | `_initpop_comment` | add "This seed (K before the floor) starts a 40-yr spin-up; the 1902 state is the spun-up total (`initpop_spun`)." |
| C18 | `observation_mask._pinned_comment` | "Selection-stage masked twins only. The all-data candidate has no holdout_blocks key, and every BBS row has weight 1." |
| C19 | `_claude_recovery` | prefix "HISTORICAL STEP (not the candidate):" |
| C20 | `_final` | correct to "native departure r ~ Beta(1.5, 1.5)" (the value that runs); the J110 overlay's `native_departure._comment` has the same Beta(17, 3) error |
| C21 | `_grid54_step1_comment` | prefix "HISTORICAL STEP: superseded by the absolute clumped Allee (allee_prior.form = absolute, aggregation_k 3, per_capita)" |
| C22 | `_grid54_step3_comment` | append "(superseded: route effects are Laplace-integrated by laplace_fit.py with sigma free, HalfNormal(1); sigma_u does not enter)" |
| C23 | `pseudo_zero_subsample._comment` | "27 km design target 1814 × 38 = 68,932; the 54 km candidate input holds 68,704 pseudo-zero rows (1,808 cells × 38 yr)" |
| C24 | `_paths_comment`, `path_features`, `community_bridge`, `_grid54_comment` | "Z_gathered is the FIELD54 production cube field54prod_ema s64 (native 54 km, not coarsened). path_features.input_dir / output_dir are NOT the Z that runs; visualizers that read them show the wrong Z." Also repoint `path_features` (a value change: the lead's call) |
| C25 | `_later_established_state`, `_overflow_early`, `_crowding_curvature_experiment`, `_floor_combination`, `_pathology_candidate` | prefix each with "HISTORICAL STEP RECORD — values inside are not the candidate's settings." |
| (P9) | `population_model.capacity_floor_n50_multiple` | a value, not a comment: it is unread. It reads 8 locally and 20 on TACC; the true ratio K_min/ñ50 is 20. Set it to 20 or remove it |

## Appendix C. `config/data_config.json` `timeline._comment` (T1/T3)

Not edited, because the task limited config edits to `age_model_config.json`.

**Proposed replacement for the two clauses.**
> invasion_year=1940 (NYC release; inv_timestep is DERIVED as invasion_year-first_year, never hardcoded; the 54 km candidate uses 1951 via config/overlays/grid54_data.json and established1951 inputs)

> disease_start_year=1993 (...; disease_timestep is likewise DERIVED, and bounds the epizootic term: K depression under disease_prior.target = capacity, added-mortality amplitude under target = survival)

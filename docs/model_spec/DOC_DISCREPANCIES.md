# Documentation vs runtime: discrepancies for HMC candidate LA1

Compiled 2026-11-08 (read-only audit). The runtime truth comes from executed code: worktree `route-variance` @ 0f73b31,
main-repo `research/exp/invasion_sandbox/laplace_fit.py` @ dee075d, the overlay `J200VW_ALL_local.json` (session
scratchpad) merged over `config/age_model_config.json`, the numpyro trace at the LA1 parameters (`ms_spec/run1/sites_table.txt`)
and the measured coverage (`ms_spec/run1/cov_summary.md`). Each row below was checked by reading both the doc line and
the code. Unmarked rows are established. **[L]** marks a likely reading. **[U]** marks a claim that could not be checked.

Severity:
- **wrong**: the doc describes a mechanism or formula that does not run.
- **stale**: a value, count or date the code has since changed.
- **ambiguous**: true for some configuration, but misleading for LA1.
- **units**: a unit or gauge mismatch.

Abbreviations:
- `AP` = `src/model/age_priors.py`, `AF` = `src/model/age_fields.py`, `AFW` = `src/model/age_forward.py`.
- `LF` = `research/exp/invasion_sandbox/laplace_fit.py` (main repo).
- `OV` = overlay `J200VW_ALL_local.json`, `BASE` = worktree `config/age_model_config.json`.

---

## 1. docs/PRIORS.md

| # | doc file:line | what it says | what runs (evidence) | severity |
|---|---|---|---|---|
| P1 | PRIORS.md:24 | "the likelihood is EXACTLY invariant under (K -> cK, ...)"; "EVERY absolute-scale prior must be declared in ROUTE-COUNT units" | `capacity_floor_internal` = .0231039 is added to K in INTERNAL units, with no gauge conversion (`AF:33`, `AF:378`). Changing pop_scalar would change its meaning, so the invariance and the rule both fail for the floor. q95 "19.4, rounded to 20" is the 27 km figure; `OV:401` gives 54 km q95 17.3. | units |
| P2 | PRIORS.md:30, :42 | anchor "2.6183 counts"; "alpha_k_scale = 0.8" | `capacity_level_prior.target_level_median_route_counts` = 2.582 and `alpha_k_scale` = 1.0 (`OV:94-95`). The trace prior is alpha_k ~ Normal(-1.9819, 1.0) = softplus^-1(2.582/20). | stale |
| P3 | PRIORS.md:36 | "K = softplus(alpha_k + gamma_k*H_k + trend)" | K = softplus(alpha_k + H_K) + capacity_floor_internal (`AF:359`, `AF:378`). The floor is .0231 internal = .462 route counts. gamma_k is fixed at 1 (`AP:1284`), and the trend is off (`k_trend.enabled` false, so `w_k_trend` = 0). | stale |
| P4 | PRIORS.md:48 | Rank-2 angular prior. Four fields with own communality and angle; Corr(Sa,Sj) ~ .83; Corr(F,K) ~ .83; cross-group ~ .69 | `manifold_prior.form` = capacity_demography (`OV:101`, `AP:931-932`). Sa, Fmax and Sj share ONE direction f, so their coefficient correlation is exactly 1. beta_K = w_K(r f + sqrt(1-r^2) g), r = sigmoid(N(logit .85, .15)) (`habitat_geometry.py:9-14`, `AP:792-794`). Trace: `env_corr_survival_adult_juv` = 1.0, `capacity_demography_correlation` = .787. | wrong |
| P5 | PRIORS.md:54 | "Sj reuses survival's amplitude_scale (0.5628)"; amplitude percentiles quoted | `amplitude_scale` = [1.0263, 1.0263, 1.0227, 1.0263] (`OV:115-120`; trace `w_scale_raw` prior scale). The quoted 5-95% bands describe the old scales. | stale |
| P6 | PRIORS.md:60 | "`manifold_prior.form`: `rank2` \| `exchangeable`" | Two more forms exist, and LA1 uses one of them: `capacity_demography` (also `survival_breeding`) (`AP:518`). The doc does not cover the form that runs, or its `coupling_priors` (P27). | stale |
| P7 | PRIORS.md:70 | "allee_factor = 1 - 2^(-C/n50) ... at C = n50 the brake is exactly half released"; "n50_raw_loc = -2.0 gives median n50 = 0.127"; "fecundity retained 42% at 0.1 counts..."; "Allee fold near 1/gamma = n50/ln2" | `allee_prior.aggregation_k` = 3 and `per_capita` = true (`OV:67-69`, `AP:362-363`), so A = 1-(1+x/3)^-4 with x = ln2·C/n50 (`AFW:1117`). Half saturation falls at C = .82 n50, not n50. The retention table and the 1/gamma fold are for the k=None form. `n50_raw` is PINNED at -3.7562 (`OV:217`), so n50 = .0231 counts and gamma = 600 (trace `allee_gamma`). The prior loc is -3.4915, scale .25 (`OV:65-66`). | wrong |
| P8 | PRIORS.md:72 | "The current pin is n50_raw -2.1, i.e. .1155 median-route counts ... internal .00578" | LA1 pins n50_raw = -3.7562, so softplus = .0231 counts and .001155 internal (`OV:217`). That is 0.2 x the documented pin, below the x.33-x.5 brackets in the table at :80-85. On the doc's own bridge it is about .035 [.018] birds/km2 [L, arithmetic]. | stale |
| P9 | PRIORS.md:89 | "The K floor (`capacity_floor_internal`) is absolute: .0462 internal = 1.42 [.71] birds/km2"; "`capacity_floor_n50_multiple` is a label only" | Floor = .0231039 internal (`OV:300`). "Label only" holds (no model read). The label value 8.0 (`OV:310`) does not match the actual ratio: floor / n50 = .0231039 / .0011552 = 20. The floor equals n50 numerically, but in counts (n50) vs internal units (floor). [L: a counts/internal mix-up when the floor was set]. | stale + units |
| P10 | PRIORS.md:97 | K trend described as an active safety valve with budget .05 | `k_trend.enabled` = false (`OV:146`). `w_k_trend` is a deterministic 0 (`AP:1040-1043`; trace `D w_k_trend [0,0,0]`). | ambiguous |
| P11 | PRIORS.md:103 | "The 1940 release ... global_q50 ... 0.7083 ... n = 3791 cells" | Release year 1951 (inv_timestep 49; inputs `established1951`). `global_q50_route_counts` = .6852 (54 km; `OV:80`). Founders = .6852/20·exp(log_inv_total) with log_inv_total PINNED 0 (`OV:218`, `AP:1477-1485`), so .03426 internal in ONE cell, (27, 89). | stale |
| P12 | PRIORS.md:109 | "invasion_sites = 9 ... 9 NEAREST LAND CELLS ... 9 x 10 = 90 total" | `invasion_pulse_prior.form` = single and `n_sites` = 1 (`OV:83-84`), so one cell and one year (`AP:1465-1485`). `invasion_sites` = 4 in the inputs; only the first is used. | stale |
| P13 | PRIORS.md:115 | "90 free (site, year) coefficients ... log_budget=3.0 ... per-coefficient log_sd = 0.316" | No `log_inv_pulse_fraction` site. form single, and `log_inv_total` pinned (deterministic in trace). | stale |
| P14 | PRIORS.md:121 | "effect on carrying capacity ... K = K_base * (1 - severity(x) * onset_gate * (1 - recovery))" | `disease_prior.target` = survival (`OV:176`). K has no disease term (`AF:367-368`). The epizootic is added mortality on Sa, Sj and juvenile arrivers: m = .8·sigmoid(mu + b_late·dec + b_native·shape + basis·w_sev)·gate·(1-rec(1-e^{-age/tau_rec}))·N^n/(N^n+k_half^n) (`AF:100-120`, `AFW:744-751`, `AP:1716-1720`). | wrong |
| P15 | PRIORS.md:127 | "severity(x) = severity_ceiling * K_base^n/(K_base^n + K_half^n)"; "severity keys off K_base ... because ... N does not exist yet"; "n~3 a sharp invasion threshold" | The Hill term acts on CURRENT post-dispersal density N inside the forward step (`AFW:746-748`). The ceiling is `mortality_ceiling` = .8, not `severity_ceiling` .5 (`AP:162-163`). The n bounds follow that ceiling: n in (0.5, 0.85(4/.8-2)) = (0.5, 2.55) (`AP:1156-1165`), so n ~ 3 cannot occur. `hill_n_log_sd` .4 is used as a logit-scale sd of 0.8 on a bounded sigmoid, not as a log sd (trace `disease_hill_n_raw` Normal(-.049, .8)). | wrong |
| P16 | PRIORS.md:133 | "severity = severity_ceiling * sigmoid(mu + ...) * hill(K_base) ... prior median severity of 0.8*0.5 = 40%" | Ceiling .8 (mortality), so the prior-median peak added mortality is .8·.8 = 64% of survivors at saturating density, before gate and recovery. The hill is on N, not K_base. | wrong |
| P17 | PRIORS.md:151-155 | Seed "AT capacity"; "at N/K = 1.0 against dispersal_target_capacity_fraction = 0.8, the emigration logit starts at +0.2*slope, so the native range is a NET EXPORTER" | `dispersal_departure.form` = veit (`OV:221`): p = clip(x - softplus(10(x-1))/10, 1e-6, 1), then multiplied by (1 - r·w_nat) with native_departure r ~ Beta(1.5, 1.5) (`AFW:333-345`, `AP:1845-1849`). The logit intercept and slope and the .8 target have no effect (excluded from HMC coords, `LF:269`). At the LA1 point r = .989, so native-core departure is about 1% of the Veit rate. | wrong |
| P18 | PRIORS.md:159 | "it seeds 1902, not 1966" (the seed is the 1902 state) | `spinup.enabled`: the model first runs 40 years at the mean of the 1902-06 rates, without disease or release, starting from the seed. 1902 then starts from the spun-up state (`AP:1970-1986`; trace `initpop_spun`). The seed uses K_base WITHOUT the floor (`AP:1779`). | stale |
| P19 | PRIORS.md (whole file) | "The reasoning behind each prior ... in config" | No section for the levers LA1 runs: capacity_demography + coupling_priors, readout.split/within_shared, juvenile_survival ratio, reproduction yearling + crowding_power 2, adult_survival_dd, spinup, native_departure, veit departure, observation breeding + observer effect, nest likelihood, capacity floor. | stale |
| P20 | PRIORS.md (no heading) | (target of `OV:22` / `BASE:22` link `docs/PRIORS.md#dispersal-kernels`) | There is no "Dispersal kernels" section: the anchor is broken. | stale |

## 2. docs/TEMPORAL.md

| # | doc file:line | what it says | what runs (evidence) | severity |
|---|---|---|---|---|
| T1 | TEMPORAL.md:20 | `invasion_year` **1940** | LA1's resolved data config `timeline.invasion_year` = 1951 (`ms_spec/run1/resolved_data_config.json:25`). The input metadata `inv_timestep` = 49, set by `kernel_sens.py derive`, which changed only `inv_timestep` 38 -> 49 (docs/model_spec/DESK_PROVENANCE.md row 6). The worktree `config/data_config.json:25` still says 1940. | stale |
| T2 | TEMPORAL.md:66-69 | "the release always fires in calendar 1940 ... (1902 -> index 38) ... never hardcoded" | It fires at index 49 (1951) in cell (27, 89). The index comes from the derived input metadata, not from `temporal.invasion_timestep` on the worktree timeline. | stale |
| T3 | TEMPORAL.md:76-79 | `disease_timestep` "bounds the K-depression term in `src/model/age_fields.py`: that term is identically zero before this index" | It bounds the added-mortality amplitude: `jnp.where(_tt >= disease_timestep, amplitude, 0)` (`AP:1716-1720`). No K depression runs (`AF:367`). `disease_timestep` = 91 (1993) is correct. The same "K-depression" wording is in the data-config `timeline._comment`. | wrong |

## 3. docs/DATA_SOURCES.md

| # | doc file:line | what it says | what runs (evidence) | severity |
|---|---|---|---|---|
| D1 | DATA_SOURCES.md:129 | "The model grid (27 km, `grid.target_res_m = 27000`)" | LA1 runs on 54 km: resolved data config `grid.target_res_m` = 54000; trace grids are 67x112 with 4,318 land cells. The worktree `config/data_config.json:31` says 27000. | stale |
| D2 | DATA_SOURCES.md:108-125 (product table) | Product list | Data entering LA1's objective are missing from the table: NestWatch nests (`nest_likelihood`), early CBC (`--cbc_counts`), BBL dead recoveries and summer-winter pairs (`--bbl`, `cbc_obs`), GBIF museum specimens plus Aves effort (`--specimens`), native-MX hull / range-absence npz, and the disease arrival map. | stale |

## 4. Config comment keys (overlay `OV`, with the same text in `BASE` where noted)

The resolved config takes every `_`-key from the overlay. BASE's copies are correct for BASE's own defaults (rank2,
capacity target, k = None, latent_dim 24). They become wrong only because the overlay carries them into LA1.

| # | doc file:line | what it says | what runs (evidence) | severity |
|---|---|---|---|---|
| C1 | OV:63 (`_allee_comment`); BASE:58 | "allee_factor = 1 - 2^(-C/n50), half released at C = n50" | k = 3 per-capita, so half release is at C = .82 n50 (see P7). | wrong (for LA1) |
| C2 | OV:61 `_disease_prior_comment`; BASE:56 | "effect on capacity ... K = K_base * (1 - severity ...)" | Added mortality; K is disease-free (see P14). | wrong |
| C3 | OV:151 `_severity_comment`; BASE:121 | "a Hill function of local carrying capacity" | Hill of current density N (`AFW:746-748`). | wrong |
| C4 | OV:152 `_ceiling_comment`; BASE:122 | "severity_ceiling is now LOAD-BEARING FOR IDENTIFIABILITY ... 0.5" | `severity_ceiling` is unused when target = survival. `mortality_ceiling` .8 is used (`AP:162-163`). | wrong |
| C5 | OV:158 `_mu_comment`; BASE:128 | "severity = severity_ceiling * sigmoid(mu + ...) * hill(K_base)" | m = mortality_ceiling·sigmoid(...)·gate·(1-rec)·hill(N). | wrong |
| C6 | OV:7 `_disease_arrival_map_comment`; BASE:7 | "Consumed ... as the onset gate for the K depression" | It is the onset of the added-mortality gate: compact_start smoothstep over 3 yr (`disease_onset.py:63-77`). | stale |
| C7 | OV:138 `_k_range_comment`; BASE:109 | "K is softplus(...) with nothing bounding it below"; "24x is +/-2 log SD" | K >= capacity_floor_internal .0231 (`AF:378`; trace `K_flat` min .023104). The value is `max_fold_deviation` 26 (`OV:140`). | wrong |
| C8 | OV:92 `_link_comment`; BASE:79 | "K = softplus(alpha_k + gamma_k*H_k + trend)" | Plus the floor; gamma_k = 1; trend off (see P3). | stale |
| C9 | OV:62 `_capacity_level_comment`; BASE:57 | "anchored on ... (median 2.618)" | target 2.582 (`OV:94`). | stale |
| C10 | OV:100 `_form_comment` | "rank2 ... or exchangeable" | form = capacity_demography. `coupling_target`, `target_corr_*`, `communality_logit_scale` and `angle_scale` (`OV:102-108`) are not read by this form (`AP:772-809`). | stale |
| C11 | OV:121 `fixed_coupling: 1.0` + OV:408 `_grid54_step2_comment` | "single habitat field, manifold_prior.fixed_coupling 1.0 (... every rate, capacity included, reads H = Z.f ...)" | capacity_demography ignores `fixed_coupling` (its docstring says so, `AP:776-777`). K departs through r = sigmoid(`capacity_correlation_logit`) (trace r = .787), plus its own `capacity_factor` (64 coords). | wrong |
| C12 | OV:400 top-level `_comment` | "one shared space-for-time coefficient field" | Space-for-time holds for Sa/F/Sj. K's spatial level uses its own direction (r f + sqrt(1-r^2) g), while its time change uses f (`readout.within_shared` = demographic, `AP:950-962`). | ambiguous |
| C13 | OV:17 `kernel_contract._comment` | "retains the first latent_dim (24) eigenfeatures" | `latent_dim` 64 (`OV:11`), `readout.rank` 64 (`OV:237`), and the trace `env_features` plate = 64. | stale |
| C14 | OV:22 `_kernel_source_comment`; BASE:22 | "See docs/PRIORS.md#dispersal-kernels" | The anchor does not exist (P20). | stale |
| C15 | OV:76-78 `_invasion_pulse/_sites/_budget_comment`; BASE:68-70 | "The 1940 release"; "invasion_sites = 9"; "90 free (site, year) coefficients" | 1951, one cell, one year, total pinned (see P11-P13). The value at `OV:55` is `invasion_sites` 4. | stale |
| C16 | OV:56 `_release_allee_comment`; BASE:51 | "the 9 candidate release cells ... 27 km cell ... Matches invasion_window_years" | The value is 0 (no exemption, `OV:57`): one release cell, a 54 km grid, and the comment describes an inactive feature. | stale |
| C17 | OV:71 `_initpop_comment` | "The t=0 (1902) native-range seed ... FRACTION OF LOCAL FITTED CAPACITY" | The seed is K_base(1902) without the floor, then a 40-yr spin-up, so the 1902 state is `initpop_spun` (P18). | ambiguous |
| C18 | OV:229 `observation_mask._pinned_comment` | "withhold exactly the 31 216-km blocks" | No `holdout_blocks` key. All 191,895 rows carry weight 1 (trace `obs` scale min = max = 1; `data_loading.py:705-719`). | stale |
| C19 | OV:521 `_claude_recovery` | "candidate + delayed host-adaptation recovery; 2021-25 still HELD OUT" | `recovery.form` = exponential (`OV:183`; no `disease_delay_rec` in trace). No years are held out. (`OV:523 _leader` is consistent.) | stale |
| C20 | OV:524 `_final` | "native departure r ~ Beta(17,3) (BBL-anchored)" | `native_departure` beta_a 1.5, beta_b 1.5 (`OV:322-326`; trace `native_departure_r` Beta(1.5, 1.5)). | wrong |
| C21 | OV:406 `_grid54_step1_comment` | "RELATIVE Allee (mate-finding by N/K), n50/K pinned at .0197" | `allee_prior.form` = absolute (`OV:68`), with `allee_gamma` multiplying N (`AP:1506`). | stale |
| C22 | OV:408 `_grid54_step3_comment` | "observer x route effect (sigma_u 0.97 fixed, eta ~ N(0,1))" | LF integrates the route effects by per-route Laplace with e ~ iid N(0,1). sigma is FREE with HalfNormal(`sigma_prior_scale` = 1) (`LF:616`, `LF:671-672`, `LF:686-690`); sigma_u .97 does not enter the objective. | stale |
| C23 | OV:403 `pseudo_zero_subsample._comment` | "1814 x 38 = 68,932" absence observations | Measured: 68,704 pseudo-zero rows (no route id) in the LA1 input. [L: the comment describes the 27 km design target; the 54 km input differs by 228 = 6 x 38 rows] | stale [L] |
| C24 | OV:2 `_paths_comment`, OV:376 `path_features`, OV:381 `community_bridge`, OV:401 `_grid54_comment` | Z built from `cube_s64_grid54` by `coarsen_model_grid.py --factor 2` | Z_gathered = the FIELD54 production cube `field54prod_ema`, native 54 km, not coarsened (docs/model_spec/DESK_PROVENANCE.md headline and row 3). `_grid54_comment` also cites `holdout_blocks` (C18) and `target 3.0->2.582`. | stale |
| C25 | OV:419-520 (`_later_established_state`, `_overflow_early`, `_crowding_curvature_experiment`, `_floor_combination`, `_pathology_candidate`) | Historical step blocks carrying `native_departure: false`, `capacity_floor_internal` 0 / .0118, `capacity_floor_n50_multiple` 0 / 4 | These are records of parent steps, not LA1 settings. A reader can mistake them for live values. | ambiguous |

## 5. Code docstrings and comments (executed functions)

| # | doc file:line | what it says | what runs (evidence) | severity |
|---|---|---|---|---|
| A1 | AF:9-26 (module docstring) | "Each rate reads its OWN manifold ... share a RANK-2 prior ... Corr(F_max, K) ~ 0.85 sits above Corr(F_max, survival) ~ 0.70"; "2.6183 route counts" | capacity_demography: Sa/F/Sj share one direction (corr 1); K has corr r with them (P4). Target 2.582. | wrong |
| A2 | AF:190-203 (`project_and_scatter_age_structured`) | "Q ... reuses the juvenile intercept/slope on ... H_sj_disp"; "Timesteps before `disease_timestep` get K_base unmodified ... which is what pins `k_level`" | Under juvenile_survival ratio: Q_k = sigmoid(alpha_a + H_Sa^disp,k)·sigmoid(alpha_j + H_Sj^disp,k) (`AF:390`, `AF:400-404`). Disease never modifies K (`AF:367`), and K adds the floor at every step (`AF:378`). | stale |
| A3 | AF:252-256 (comment) | Sj "coupled ... through a PRIOR on their correlation (rank-2 manifold prior, target 0.85)" | Sj and Sa coefficients are identical up to amplitude (correlation exactly 1). | stale |
| A4 | AF:339-343 (comment) | "measured 2.6183 route counts" | 2.582 | stale |
| A5 | AF:38-81 (`disease_severity` docstring; executed only for the `disease_severity_map` deterministic) | "Per-cell PEAK fraction of carrying capacity the epizootic removes ... K = K_base*(1 - severity(K_base))" | Under target = survival nothing is removed from K. See A13: the deterministic this feeds is not the mortality that runs. | ambiguous |
| A6 | AP:816-823 (`sample_priors` docstring) | "rank-2 angular prior ... continental time trend on K, the structured mycoplasmal-conjunctivitis effect on K" | capacity_demography; trend off; disease on survival. | stale |
| A7 | AP:1050-1064 (comments) | "MYCOPLASMAL-CONJUNCTIVITIS EFFECT ON K: K = K_base * (1 - ...)"; "severity = severity_ceiling * sigmoid ... median severity of 0.8 * 0.5 = 40%" | See P14 and P16. | wrong |
| A8 | AP:1305-1314 (comment) | "At loc -2.0 (median n50 0.127 counts) the retained fecundity is 42% at 0.1 counts, 94% at 0.5 ..." | n50_raw pinned at -3.7562 and k = 3 per-capita (see P7). | stale |
| A9 | AP:1399 (`build_model_2d` docstring) | "runs the age-structured forward simulation from the invasion year" | It runs from 1902 (t = 0) after a 40-yr spin-up. The release enters at t = 49 (`AP:1970-1987`, `AFW:613-617`). The docstring also omits the nest likelihood and the coupling-prior factors, which are in the objective. | wrong |
| A10 | AP:1450-1471 (comments) | "The 1940 release(s)"; "One coefficient per (candidate site, year)"; "(default 1 = 1940)" | form single: one cell, one year (1951), pinned total. | stale |
| A11 | AP:1770-1776 (comment) | "Since dispersal_target_capacity_fraction is 0.8, that puts the seed at N/K - 0.8 = +0.2 on the emigration logit ... NET EXPORTER" | Veit departure plus native_departure (P17). The logit has no effect. | wrong |
| A12 | AP:1794-1796 (comment) | "local linearized matrix is [[Sa, Sj], [F*Sa, 0]]. Thus fecundity at lambda=1 is (1-Sa)/(Sa*Sj)" | yearling_breeding: lambda = Sa + Sj·F, so F = (1-Sa)/Sj (`AP:699-712`). With adult_survival_dd, Sa' = Sa/(1+c_s), c_s = theta·max(Sa/clip(1-Sj·Fmax, .05, 1) - 1, 0). | stale |
| A13 | AP:1784-1789 (comment + `disease_severity_map` deterministic) | "Per-cell peak severity at MODERN capacity -- the model's falsifiable claim about the epizootic" | The map evaluates ceiling .8·sigmoid(...)·hill(K_base[2025]). The running mortality uses hill(current N) times gate times (1 - recovery) (`AFW:744-751`). The map is not the mortality applied: it has no gate or recovery, and its Hill acts on K_base, not N. | ambiguous |
| A14 | AP:683-688 (`equilibrium_age_quantities` docstring); AP:690 (comment) | "the local projection matrix is [[Sa,Sj],[F*Sa,0]]"; "default: 1 - exp(-x), as before" | The yearling branch runs (lambda = Sa + Sj·F_at_K, `AP:699-711`). The Allee factor is 1-(1+x/3)^-4. | stale |
| A15 | AP:238-247 (`observer_log_offset` docstring) | "u = sigma_u * eps, eps ~ ZeroSumNormal(1)"; "sigma FIXED by default"; "Zero-sum ... removes the common-shift direction of u that would otherwise trade against the capacity level" | The numpyro site still exists, but LF sets it to 0 (`LF:264`). The objective integrates iid N(0,1) route effects per route by Laplace, with sigma free under HalfNormal(1) (`LF:624-659`, `LF:690`). With no zero-sum constraint, the common shift is held only by the N(0,1) prior. | ambiguous (wrong for LA1's objective) |
| A16 | AP:369-372 (config comment, `readout`) | "level ... and within-cell deviation ... get separate manifold weights and amplitudes" | With `within_shared` = demographic: no separate weights or amplitudes. `w_scale_within` = w_scale, and every field's deviation reads the Sa direction f (`AP:950-962`; trace `w_scale_within` = `w_scale`). | ambiguous |
| A17 | AFW:1105-1112 (`allee_mated_fraction` docstring) | "k > 0: ... E[1 - exp(-gamma X)] = 1 - (1 + x/k)^(-k)"; "birds do not experience the 27 km cell mean" | per_capita: exponent -(k+1) = -4 (`AFW:1117`; explained only in the module comment at `AFW:1099-1102`). The cell is 54 km. | ambiguous + units |
| A18 | AFW:1157-1161 (`reproduction_age_structured` docstring) | "Allee factor 1 - exp(-allee_gamma * N)"; "surviving adults produce the next juvenile cohort" | 1-(1+gamma·N/3)^-4. yearling_breeding: births = all winter survivors (adults + Sj stayers + Q-surviving arrivers) times F_eff times A (`AFW:1195`, `AFW:1213-1216`). F_eff = Fmax/(1 + c (N/K)^2) (crowding_power 2, `AFW:1173`). | stale |
| A19 | AFW:311-317 (`dispersal_step_age_structured` docstring) | "Departure probability rises logistically with local density relative to K (dispersal_logit_*, target_fraction)" | `veit_departure` replaces it with the smoothed p = min(N/K, 1) (`AFW:333-339`), times (1 - r w_nat) (`AFW:344-345`). | stale |
| A20 | AFW:471-488 (`forward_sim_age_structured` docstring) | "several nearby candidate cells and a per-(site, year) pulse"; "The 1940 release ... into a 27 km cell" | One cell, one year (1951). 54 km grid. `allee_exempt_years` = 0. | stale + units |
| A21 | LF:715-716 (comment in `parts`) | "the old pseudo-zeros are replaced: remove their contribution (rows with no route id)" followed by `pass` | It is a no-op. The 68,704 pseudo-zero rows stay in the Laplace NB2 BBS term at route effect 0 (`on` false at `LF:647`; `train` keeps them unless `--drop_pseudozeros`, `LF:279-282`). They count IN ADDITION to the range-absence term. The help text at `LF:122-124` says the same. | wrong |
| A22 | LF:14 (docstring) | "Everything else is Codex's posterior objective (newton_fit.py / local_joint_fit.py --objective posterior)" | The LA1 objective also carries CBC counts (latent effort, BBL source, linear drift), the BBL m-array with disease, specimen counts, range absences and native_viable (`LF:764`). The docstring names none of these. | ambiguous |
| A23 | LF:194-195 (`--range_absence` help) | "(full pre-1940 footprint + 1940-65 outside the native / introduced envelopes; rangev2 restoration)" | LA1 uses `range_absence_halo400_nativeMX.npz`: 74,306 cell-years, 400 km halo plus native Mexico, release 1951 (extractor; DESK_PROVENANCE row 8). The builder was not re-read here. | stale [L] |
| A24 | LF:209-210 (`--specimens` help) | "effort-aware museum-specimen occupancy likelihood" | With `--specimen_counts`, NB2 counts per cell-decade (`LF:756-758`). | ambiguous |
| A25 | specimen_lik.py:10-19 (module docstring) | "MODEL ... P(y_cd > 0) = 1 - exp(...)"; "OCCUPANCY (detection) form, not a count: ... the count beyond the first specimen carries collector behaviour"; "One nuisance ... Beta(2, 2) on beta" | LA1 executes `specimen_nll_counts` (NB2 counts, coverage 9/9) and not `specimen_nll` (cov_summary). It adds a 4th nuisance, log phi ~ N(0, 2) (`LF:757-758`). The "collector filter" is in the precomputed cache [U: builder not re-read]. | wrong |

## 6. research/CONSIDERATIONS.md (model-fact items only)

| # | doc file:line | what it says | what runs (evidence) | severity |
|---|---|---|---|---|
| R1 | CONSIDERATIONS.md:7 (G14) | "the current pin (n50_raw -2.1 = .1155 median-route counts = .00578 internal)" | LA1 pin: -3.7562, i.e. .0231 counts and .001155 internal (`OV:217`). | stale |
| R2 | CONSIDERATIONS.md:9 (G14) | "K floor (.0462 internal) = 1.42 [.71] birds/km2" | .0231039 internal (`OV:300`), about half. | stale |
| R3 | CONSIDERATIONS.md:13 (P7) | "The decision [-2.1 pin] stands until the user revisits it" | LA1 runs a different pin (-3.756). Either the decision was revisited without a record here, or LA1 departs from it [U: no decision record grepped]. | stale [U] |
| R4 | CONSIDERATIONS.md:50-55 (G6) | "The age model consumes RAW instantaneous z, first 24 of 64 dims"; "In 1902-1939 the only data are pre-invasion pseudo-zeros" | Rank 64 (`OV:237`). Z is an EMA cube (DESK_PROVENANCE headline, [E] there). Pre-1940 data now also include range absences, early-CBC counts 1902-65, and specimen counts from 1880-1901 onward (read at 1902) (`LF` terms; cov_summary). | stale |

---

## Top 10

1. **Disease mechanism (P14-P16, C2-C6, A7, T3).** Docs say the epizootic depresses K through severity(K_base) with ceiling .5. LA1 runs added mortality on Sa, Sj and juvenile arrivers with Hill on CURRENT density N, `mortality_ceiling` .8, and hill_n bounded to (0.5, 2.55). K has no disease term. `disease_severity_map` (A13) is a K_base-based diagnostic, not the mortality that runs.
2. **Allee formula (P7, C1, A8, A14, A17, A18).** "1 - 2^(-C/n50), half at n50" and "1 - exp(-gamma N)" are the k = None form. LA1 runs 1-(1+gamma N/3)^-4: half saturation at .82 n50, with n50_raw pinned at -3.7562 (gamma 600).
3. **Pin and floor values (P8, P9, R1-R3).** PRIORS and CONSIDERATIONS give pin -2.1 / .1155 counts and floor .0462. LA1 has -3.756 / .0231 counts (0.2 x the documented pin) and floor .0231039 internal. The floor equals n50 numerically, in internal vs count units: the actual ratio is 20, not the labelled 8 (units [L]). The floor also breaks the gauge-invariance claim (P1).
4. **Habitat prior form (P4-P6, A1, A3, A6, C10-C12, A16).** Docs describe rank-2, four own manifolds. LA1 is capacity_demography: one shared direction for Sa/F/Sj, K mixed at r ~ logit-normal(.85, .15). The split readout reuses f for every field's time change. `fixed_coupling` 1.0 is inert, and `_grid54_step2_comment` claims capacity reads the same field.
5. **Invasion (P11-P13, T1-T2, C15-C16, A10, A20).** Docs say 1940, 9 sites, 90 coefficients, per-(site, year) pulse. LA1 releases in ONE cell, (27, 89), in 1951 (t = 49), with log_inv_total pinned at 0 (.03426 internal) and no Allee exemption.
6. **Departure and initial state (P17-P18, A11, C17).** The "logit at target .8, net exporter" narrative is inert. Veit departure times (1 - r w_nat) with r ~ Beta(1.5, 1.5) runs (r = .989 at LA1, so the native core is almost sedentary). 1902 starts from a 40-yr spin-up, not the seed.
7. **Route effects and pseudo-zeros (A15, C22, A21).** The observer docstring (ZeroSumNormal, sigma_u .97 fixed) does not describe LA1's objective: iid N(0,1) route effects, Laplace-integrated, sigma free under HalfNormal(1). The `LF:715` "remove their contribution" is a `pass`: 68,704 pseudo-zero rows still count alongside the range absences.
8. **`_final` says native departure r ~ Beta(17,3) (C20).** It runs as Beta(1.5, 1.5).
9. **Specimen likelihood (A25, A24).** The docstring argues for a binary occupancy form and against counts. LA1 runs NB2 counts (`specimen_nll_counts`) with an extra log-phi nuisance.
10. **Rank, grid and holdouts (C13, D1, R4, C18-C19, C7).** Docs say rank 24 (and raw z) and a 27 km grid. LA1 is rank 64 on a 54 km grid with an EMA cube. Holdout comments ("31 blocks withheld", "2021-25 still HELD OUT") are stale: all 191,895 rows have weight 1. `_k_range_comment` says nothing bounds K below, but the floor does.

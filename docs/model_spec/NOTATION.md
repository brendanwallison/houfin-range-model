# Notation for the LA1 model

This is the one symbol table for MODEL_SPEC.md. Each symbol below has exactly one meaning.

- **Code locations.** `src/` = worktree `0f73b31`. `LF` = `research/exp/invasion_sandbox/laplace_fit.py` @ `dee075d`.
- **Trace sites** are numpyro site names in the LA1 trace (`trace_sites_LA1.txt`). "—" means the quantity has no site.
- **Tags.** [E] measured or quoted, [L] inferred.

## 1. Units and scales (read this first)

| scale | symbol | definition | value | evidence |
|---|---|---|---|---|
| internal density | N | the state unit of the forward model ("relative units", "model units") | — | `simulated_density`, `Na_grid`, `Nj_grid` |
| gauge | s | route counts per internal unit (`pop_scalar`) | 20 | `population_scale_route_counts_per_relative_unit` = 20; `data['pop_scalar']` = 20.0 |
| route-count scale | C = sN | the expected BBS route count at the median observer×route, with route effect 0 | — | `age_priors.py:2066` |
| Allee half-constant | $n_{50}$ | **route counts**; $\tilde n_{50}=n_{50}/s$ in internal units | 0.0231039 RC = 0.00115520 internal | `age_priors.py:1305-1322, 1506` |
| K floor | $K_{\min}$ | **internal** units | 0.0231039 internal = 0.462 RC = 20 $n_{50}$ | `capacity_floor_internal`; `age_fields.py:33,378` |
| disease Hill half | $k_h$ | internal; the prior is stated in RC | 0.0136 internal (0.272 RC) | `disease_k_half`, `disease_k_half_route_counts` |
| founders | R | internal; defined as $q_{50}/s$ | 0.03426 internal = 0.685 RC | `inv_pop_relative` |
| capacity anchor | — | median of K⁰ at H_K = 0 (prior), RC | 2.582 RC → α_K loc −1.982 | `target_level_median_route_counts`; `_ALPHA_K_LOC` |
| birds/km² | — | **no conversion enters the likelihood.** There are three incompatible conversions; see §3, items D1–D3 | — | — |

**The numeric coincidence $K_{\min}=n_{50}$ is a trap** [E]. Both are 0.0231039, but $K_{\min}$ is in internal units and $n_{50}$ is in route counts.
- In common units, $K_{\min}=20\,n_{50}$. The LA1 overlay builder says the same: `capacity_floor_n50_multiple` = 20 (TACC overlay).
- The local overlay still says 8.0. Neither value is read by the code.
- The floor was set by halving the absolute 0.0462 floor, and $n_{50}$ by ×0.2 of softplus(−2.1). The coincidence is arithmetic: 0.0462 × 0.5 = 0.1155 × 0.2.

## 2. Symbol table

### Indices and data

| symbol | meaning | units | code | config | trace site |
|---|---|---|---|---|---|
| x | land cell (4,318) | — | `land_rows`, `land_cols`, land index | — | — |
| t | year index 0…123 (1902 + t) | yr | `t`, `t_idx`, `obs_time_indices` | — | — |
| k | juvenile kernel (direction × radial band), 1…12 | — | kernel axis of `juvenile_fft_kernel_stack`, `Q_flat[...,k]` | `dispersal.juvenile_radial_splits_km` | — |
| ℓ | field index ∈ {a, f, K, j} (adult survival, fecundity, capacity, juvenile survival) | — | column of `w_env` [Sa, Fmax, K, Sj] | `manifold_prior` | — |
| i | BBS row | — | row of `observed_results` | — | `obs` |
| r(i) | observer×route of row i (−1 = pseudo-zero) | — | `obs_obsroute_idx`, LF `idc`/`on` | — | — |
| ϱ | CBC circle-year row / specimen cell-decade row | — | LF `ccnt` rows / `spc` rows | — | — |
| c(ϱ) | CBC circle | — | `ccnt['c']`, `circ_row` | — | — |
| n | nest attempt | — | `nest_*` arrays | `nest_likelihood` | — |
| $y_i$ | BBS count | birds/route | `observed_results` | `bbs_npz` (ingest) | `obs` |
| $\mathrm{FY}_i$ | observer's first year on the route (0/1) | — | `obs_first_year` | — | — |
| $\mathrm{tier}_i$ | quality tier (1 = Mexico) | — | `obs_quality` | — | — |

### Habitat and fields

| symbol | meaning | units | code | config | trace site |
|---|---|---|---|---|---|
| $Z_t(x)$ | DESK latent features (64) | — | `Z_gathered`, `z_t` | `input_dir`, `readout.rank` = 64 | — |
| $Z^d_{t,k}(x)$ | land-conditioned path features of kernel k | — | `Z_disp_gathered`, `z_disp_t` | `dispersal.path_*` | — |
| $\bar Z(x)$, $\bar Z^d_k(x)$ | 1996–2025 mean of Z / $Z^d$ | — | `Z_ref`, `Z_disp_ref` | `readout.ref_window` | — |
| $z_c$ | mean Z over occupied cell-years | — | `habitat_center_occupied`, `z_center` | `habitat_centering` = occupied | `habitat_center_offsets` (= $z_c\cdot\beta_\ell$) |
| f | shared demographic coefficient direction | — | `shared` | — | `manifold_factor` |
| g | capacity-only coefficient direction | — | `capacity` | — | `capacity_factor` |
| $\xi_\ell$ | raw amplitude | — | `w_scale_raw` | `amplitude_loc`, `amplitude_scale` | `w_scale_raw` |
| $w_\ell$ | amplitude, softplus(ξ_ℓ) | logit / softplus-arg per unit Z·f | `w_scale` | — | `w_scale` (`environment_kernel_variance` = w²) |
| $\xi_r$ | logit of r | — | — | `capacity_correlation_target`, `_logit_scale` | `capacity_correlation_logit` |
| r | K-vs-demography coefficient correlation (prior construction) | — | `r` | — | `capacity_demography_correlation` (also = `env_corr_*_capacity`) |
| $\beta_\ell$ | coefficient vector of field ℓ | — | `beta_s`, `beta_r`, `beta_k`, `beta_sj`; columns of `w_env` | — | `w_env` |
| $\beta^w_\ell$ | within-cell coefficient (= $w_\ell f$) | — | `beta_*_w` | `readout.within_shared` | `w_env_within` |
| $h(x,t)$ | shared demographic habitat index $(Z_t-z_c)\cdot f$ | — | (implicit: `H_s_local/w_a`) | — | — |
| $h^d_k$ | the same on path features | — | (implicit) | — | — |
| $H_\ell(x,t)$ | linear predictor of field ℓ | link scale | `H_s_local`, `H_r_local`, `H_k_local`, `H_sj_local` | — | — |
| $\alpha_a,\alpha_f,\alpha_K,\alpha_j$ | intercepts at $z_c$ | logit / softplus⁻¹ / softplus⁻¹ (internal) / logit-ratio | `alpha_a`, `alpha_f`, `alpha_k`, `alpha_j` | `adult_survival_prior` (absent → default), `fecundity_prior`, `capacity_level_prior`, `juvenile_survival` | same names |
| $S_a$ | adult annual survival (disease-free, unbraked) | prob | `Sa_flat`, `S_a_val`, `Sa_g` | — | `Sa_flat` |
| $S_j$ | juvenile survival of non-dispersers ($=S_a\sigma(\cdot)$) | prob | `Sj_flat` | `juvenile_survival.form` = ratio | `Sj_flat` |
| F | max fecundity (fledged juveniles per breeder at low density, no Allee) | juv/breeder/yr | `Fmax_flat`, `F_max_val` | — | `Fmax_flat` |
| $K^0$ | capacity before the floor | internal | `Kbase_flat`, `K_base_val` | — | `K_base_flat` |
| K | capacity, $K^0+K_{\min}$ | internal | `K_flat`, `K_val`, `K_g` | `capacity_floor_internal` | `K_flat` |
| $K_{\min}$ | additive K floor | internal | `_FLOOR` | `capacity_floor_internal` = 0.0231039 | — |
| $Q_k$ | juvenile journey survival via kernel k | prob | `Q_flat`, `Q_g` | — | `Q_flat` |
| $\lambda^{\rm pot}$ | potential growth $S_a+S_jF$ | /yr | LF `lam0`; harness `potential_lambda` | — | — |
| $\chi_{\rm coef}$, $\chi_{\rm map}$ | realised Sa–K coefficient cosine / map correlation | — | `_cos`, `_mc` | `manifold_prior.coupling_priors` | `coupling_coef_cos_s_k`, `coupling_map_corr_s_k`; factors `coupling_coef_prior`, `coupling_map_prior` |

### Dynamics

| symbol | meaning | units | code | config | trace site |
|---|---|---|---|---|---|
| $A_t$, $J_t$ | adults / juveniles at the end of year t (post-breeding) | internal | `N_a_new`, `N_j_new`; outputs `Na_densities`, `Nj_densities` | — | `Na_grid`, `Nj_grid` |
| $N_t$ | $A_t+J_t$ | internal | `N_total_new`, `densities` | — | `simulated_density` |
| $A^0$, $J^0$, $N^0$ | start-of-year pools (+ release) | internal | `N_a`, `N_j` in `step` | — | — |
| $A'$, $J^s$, $J^m$, $N'$ | post-dispersal adults, juvenile stayers, juvenile arrivers, total | internal | `N_a_post`, `juvenile_stayers`, `juvenile_arriving`, `N_total_post`/`Nloc` | — | — |
| p | departure probability (both ages) | prob | `p_total`, `p_adult`, `p_v` | `dispersal_departure.form` = veit | — |
| ν | native-range departure reduction | — | `r_nat` | `native_departure.beta_a/b` | `native_departure_r` |
| $I_0(x)$ | native seed shape (core 1, margin <1) | — | `initpop_latent` (inputs), `native_shape` | `initpop_seed` | — |
| $w_{\rm nat}$ | $I_0/\max I_0$ | — | `w_nat` | — | — |
| $\mathcal K_A$, $\mathcal K_k$ | adult / juvenile dispersal kernels (FFT) | — | `adult_fft_kernel`, `juvenile_fft_kernel_stack` | `dispersal.*`; KERNEL_DISCRETISATION.json | — |
| $e_A$, $e_k$ | land-fraction edge corrections | — | `adult_edge_correction`, `juvenile_edge_correction_stack` | — | — |
| $\mathbb F$ | smooth FFT round-off floor | — | `rightpad_convolution` | `numerics.*` | — |
| $\mathcal A(N)$ | mate-finding (Allee) factor | — | `allee_factor`, `allee_mated_fraction` | `allee_prior.*` | — |
| γ | Allee encounter rate | per internal unit | `allee_gamma` (= `gamma_raw`·s) | — | `allee_gamma` |
| $k_A$ | Allee aggregation (clumping) shape | — | `ALLEE_AGG_K` | `allee_prior.aggregation_k` = 3 | — |
| c | fecundity crowding constant | — | `c_flat`, `c_g`, `c` | `reproduction.crowding_power` | — |
| $c_s$ | adult-survival crowding constant | — | `c_s_flat`, `survival_dd_c`, `c_s` | `adult_survival_dd` | `survival_dd_c` |
| θ | share of the brake on survival | — | `_dd_theta` | `adult_survival_dd.theta_a/b` | `dd_survival_share` |
| u | crowding ratio N′/K | — | `N_total_post / K_safe` | — | — |
| $F^{\rm eff}$ | realised births per breeder | juv/breeder | `F_actual` | — | `nest_F_real` (at nest cells) |
| $\omega_K$ | juvenile fraction at K (age-structure prior argument) | — | `rho_K`, `rho` (local) | `age_structure_prior` | factor `local_age_structure_regularizer` |
| R, $x_{\rm rel}$, $t_{\rm rel}$ | founders, release cell, release year index | internal; cell; 49 (1951) | `inv_pop`, `inv_locations[:1]`, `inv_timestep` | `invasion_pulse_prior.*` | `inv_pop_relative`, `log_inv_total` |
| $q_{50}$ | founder reference | RC | `_q50_density`·s | `global_q50_route_counts` = .6852 | — |
| $N^{\rm seed}$, $N^{\rm sp}$ | native seed; spun-up 1902 state | internal | `_seed_flat` / `initpop_seeded` (first value); `initpop_spun` | `spinup.*` | `initpop_seeded`, `initpop_spun` |

### Disease (added mortality)

| symbol | meaning | units | code | config | trace site |
|---|---|---|---|---|---|
| m | added mortality fraction this year | prob | `m_g` | `disease_prior.target` = survival | — |
| Δ(x,t) | density-free mortality amplitude | prob | `disease_m_amp`, `amp` | — | `disease_mortality_amp` |
| $m_{\max}$ | mortality ceiling | — | `ceiling`, `_DISEASE_CEILING` | `mortality_ceiling` = .8 | — |
| $\mu_{\rm sev}$, $b_{\rm late}$, $b_{\rm nat}$ | modifier intercept, epidemic-age slope, native suppression | logit | `mu_sev`, `b_late`, `b_native` | `mu_*`, `late_arrival_*`, `native_resistance_raw_*` | `disease_mu_sev`, `disease_b_late`, `disease_b_native(_raw)` |
| B(x) | 24-function land-centred spatial basis | — | `disease_sev_basis` (= `disease_lag_basis`) | `severity_/lag_space_frequencies` = 4 | — |
| $\omega_{\rm sev}$, $\omega_{\rm lag}$ | basis weights | logit / yr | `w_sev`, `w_lag` | `sev_field_budget`, `lag_field_budget` | `disease_w_sev`, `disease_w_lag` |
| o(x), $o^*(x)$, $o_{\rm dec}$ | mapped arrival; arrival with slack; decades since 1993 | timestep; timestep; decades | `disease_onset`, `onset_t`, `disease_onset_decades` | `disease_arrival_map` | — |
| $\delta_0$ | continental onset slack | yr | `lag0` | `lag0_scale` | `disease_lag0` |
| G | onset gate (smoothstep, 3 y) | — | `onset_gate` | `onset_gate` | — (`disease_tau` = 1.0 unused) |
| $k_h$, $n_h$ | Hill half-density and steepness | internal; — | `k_half`, `hill_n` | `k_half_median_route_counts`, `hill_n_*` | `disease_k_half`, `disease_hill_n` |
| ζ, $\tau_{\rm rec}$ | eventual recovered fraction; e-folding time | —; yr | `rec`, `tau_rec` | `recovery_*` | `disease_rec`, `disease_tau_rec` |

### Observation and data terms

| symbol | meaning | units | code | config | trace site |
|---|---|---|---|---|---|
| $\mu_i$ | expected BBS count | RC | `densities_obs` (**route-count units despite the name**), `expected_obs`, LF `mu0`·e^u | — | `expected_obs` (route effect 0) |
| $\rho_{\rm fl}$ | fledgling detectability weight | — | `phi` (age_priors:2062) | `fledgling_weight_a/b` | `obs_fledgling_weight` |
| η | observer first-year effect | log | `eta` | `observer_effect.eta_prior_scale` | `observer_first_year_eta` |
| $e_r$, σ | standardised route effect; its SD | —; log | LF `e`, `sg`; best_params `laplace_route_sigma` | `observer_effect.sigma_prior_scale` (σ prior); **not** `sigma_u` | — (model site `observer_route_eps` fixed 0) |
| $\phi_0$, q, κ | NB2 base concentration, Mexico multiplier, density power | — | `concentration`, `q_mult`, `kappa` | `conc_density_power_scale` | `concentration`, `quality_conc_mult`, `obs_conc_kappa` |
| $\phi_i$ | row NB2 concentration | — | `conc_obs`, LF `ph`/`phib` | — | `obs_concentration` |
| $H_r$ | route data curvature (Laplace) | — | LF `H` | — | — |
| $k_P$ | NB2 size for the presence/absence detection | — | LF `presence_k`, `PRESENCE_K` | LF `--presence_k` = .43724 | — |
| $s_C,p_C,\phi_C,\gamma_C,v_c$ | CBC log scale, effort power, NB size, decadal drift, circle effect | — | LF `s_`, `pw`, `ph0`, `gam`, `c` | LF flags | — |
| $W$, $W^L$, $P$, $m^{\rm mig}$ | CBC source matrix, local weights, migrant destination density, migrant share | — | `cbc_obs.build_source_matrix` | — | — |
| $\ell_\varrho$, $v_\varrho$, $m(t)$, $\tau_E$ | log effort, latent-effort variance, effort trend, its SD | — | LF `le`, `v`, `m_t`, `tau` | — | — |
| $S^B$, $\pi^B$, $\lambda^B$ | BBL survival, reporting probability, its logit | — | LF `S`, `r`, `rlog` | — | — |
| $n_i$, $m_{ik}$, $b_i$, $g_i$ | BBL releases, recoveries, banding year, region | — | `bbl['n']`, `['m']`, `by`, `grp` | — | — |
| $s_e$, $\beta_E$, $\phi_S$, $E_\varrho$, $T$ | specimen collecting rate per era, effort exponent, NB size, effort, decade window | — | `log_s`, `beta`, `log_phi`, `E`, `tmask` | — | — |
| $B$ | broods per season (nest model) | — | `_B`, `nest_broods` | `nest_likelihood.broods.values` = 2.5 | `nest_broods` |
| $\delta_m,\delta_s,a_s,\lambda_c$ | nest mean/survival covariate effects, DSR logit, log NB conc. | — | `_dm`, `_ds`, `_as`, `_lc` | `nest_likelihood.*` | `nest_mean_delta`, `nest_surv_delta`, `nest_dsr_logit`, `nest_log_conc` |

## 3. Variable drift: one name for several things, or several names for one thing

| # | name in code/docs | the different quantities | proposed reconciled symbol |
|---|---|---|---|
| V1 | `phi` | the fledgling weight (`age_priors.py:2062`); the NB2 concentration (LF `ph`); the CBC and specimen NB sizes; the rank-2 angle `_phi` (`age_priors.py:863`) | ρ_fl; φ_i/φ₀; φ_C; φ_S; drop the angle (rank2 is not executed) |
| V2 | `rho` / ρ | trace `rho` = corr[0,1] (survival–fecundity coefficient corr, = 1 here); `rho_K` = juvenile fraction at K; exchangeable coupling ρ (not executed) | ω_K for the juvenile fraction; report `rho` as r_{af} ≡ 1 |
| V3 | `c` | the fecundity crowding constant `c_flat`; LF CBC circle effects `c`; `C` = route-count density in comments ("C/n50") | c; v_c; C = sN |
| V4 | `gamma` / γ | `allee_gamma` (per internal unit) vs `gamma_raw` (per route count); the field slopes `gamma_a/f/j/k` ≡ 1; CBC drift `gam` | γ (internal only); drop the slopes; γ_C |
| V5 | `kappa` / κ | NB density power `obs_conc_kappa`; `crowding_kappa` 5.0 (inactive); CBC bearing concentration 6.24; `directional_choice_kappa` (inactive) | κ; κ_bear for CBC |
| V6 | `K` | carrying capacity; `meta['K']` = 12 (number of juvenile kernels); BBL `_K` = 65 (max recovery intervals); `k` aggregation (3) and `presence_k` (.437) | K; n_ker = 12; k_A; k_P |
| V7 | `K_base` | docs and comments mean "K before disease"; in LA1, K has no disease, and `K_base_flat` = K **without the floor** (used for the seed and `disease_severity_map`) | K⁰ = pre-floor capacity |
| V8 | n50 | `age_priors` defines n50 in **route counts**; `floor_share.py` and the records use n50/pop as "internal n50"; docs call it the half-saturation density, but under k_A = 3 per-capita the half point is 0.819 n50 | n_50 (RC) as defined; ñ_50 = n_50/s; C_½ = 0.819 n_50 is the actual half point |
| V9 | floor multiple | `capacity_floor_n50_multiple` = 8 (local overlay) or 20 (TACC); records say "8 n50", "20 n50 = 14x the Allee fold"; none is read by code | state K_min in internal units only; K_min/ñ_50 = 20 |
| V10 | `w` | `w_env` (coefficients), `w_scale` (amplitudes), `w_sev`/`w_lag`, `w_k_trend`, `w_nat`, `obs_weight`, absence/specimen row weights `w`, `cbc_w_local` | β (coefficients), w_ℓ (amplitudes), ω_sev/ω_lag, w_nat, row weights unnamed (all 1 here except specimens) |
| V11 | `r` | `capacity_demography_correlation` r; `native_departure_r`; route index r; disease `rec`; BBL reporting `r` | r; ν; r(i); ζ; π^B |
| V12 | `m` | disease mortality; nest mean fledged; BBL recoveries; CBC migrant share; LF route mean `m` | m; m_n; m_ik; m^mig; (route mean inactive) |
| V13 | `Q` / `q` | journey survival Q; Mexico concentration multiplier q; founder reference q50 | Q_k; q; q_50 |
| V14 | `sigma` | the Laplace route SD σ (free, HalfNormal(1)); config `sigma_u` = .97 (not used); the model site name `observer_route_sigma` (prior mode, inactive); `laplace_route_sigma` in best_params | σ only; mark `sigma_u` as not used |
| V15 | `density`, `densities_obs` | `simulated_density` is internal; `densities_obs` is the expected route count (×s, ×observer offset) | N vs μ |
| V16 | `Na` in LF absences | `Na = (Na_grid + Nj_grid)·scale` is the **total** in route counts (LF:713) | sN_t |
| V17 | `initpop_seeded` | the trace site holds the seed $N^{\rm seed}$, but the Python variable is reassigned to the spun-up state (`age_priors.py:1986`), which is what the t = 0 BBS breeders use | N^seed vs N^sp |
| V18 | `Sa_flat` | trace site = disease-free, unbraked S_a. The forward's realised survival is $\tilde S_a$; harness "potential lambda" uses `Sa_flat` | S_a vs $\tilde S_a$ |
| V19 | `beta` | habitat coefficients β_ℓ; specimen effort exponent `beta`; Beta-distribution parameters; dispersal "beta_hi" (comments) | β_ℓ; β_E |
| V20 | `alpha` | field intercepts; V&L kernel shape `vl_a` (.468); route-skew α (inactive); `habitat_alpha` (harness) | α_ℓ; a_VL |
| V21 | `H` | habitat predictors H_ℓ; Laplace curvature H_r / H_c | H_ℓ; H_r |
| V22 | `tau` / τ | `disease_tau` (constant 1.0, unused by the compact gate); τ_rec; CBC effort SD `tau`; FFT tolerance `tol` | τ_rec; τ_E; τ_FFT |
| V23 | `eta` / η | the observer first-year effect; trend_routing η and habitat_change η (inactive) | η |
| V24 | release year | overlays and docs say 1940 (`invasion_*` comments); LA1 inputs are "established1951" (`inv_timestep` 49) | t_rel = 49 (1951) |
| V25 | Z rank | `kernel_contract._comment` says 24 retained; runtime is 64 | rank 64 |
| V26 | trough "D" (records) | records use D for the GP−ETF potential-lambda contrast; this spec uses Δ for disease amplitude to avoid a clash | keep D for the trough metric only |
| V27 | `n` | Hill steepness `hill_n`; nest attempt index; BBL releases | n_h; n; n_i |

### Birds/km² conversions in use (D1–D3, mutually inconsistent) [E]

- **D1, harness.** `harness.py:34`: `BIRDS_PER_KM2_PER_UNIT = 20/(50π·0.4·0.5)` = 0.637 birds/km² per internal unit, i.e. 0.032 birds/km² per route count.
  - Its comment says "25.13 km² sampled, detection .5", which would give 20/(25.13·0.5) = 1.59. The code multiplies π·r rather than π·r², so code and comment disagree by a factor of 2.5.
  - Not used in the likelihood.
- **D2, occupancy.** `age_priors.py:483`: `OCC_BIRDS_PER_COUNT` = 58 birds per 27 km cell per route count, about 0.080 birds/km² per RC. Inactive.
- **D3, PIF bridge.** abs_calib.py, records "G14": n50 ×.2 = 0.035 birds/km² at 40 M birds, i.e. about 1.5 birds/km² per route count. This is 47× D1.
- **Proposal.** State every absolute-density claim with its bridge (PIF total) and never use D1/D2 in documents.

## 4. Reconciled glossary for prose

- **"capacity"** = K (with floor). Use "pre-floor capacity" K⁰ only for the seed and diagnostics.
- **"Allee half-constant n₅₀"**: always route counts. The realised half point is 0.82 n₅₀.
- **"breeders in year t"** in the BBS mean = $N_{t-1}$, the previous end-of-year total (MODEL_SPEC §4, [L]).
- **"potential lambda"** = $S_a+S_jF$: disease-free, Allee-free, density-free.
- **"internal units"**: the forward state. **"RC"**: route counts = s × internal.

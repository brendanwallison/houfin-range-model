# Independent verification of the rewritten model documentation (2026-11-08)

**Scope.** Every factual claim added or changed in the following documents:
- `docs/model_spec/METHODS_DRAFT.md`;
- the uncommitted diffs of `docs/PRIORS.md`, `docs/TEMPORAL.md`, `docs/DATA_SOURCES.md` and `config/age_model_config.json` (worktree `route-variance`, HEAD `0f73b31`);
- items G14 and P7 of `research/CONSIDERATIONS.md` (main checkout, uncommitted).

**Ground truth.**
- Model code: worktree `src/` at `0f73b31`.
- Overlay: scratchpad `J200VWfl_ALL_local.json`. It differs from LA1's `J200VW_ALL_local.json` only in `input_dir` (`_vlF` vs `_vl`).
- Driver: main-repo `research/exp/invasion_sandbox/laplace_fit.py`, functions `parts()` and `run_hmc()`, plus `cbc_obs.py` and `specimen_lik.py`.
- Run evidence: `trace_sites_LA1.txt`, `RESOLVED_CONFIG.json`, `EXECUTED_CODE.md` / `executed_coverage.json`, and the measurement log `ms_spec/run1.log`.
- Input metadata and data files, read in WSL (read-only):
  - `inputs_shape/J200_established1951_vlF_nativeMX/metadata.pkl`;
  - the absence, specimen, native-viability and BBL npz files;
  - the masked-twin and LA1 run logs in `sandbox/ROUTEMARG`.
- Records: `research/experiments/SANDBOX_INVASION_SHAPE_2026-11.md` (cited as `S:<line>`).

**Verdicts.**
- **CONFIRMED:** code, config or measured evidence says exactly this.
- **WRONG:** contradicted; the correct statement is given.
- **UNSUPPORTED:** no evidence found.
- **AMBIGUOUS:** literally defensible but misleading, stale, or imprecise in a way that matters; a fix is suggested.

The documents' own citations were not trusted; each was re-derived.

`LF` = `laplace_fit.py`; `AP` = `src/model/age_priors.py`; `AF` = `src/model/age_forward.py`; `AFi` = `src/model/age_fields.py`; `DL` = `src/model/data_loading.py`; `meta` = the `_vlF` input metadata.

---

## A. METHODS_DRAFT.md

| line | claim (short) | verdict | evidence | correct statement / note |
|---|---|---|---|---|
| 10-13 | Current habitat-path input `_vlF` (land-share floor 1e-4); LA1 values measured on `_vl`, differing at ~170 coastal cell-band pairs, max .31 | CONFIRMED | RESOLVED_CONFIG `environment.AGE_MODEL_CONFIG` = `J200VW_ALL_local.json` (`_vl`); overlay diff = `input_dir` only; `_vlF/KERNEL_DISCRETISATION.json` land_share_floor 1e-4; S:10584-10587 (172 pairs, max .31) | |
| 11-12 | "The candidate is being refit on it" | AMBIGUOUS | S:10606-10609: the `_vlF` all-data refit L0F1 has landed (201,573.1) and is the HMC slot-A init | Stale: say the refit (L0F1) exists. |
| 14-16 | Two pending decisions (pseudo-zeros; FIELD54 v1 → v2) | CONFIRMED | `LF:122-124,714-716`; DESK_PROVENANCE §d | |
| 22-24 | ESRI:102003, 67 × 112 cells, 54 km (2,916 km²), 4,318 land cells, t = 0…123 ↔ 1902–2025 | CONFIRMED | meta Ny 67, Nx 112, N_land 4,318, years; `config/overlays/grid54_data.json` | |
| 26-30 | s = 20; C = sN = expected count of median observer×route at route effect 0; RC priors ÷ s; K_min internal; no birds/km² in the likelihood | CONFIRMED | ovl pop_scalar 20; `AP:570-577,1126-1135,1464`; `AFi:33,378`; `LF:618-655` (route effects iid N(0,1), mean 0) | Strictly, C is for an experienced observer (FY = 0). The BBS mean is s[N_{t−1}+ρ_fl J_t], eq. (18). |
| 34-39 | Z from DESK, House Finch excluded; 3-step FIELD54 build (EMA half-life 10.9 yr; field correction on all 1966–2025 summer BBS rows; summer 64 dims) | CONFIRMED (provenance) | DESK_PROVENANCE rows 26-28, 41, 43, 91 [E]; not re-executable locally (TACC-only) | |
| 43-45 | "With iid coefficients over features, each linear predictor below is a GP with kernel w² Z(x)·Z(x′)" | WRONG | `AFi:216-263`: predictors are centred at z_c and split; `AP:1665-1697`: coupling factors act on f, g | As executed, H_ℓ for a, f, j has kernel w_ℓ² (Z_t−z_c)·(Z_t′−z_c), not w² Z·Z. H_K mixes its level direction β_K with its change direction w_K f, so its covariance has r_K cross-terms. The coupling factors (eq. 29) also make the coefficient prior non-iid. The uncentred Ružička kernel describes the feature map, not the executed predictors. |
| 47-49 | Path features: land-conditioned average over fractional displacements, water excluded, land share renormalised, 20 steps | CONFIRMED | ovl `path_integration_steps` 20, `_path_comment`; `build_vl.py` `zdisp_year` (integrate_paths, land_share_floor) | |
| 52-54 | Z̄, Z̄^d = 1996–2025 means; z_c = mean Z over distinct cell-years with a positive BBS count | CONFIRMED | `DL:479-485` (Z_ref, Z_disp_ref over ref_window); `DL:21-41` | |
| 56-66 | Eq. (1)–(2): w = softplus(ξ), r_K = expit(ξ_r); β_a = w_a f, β_f = w_f f, β_j = w_j f, β_K = w_K(r_K f + √(1−r_K²) g) | CONFIRMED | `AP:772-809,896-902`; `habitat_geometry.py:9-30`; trace `manifold_loadings` | |
| 71-79 | Eq. (3) split readout; level reads β_ℓ, change reads w_ℓ f; a, f, j reduce to w_ℓ h | CONFIRMED | `AFi:224-263`; `AP:950-962` (`w_env_within` = w_scale · f) | |
| 80-84 | Eq. (4) journey index h^d_k = (Z^d − z_c)·f | CONFIRMED | `AFi:293-294,400-404` with β_a = w_a f, β_j = w_j f | |
| 88-96 | Eq. (5)–(6): S_a, F, S_j = S_a expit(·), K⁰ = softplus, K = K⁰ + K_min, K_min = .0231039 internal | CONFIRMED | `AFi:321-378`; ovl `capacity_floor_internal` 0.02310390463595099; trace `K_flat` min .0231039 | |
| 99 | Eq. (7) Q_k = expit(α_a + w_a h^d) expit(α_j + w_j h^d) | CONFIRMED | `AFi:390,402-404` | |
| 106-107 | Q_k evaluated at the arrival cell [L] | CONFIRMED | `AF:229-232,266`: Q multiplies the convolved (destination) grid | The [L] tag can become [E]. |
| 109-110 | No trend or disease on K; K_min = .462 RC = 20 ñ₅₀ | CONFIRMED | trace `w_k_trend` 0; `AFi:367-368`; arithmetic | |
| 112-122 | Eq. (8) S_a^min, c_s, S_a^K, c | CONFIRMED | `AP:699-712,1797-1803` | |
| 122 | "Together they make S_a^K + S_jF/(1+c) = 1 at N = K" | AMBIGUOUS | `AP:706`: c = max(·, 0) | Holds (to 1e-6) only where F S_j > 1 − S_a^K. In sink cells c is clipped at 0 and the sum is < 1. |
| 124 | θ = 1.19 × 10⁻⁴ | CONFIRMED | trace `dd_survival_share` 1.1908e-4 | |
| 128-129 | State at end of year t after breeding; year t starts from (A_{t−1}, J_{t−1}) | CONFIRMED | `AF:605-873` | |
| 131-140 | Release t = 49 (1951), cell (27, 89), A⁰, J⁰ += R/2; eq. (9) R = .6852/20 = .03426; q₅₀ = median modern per-cell route count over surveyed cells; no exemption | CONFIRMED | `AF:613-617`; `AP:1464-1485`; meta inv_timestep 49, inv_locations[0] (27, 89); trace `inv_pop_relative` .03426; base `docs/PRIORS.md` q50 definition; ovl exemption 0 | |
| 142-149 | Eq. (10) Veit departure with u⁰ = N⁰/max(K, 1e-6), × (1 − ν w_nat), w_nat = I₀/max I₀; ν = .989 | CONFIRMED | `AF:320-345,631-644` (eps 1e-6); `AP:1844-1849`; trace `native_departure_r` .98856 | |
| 148 | I₀ is 1 in the core, .5 at the margin, 0 elsewhere | WRONG (minor) | meta `initpop_latent` on land: 350 cells = 1, 935 = .5, **124 at other fractions (.125–.875)**, 2,909 = 0 | I₀ also takes coarsening fractions in 124 cells. Same error in PRIORS 640-642, 677-678. |
| 151-158 | Eq. (11) movement; ε + 1e-6 edge correction; Q after convolution | CONFIRMED | `AF:229-232,347-376` | |
| 159-162 | Linear FFT with zero padding; 𝔽(ϖ) = ϖ₊³/max(ϖ₊² + τ², 1e-150), τ = 64 ε_mach max\|ϖ\| (stop-gradient) | CONFIRMED | `AF:109-129`; ovl numerics (fft_pad 0, ulps 64, smooth true); `AP:351-358` | |
| 163-165 | Adult kernel: isotropic area density exp(−(d/b_A)^.859), mdd 100 km | CONFIRMED | `build_kernels.py:66-102`; `dispersal_spec.py:22-29`; ovl | |
| 166-171 | 4 directions × 3 bands [0, 155.4), [155.4, 482.7), [482.7, ∞); V&L 2-D density ∝ exp(−(d/b)^a)/d, a .468, b 24.937 km (1-D mean 200 km); raised-cosine wedges, logistic band edges; cell-integrated | CONFIRMED | `build_vl.py:37-61` (raised cosine, logistic cdf), `:95-112` (exp(−(r/b)^a)/max(r, rmin), 31 sub-offsets); KERNEL_DISCRETISATION.json; b Γ(2/a)/Γ(1/a) = 199.9 km | |
| 172-173 | Adults no journey mortality; arrivers survive with Q in place of S_j | CONFIRMED | `AF:355-361,1191-1195` | |
| 175-179 | Eq. (12) N′ = A′ + J^s + J^m; Hill with +1e-12 | CONFIRMED | `AF:744-751` | |
| 182-193 | Eq. (13): m_max .8; 1[t ≥ 91]; o* = o + δ₀ + B·ω_lag; o_dec = (o + 1902 − 1993)/10; 24-function land-centred basis; smoothstep G over 3 yr; recovery | CONFIRMED | `AFi:100-134`; `disease_onset.py:54-77`; `AP:162,1716-1720`; meta disease_timestep 91, onset_decades = (onset−91)/10, sev_basis == lag_basis (24 × 4,318) | |
| 195 | LA1: k_h .0136 (.272 RC), n_h 2.18, ζ .0146, τ_rec 19.2, max Δ .431 | CONFIRMED | trace | |
| 197-200 | Eq. (14): u = N′/max(K, 1e-12) and 𝒜(N′), with N′ = A′ + J^s + J^m | **WRONG** | `AF:751` multiplies `juvenile_arriving` by (1−m) **before** `reproduction_age_structured` (`AF:776-786`), which sums `N_total_post` = A′ + J^s + J^m(1−m) (`AF:1170`) for both crowding (`AF:1173,1190`) and Allee (`AF:1175`) | Crowding ratio and Allee use N″ = A′ + J^s + (1−m)J^m. The Hill term (12) uses N′. They are identical before 1993 (m = 0). Same in PRIORS 452-455 and MODEL_SPEC §3.5. |
| 204 | Eq. (15) A_t = A′S̃_a + J^sS_j(1−m) + J^m(1−m), J_t = A_t F^eff | CONFIRMED | `AF:744-751,1189-1216` | |
| 207-208 | Mask and clip at 0; yearlings breed; F^eff = realised births per breeder | CONFIRMED | `AF:816-817,1185,1213-1221` | |
| 212-216 | Eq. (16) 𝒜 = 1 − (1 + γN/k_A)^−(k_A+1), k_A = 3, γ = s ln2/(n₅₀ + 1e-6) | CONFIRMED | `AF:1105-1117`; `AP:361-363,1318-1322,1506` | |
| 212 | 𝒜 "evaluated on N′ before survival" | **WRONG** | as row 197-200 | Evaluated on A′ + J^s + (1−m)J^m (before survival, after disease on arrivers). |
| 218-224 | n50_raw pinned −3.756179 → n₅₀ .0231039 RC = .2 × the 11-06 pin (−2.1); ñ₅₀ .0011552; γ = 600.0; 𝒜 = ½ at .819 n₅₀ = .01892 RC; unclumped form not used | CONFIRMED | trace `n50_raw` (observed), `allee_gamma` 600.00004; 3(2^¼−1)/ln2 = .8189; ALLEE_AGG_K = 3 | |
| 227-228 | Seed = I₀ K⁰(x, 0), pre-floor, core at full local capacity | CONFIRMED | `AP:1779-1783` | |
| 230-232 | Native-Mexico hull = union of ≤1970 western BBS hull and pre-1940 Mexican specimen hull; 1,409 seeded, 350 core | CONFIRMED | meta `native_hull_variant`; land counts 350 + 935 + 124 = 1,409 | |
| 234-238 | Spin-up 40 yr at 1902–06 means, Δ = 0, no release; N^sp split equally; spin-up age split discarded | CONFIRMED | `AP:1970-1986` (tile of first 5 years, dis zeros, inv_t = 1040, only `_spin[0][-1]` kept); `AF:544-545` | |
| 242-246 | Eq. (17) NB2 log pmf | CONFIRMED | `LF:412-413,618-622`; `specimen_lik.py:107` | |
| 248-249 | 191,895 rows: US/Canada 1966–2025, Mexico 2008–18, 68,704 pseudo-zeros 1902–39 | CONFIRMED | run1.log `rows` n 191,895, train_with_route 123,191; meta (via fork) | |
| 252-255 | Eq. (18) μ_i and φ_i = φ₀ q^tier max(μ, 1e-3)^κ | CONFIRMED | `AP:2061-2066,2090-2093,2113-2145`; `LF:618-622,679-686` | |
| 257-258 | N_{t−1}; N_{−1} ≡ N^sp | CONFIRMED | `AP:2064-2065` (`initpop_seeded` = `initpop_spun`) | |
| 259-265 | ρ_fl 3.7e-4; η; e_r N(0,1) over 19,127 routes, SD σ; φ₀, q (236 tier-1 rows), κ; pseudo-zeros have no route term | CONFIRMED | trace; `LF:263-264,643-652`; meta | |
| 267-275 | Eq. (19) Laplace: 30 Newton steps from 0 clipped ±1; 𝓘_r = observed information wrt log μ | AMBIGUOUS | `LF:638-655` | The formula is right, but the derivative is total: φ_i depends on μ_i through κ (`LF:620`). Writing NB2(y_i \| μ_i, φ_i) suggests φ is held fixed; say φ_i = φ_i(μ_i). |
| 277-281 | Eq. (20) BBS term: route-marginal + pseudo-zeros at e = 0 | CONFIRMED | `LF:640-655` (f sums all training rows; `on` masks the route term) | |
| 283 | L_BBS = 165,500.34; σ = 1.196 | CONFIRMED | EXECUTED_CODE §2; log σ .17879 → 1.1958 | |
| 285 | NestWatch 1,089 attempts in 330 cells, 1902–2020 | AMBIGUOUS | `DL:378-414` filter; fork recomputation: kept attempts span 1968–2020 | 1902–2020 is the filter window; the data are 1968–2020. |
| 286-287 | X_n is a 0/1 design over five substrate/record covariates | CONFIRMED | `DL:366-375`; ovl covariates | |
| 288-294 | Eq. (21) nest terms | CONFIRMED | `AP:649-666,2017-2046` | |
| 296-301 | n_br 2.5 pinned, 32-day cycle; w^exp, w^cnt construction [U] | AMBIGUOUS | `DL:404-405` | Resolvable: w^exp = 1[has_exp], w^cnt = 1[outcome = success and fledged ≥ 1]. |
| 303-312 | Eq. (22)–(23) CBC source W and NB mean / size with latent effort | CONFIRMED | `cbc_obs.py:64-78`; `LF:397-424` | |
| 315-320 | m^mig .57 (east of 95°W) / .14 / .06 (Pacific); P kernel LogNormal(log 265, .83), bearing 218°, κ 6.24 east, LogNormal(log 337, .83) otherwise; "both were fitted to BBL summer–winter pairs" | AMBIGUOUS | `cbc_obs.py:24-26,64-70`; run1.log kappa_east 6.2424 | Values right. Only κ_east is computed from the pairs CSV at run time (`LF:386-391`). Shares, distance and bearing are hard-coded from an offline fit (p14_bbl_sw.py). The Pacific region is lon < −117. |
| 321-322 | Missing effort: ψ̄(t) hat-basis trend, V^E = τ_E² = .836² | CONFIRMED | `LF:347-363`; run1.log tau .836 | |
| 323-325 | s_C, p_C, φ_C, γ_C; v_ι N(0,1) Laplace, 40 clipped Newton steps | CONFIRMED | `LF:417-424` | |
| 327 | L_CBC = 9,212.40 | CONFIRMED | EXECUTED_CODE §2 | |
| 329-346 | BBL: 8,778 cohorts / 543 recoveries; eq. (24) with Hill on N_t; S^B at cohort cell and year t_band + τ; eq. (25); 3 regions × 4 eras (<1980, 1980–94, 1995–2005, ≥2006) | CONFIRMED | `LF:489-532` (digitize [1980, 1995, 2006]; yidx = by + τ); run1.log bbl | |
| 348 | L_BBL = 4,666.49 | CONFIRMED | EXECUTED_CODE §2 | |
| 350-363 | Specimens: 5,522 rows, 582 positive; eq. (26) N̄ = sΣT N, μ = s_era E^β 𝒟(N̄), 𝒟 = 1 − (1 + N/k_P)^−k_P; NB2 φ_S; two eras; 0/1 collector weight | CONFIRMED | `specimen_lik.py:96-113`; `LF:752-762`; run1.log; npz w ∈ {0, 1} | |
| 360 | "The normalisation of T is [U]" | AMBIGUOUS | `specimens_cache_collectors.npz` tmask (5,522 × 124): every row sums to exactly 1 | Resolvable: T is the mean over the decade's model years. The 1880–1901 row reads the 1902 state (`specimen_lik.py:9`). |
| 365 | L_spec = 2,415.28 | CONFIRMED | EXECUTED_CODE §2 | |
| 367-373 | Absences: 74,306 cell-years, t ≤ 1965, eq. (27), k_P = .43724 | CONFIRMED | npz t max 63, w ≡ 1; `LF:160,711-714` | |
| 375-380 | Eq. (28) native viability on λ^pot = S_a + S_jF at t = 0; Mexico 88, US 257 cells; scale .05 | CONFIRMED | `LF:589-598`; run1.log | |
| 384-406 | Table 1 (every prior location / scale / family) | CONFIRMED | trace (all model sites); `LF:423` (CBC), `:532` (BBL), `:616` (σ), `:758-762` (specimens) | |
| 408-409 | β_E: one β(1−β) factor on the logit coordinate → uniform target; comment "Beta(2,2) + Jacobian" mismatches | CONFIRMED | `LF:762`; `run_hmc` adds no Jacobian for extra blocks (`LF:40-42`) | See code issue 2. |
| 411-415 | Fixed values (n50_raw, log R_tot, departure logits, m_max, n_br, K_min, s, k_P, k_A) | CONFIRMED | ovl `pinned_sites`; trace `dispersal_logit_*` NOTCOORD; `LF:269` | |
| 417-427 | Eq. (29) one-sided coupling factors; χ_coef, χ_map definitions; .598 → −30.57, .939 → 0 | CONFIRMED | `AP:1665-1697`; trace | The code clips χ to ±.999999. χ_map's maps are land-mean-centred, so the cosine is a Pearson correlation. |
| 429-431 | Age regulariser 100·mean log Beta(clip ω_K \| 1.01, 1.01), ω_K = F_K/(1+F_K), F_K = F𝒜(K)/(1+c); NestWatch in model density | CONFIRMED | `AP:619-629,699-712,1797-1823,2046` | |
| 435-442 | Eq. (30) objective | CONFIRMED | `LF:673-678,764` | |
| 435-437 | log 𝒫_model = "all prior sites, the factors of §8, and the NestWatch term" | AMBIGUOUS | EXECUTED_CODE §2; trace | Also contains constant log-probs of held / observed sites: `observer_route_eps` ZeroSumNormal at 0 (−17,575.6), `n50_raw` (−0.093) and the dispersal logits. Needed to reproduce 𝒰 = 201,574.84. |
| 444-447 | 233 coordinates = 212 model + log σ + 12 BBL + 4 CBC + 4 specimen; Normal whitened, Beta logit, Exp / HalfNormal log | CONFIRMED | trace site sizes sum to 212; `LF:661-671,789-846`; `local_map_coordinates.py:14-36` | |
| 449-453 | Eq. (31) U_HMC = 𝒰 − log\|J\| − log σ | CONFIRMED | `LF:40-42` | |
| 455-456 | Route and circle effects integrated; 𝒰 = 201,574.84, U_HMC = 201,703.58 | CONFIRMED | run1.log `DONE 201703.579 201574.842` | |
| 456 | "reproduced locally to 2 × 10⁻⁹ of the TACC value" | AMBIGUOUS | EXECUTED_CODE §1 (…440 vs …438) | The difference is 2e-9 **nats** absolute (≈1e-14 relative); say so. |
| 458-462 | L-BFGS on 𝒰; NUTS depth 8, whitened diagonal mass, 300 + 1,000, segments, 46 h | CONFIRMED | `LF:1024` (L-BFGS-B); RESOLVED_CONFIG `hmc_flags_TACC` (`--hmc_mass whitened_diag`, segment 25) | `run_hmc` itself was not executed in the coverage run (EXECUTED_CODE §1), so these settings come from the flags. The code default would be `dense_adapt`. |
| 463 | "The computation runs in float64; FFT kernels are stored as complex64" | AMBIGUOUS | meta kernel stacks complex64; RESOLVED_CONFIG `HOUFIN_Z_FLOAT32=1`; `DL:821-828` | Z and Z^d are also stored float32. The code comment says complex64 kernels make each convolution effectively single precision. |
| 465-466 | max\|∇U_HMC\| = 908 on **f**; Hessian pre-check found 3–5 negative-curvature directions at L-BFGS stopping points | CONFIRMED | EXECUTED_CODE §1 (`manifold_factor`); S:10468-10477 | The pre-check was on the masked LC19 and LC11 points, not LA1. |
| 470-474 | Masked twins via holdout_years + holdout_blocks (4 × 4 cells = 216 km); 31 pinned blocks + late years | CONFIRMED | S:2418 (2021–25 + 31 pinned 216 km blocks); ovl `_pinned_comment` | |
| 474-476 | Twins fitted by L-BFGS on unmasked rows; held-out = route-marginal NLL [U] | CONFIRMED | `LF:278-292` (train mask), `:1024` | |
| 477-478 | LC19 held-out 48,024.3; restart spread ~6–23 nats | CONFIRMED | S:10361, S:10399-10400 | |
| 480-483 | Trough D and ETF λ rise reported alongside (definition [U]) | CONFIRMED (records) | S:10374 | |
| 485-487 | All data 191,895 rows weight 1; all-data runs "include about 26.7k more BBS rows" | AMBIGUOUS | LC19 log: train 156,845 (route rows 96,463); LA1 log: 191,895 (123,191) | 26,728 more **survey** rows; 35,050 more rows in total, because the twins also withhold 8,322 pseudo-zero rows. |
| 491-499 | Sensitivity table run IDs and status | AMBIGUOUS (stale) | S:10555 (LC31r done, 201,596.8); S:10620, 10637 (LA5, LA6 done); S:10596 (LA7, all-data J200 ×.33, 201,558.5) | Of the table's statuses, LC31r "pending" and LA5–6 "queued" are out of date. LA7 is missing; it scores better than LA1p (201,573.2). The b_VL values are CONFIRMED (fork). |
| 503-511 | Pseudo-zeros counted on top of absences (inert, ~1–2 units); `--drop_pseudozeros`; v2 changes 219 cells | CONFIRMED | `LF:714-716` (`pass`), `:122-124,279-282`; S:10574; DESK_PROVENANCE §d | |
| 515-516 | §13: the renames "remove the remaining cases of one symbol carrying two meanings" | **WRONG** | see notation rows below | Several remain. |
| notation | τ: bare τ is the BBL interval (eq. 25), alongside τ_rec, τ_E and τ_FFT; §13 itself says "τ is reserved" | **WRONG** (internal) | METHODS L338-345 vs L529 | Rename the BBL interval (e.g. j or ι_B). |
| notation | q₅₀ (median route count, eq. 9) vs q (Mexico concentration multiplier, eq. 18) | AMBIGUOUS | L135-138 vs L254, L263 | Rename one (e.g. q_M). |
| notation | λ^pot (potential growth), λ^B (BBL reporting logit), λ_c (nest log concentration) | AMBIGUOUS | L375, L403, L401 | |
| notation | π_n (nest cycle success) vs π^B (reporting probability); Π (priors) | AMBIGUOUS | L289, L345 | |
| notation | Dummy v in G (L191) vs circle effect v_ι; dummy a in G(a) vs field label ℓ = a | AMBIGUOUS | L191 vs L311, L56 | §13 renamed other symbols for exactly these clashes. |
| notation | ω_K (juvenile fraction at K) vs ω_sev, ω_lag (basis weights) | AMBIGUOUS | L430 vs L190 | |
| notation | δ₀ (onset slack) vs δ_m, δ_s (nest covariates) | AMBIGUOUS | L188 vs L289 | |
| notation | Subscripted families: m, m_max, m_n, m^mig, m^B; s, s_n, s_C, s^S; γ vs γ_C; κ vs κ_bear | AMBIGUOUS | throughout | Distinguished only by decoration. §13's rationale ("m mortality", "s gauge") is applied inconsistently. |
| notation | Undefined or implicit: y_i (BBS count), t_ϱ (CBC year index), 𝟙^fail_n, ε_mach | AMBIGUOUS | L268, L307, L293, L162 | Define once. |

## B. config/age_model_config.json (comment edits)

| key | claim (short) | verdict | evidence | correct statement / note |
|---|---|---|---|---|
| (file) | JSON still valid; only `_`-prefixed values changed | CONFIRMED | Parsed HEAD (`git show`) vs working copy in WSL. Non-comment trees are identical. 21 changed + 1 added key (`_model_spec_comment`), all `_`-prefixed | |
| `_model_spec_comment` | LA1 = this file + overlay (54 km, rank 64, capacity_demography + coupling priors, clumped Allee pinned n50, floor, disease on survival, Veit + native departure, spin-up, single 1951 release) | CONFIRMED | ovl; EXECUTED_CODE §3 | |
| `_disease_arrival_map_comment` | capacity: gates K depression; survival: gates added mortality (3-yr smoothstep), K disease-free | CONFIRMED | `AFi:367-375`; `disease_onset.py:63-77`; ovl onset_gate compact_start 3 | The smoothstep is the overlay's onset_gate, not this file's default (sigmoid). |
| `kernel_contract._comment` | candidate overrides latent_dim = 64, readout.rank = 64 | CONFIRMED | ovl | |
| `_kernel_source_comment` | V&L-faithful exp(−(d/b)^a)/d, J200 b 24.937, 1-D mean 200; recorded only in KERNEL_DISCRETISATION.json; metadata dispersal_spec carries older labels | CONFIRMED | `build_vl.py:95-112`; fork (meta dispersal_spec) | |
| `_scale_comment` | likelihood gauge-invariant except capacity_floor_internal | CONFIRMED | `AFi:33,378` | Also up to the fixed numeric floors (1e-6, 1e-12, 1e-3). |
| `_release_allee_comment` | candidate sets 0, one release cell | CONFIRMED | ovl; `AP:1472-1474` | |
| `_disease_prior_comment` | m = .8·sigmoid(μ + b_late·dec + b_native·shape + basis·w_sev)·gate·(1−recovery)·hill(N′) on adult, stayer and arriver survival; K disease-free | CONFIRMED | `AFi:100-120`; `AF:744-751` | |
| `_capacity_level_comment` | 2.618 (27 km) here; candidate 2.582, alpha_k_scale 1.0, floor after link | CONFIRMED | base config; ovl; `AFi:378` | |
| `_allee_comment` | default 1 − 2^(−C/n50); candidate k = 3 per-capita 1 − (1 + ln2·C/(3n50))^−4, half at .819 n50, n50_raw −3.756179 pinned (.0231) | CONFIRMED | `AF:1105-1117`; ovl | |
| `_initpop_comment` | seed uses K before floor; with spin-up the 1902 state is the spun-up total (40 yr, 1902–06 means, no disease, no release) | CONFIRMED | `AP:1779-1786,1970-1986` | |
| `_invasion_pulse_comment` | candidate: once in 1951 (t 49), one cell, form single, log_inv_total 0 → .6852/20 = .03426 | CONFIRMED | trace `inv_pop_relative`; meta | |
| `_invasion_sites_comment` | candidate inputs carry 4 cells; single uses the first | CONFIRMED | meta inv_locations (4 × 2); `AP:1472-1474` | |
| `_invasion_budget_comment` | "Not active when invasion_pulse_prior.form = single **or log_inv_total is pinned**" | **WRONG** | `AP:1465-1495` | log_budget is unused only under form = single. `log_inv_total` exists only in that branch, so pinning it does not switch off the grid form's budget. Say "Not active when form = single (current candidate)". |
| `capacity_level_prior._link_comment` | "K = softplus(α_k + H_k [+ k_trend]) [+ floor]"; γ_k = 1; candidate no trend, floor .0231039 | AMBIGUOUS | `AFi:352-378` | This describes K⁰. Under this file's own default (target = capacity), K = K⁰·(1 − disease fraction) + floor. |
| `_manifold_comment`, `_amplitude_comment`, `manifold_prior._form_comment` | capacity_demography structure; amplitude_scale [1.0263, 1.0263, 1.0227, 1.0263]; capacity_demography ignores coupling_target, target_corr_*, communality, angle, fixed_coupling | CONFIRMED | `AP:772-809`; trace `w_scale_raw` | |
| `_k_range_comment` | floor bounds K when set; 54 km value 26 = exp(2 × 1.6277) | CONFIRMED | exp(3.2554) = 25.93; ovl 26 | |
| `_k_trend_comment` | disabled in candidate | CONFIRMED | ovl `k_trend.enabled` false; trace | |
| `_severity_comment`, `_ceiling_comment`, `_mu_comment` | survival target: Hill on current N′, hill_n ∈ (.5, 2.55) = .85(4/.8 − 2); severity_ceiling unused; prior-median peak .8 × .8 = 64% | CONFIRMED | `AP:1149-1166`; `AF:746-748`; expit(1.386) = .80 | |

## C. research/CONSIDERATIONS.md (G14, P7; main checkout)

| line | claim (short) | verdict | evidence | correct statement / note |
|---|---|---|---|---|
| 7 | 11-06 pin −2.1 = .1155 RC = .00578 internal = .177 [.089] birds/km²; half-point .145 [.072] | CONFIRMED | softplus(−2.1) = .11552; × 1.5325 (= 41.7 × .735/20) = .177; × .819 = .145 | |
| 8 | ×.5 = .089 [.044], ×.33 = .058 [.029]; half-points .072 [.036], .048 [.024] | CONFIRMED | arithmetic | |
| 9 | LA1 runs ×.2: n50_raw −3.7562 = .0231 RC = .00116 internal = .035 [.018]; half .029 [.015] | CONFIRMED | trace; arithmetic | |
| 10 | Candidate floor .0231039 internal = .71 [.35] birds/km², "~.3 ×" median occupied K | AMBIGUOUS | arithmetic CONFIRMED (.708, .354) | The bridge (41.7, .735) and "median occupied K 2.4" were measured on LC3 (pin −2.1, floor .0462; S:10185-10189), not on LA1. Say so. |
| 11 | Executed Allee is clumped per-capita, ½ at .819 n50; unclumped form not run; MODEL_SPEC §3.5 | CONFIRMED | `AF:1105-1117`; ALLEE_AGG_K = 3; MODEL_SPEC L177-186 | |
| 15 | P7: half-point .145 [.072]; LA1 = 0.2 × pin with floor .0231039 = 20 n50 in common units; label key 8 or 20, not read by code | CONFIRMED | ovl 8, TACC 20 (EXECUTED_CODE §4); grep src/ and LF: no reader | |
| 5-6 (unedited context) | "q = 41.7 … (pop_scalar 20 × Jensen factor e^{σ²/2} 2.06)" | AMBIGUOUS (pre-existing) | 20 × 2.06 = 41.2; `abs_calib.py:24-27` also carries median N_{t−1}/N_t ≈ 1.012 (fork) | q = 20 × median ratio (≈1.012) × 2.06. |

## D. docs/PRIORS.md, docs/TEMPORAL.md, docs/DATA_SOURCES.md

These rows come from a delegated pass with the same ground truth. Line numbers refer to the NEW files.

**Changes in this merge.**
- Two rows were overridden to WRONG after the census finding in §A (PRIORS 364-373 and 452-455).
- One row's embedded `|` was escaped.

**Additional sources.**
- `LA1/` = `grid54/sandbox/ROUTEMARG/LA1_ALLDATA_..._fromLC19/`.
- `S:` = SANDBOX_INVASION_SHAPE.
- `ovl_vl` = `J200VW_ALL_local.json`.

| document | line | claim | verdict | evidence | correct statement (WRONG) / note |
|---|---|---|---|---|---|
| PRIORS | 29-30 | Candidate runs base + overlay `J200VW_ALL_local.json` | AMBIGUOUS | LA1/result.json `config` = TACC `ALL_J200VW_norout_dds_spin_natdepweak_kf4_pc3.json`; its local mirror `ovl_vl` differs only in `input_dir` and `capacity_floor_n50_multiple` (diff run). The `_vlF` overlay that the doc calls current is `J200VWfl_ALL_local.json` (TACC `ALL_J200VWfl`, S:10606-10609) | Name both: LA1 = `J200VW_ALL_local.json` (`_vl`); the `_vlF` candidate (L0F1, HMC slot A init) = `J200VWfl_ALL_local.json`. |
| PRIORS | 33-39 | Candidate summary (54 km, rank 64, J200 V&L, capacity_demography + one-sided Sa–K coupling, k_A 3 per-capita, n50_raw −3.7562, floor .0231039, disease on survival, Veit + native departure, 1951 release, 40-yr spin-up, Laplace route effects, all data) | CONFIRMED | ovl keys; trace; EXECUTED_CODE §3 | |
| PRIORS | 40-44 | `_vlF` restores land-share floor 1e-4; `_vl` differs at ~170 coastal cell-band pairs, max .31; LA1 numbers from `_vl` | CONFIRMED | `_vlF/KERNEL_DISCRETISATION.json` land_share_floor 1e-4; S:10584-10587 (J200 172 pairs / 164 cells, max .31) | |
| PRIORS | 43-44 | "The candidate is being refit on `_vlF`" | AMBIGUOUS | S:10606-10609: L0F1 (all-data `_vlF`) already landed, 201,573.1, same basin as LA1p; slot A repointed | Stale at time of verification: the `_vlF` refit L0F1 exists and is the HMC init. |
| PRIORS | 45-48 | Two pending user decisions (pseudo-zeros; FIELD54 v1 → v2) | CONFIRMED | S:10574-10575, 10580, 10593; DESK_PROVENANCE §d | |
| PRIORS | 56-58 | s = 20; C = sN = expected count of median observer×route at route effect 0 | CONFIRMED | `age_priors.py:2058-2061` (×pop_scalar), LF:640-655 (route effects N(0,1), mean 0, no route on pseudo-zeros); ovl pop_scalar 20 | |
| PRIORS | 60-63 | Likelihood invariant under (K, initpop, founders)→Λ·, γ→γ/Λ, s→s/Λ; K only via N/K; Allee via γN | CONFIRMED [L] | N/K in `age_forward.py:338,1171,1189`; γ = s·ln2/(n50+1e-6) `age_priors.py:1322,1506`; absences/CBC/specimens use sN (LF:404-407,713,754); k_h scales via `counts_to_relative` | Holds up to the fixed numeric floors (1e-6, 1e-12) and, as stated, not for K_min. |
| PRIORS | 65-69 | K_min internal, added with no gauge conversion | CONFIRMED | `age_fields.py:33,370` | |
| PRIORS | 74-76 | 54 km q95 17.3, gauge left at 20 | CONFIRMED | ovl `_grid54_comment` | |
| PRIORS | 83-86 | n50, capacity level, q50, k_h follow the RC rule | CONFIRMED | `age_priors.py:570-577` (target/s), `:1464` (q50/s), `:1126-1130` (k_h counts_to_relative), γ via n50 RC | |
| PRIORS | 95-100 | target 2.582 = 3.0×2.2537/2.6183; α_K ~ N(−1.98192, 1.0); K⁰ prior median exactly 2.582 at H_K = 0 | CONFIRMED | ovl; trace `alpha_k` loc −1.981923627 scale 1.0; softplus monotone | |
| PRIORS | 121-128 | K⁰ = softplus(α_K+H_K), K = K⁰+K_min; γ_K = 1; no trend; no disease on K | CONFIRMED | `age_fields.py:353-370` (disease_target survival → K_val = K_base); `age_priors.py:1283-1284`; trace `w_k_trend` [0,0,0] | |
| PRIORS | 169-170 | candidate scale 1.0; base 2.6183 / 0.8 | CONFIRMED | ovl; base config | |
| PRIORS | 180 | K_min = .0231039 internal = .462 RC | CONFIRMED | ovl `capacity_floor_internal` 0.02310390463595099 | |
| PRIORS | 182 | Seed uses K⁰ (no floor) | CONFIRMED | `age_priors.py:1776` (`Kbase_flat[0]`) | |
| PRIORS | 183 | LA1 min K⁰ ≈ 6e-11, min K .0231 | CONFIRMED | trace `K_base_flat` min 5.92e-11, `K_flat` min .0231039 | |
| PRIORS | 186-191 | K_min = 20 ñ50; floor = .0462078/2; n50 = .2×softplus(−2.1) | CONFIRMED | arithmetic: softplus(−2.1) = .115519; ×.2 = .0231039; softplus(−3.756179) = .0231039 | |
| PRIORS | 193-195 | `capacity_floor_n50_multiple` read by no code; 8 local / 20 TACC; objectives agree | CONFIRMED | grep src/ + invasion_sandbox (no hits); `ovl_vl` vs LA1/tacc_overlay.json diff: only input_dir and this key; EXECUTED_CODE §1 (2e-9) | |
| PRIORS | 197-199 | K_min = .71 [.35] birds/km², ~0.3 × median occupied K; .0462 = 1.42 [.71] | AMBIGUOUS | Arithmetic CONFIRMED (.0231039×41.7×.735 = .708). But the bridge (41.7, .735) and "median occupied K 2.4" were measured on LC3 (pin −2.1, floor .0462, σ≈1.20), S:10185-10189, not on LA1 | State that bridge and median K come from LC3. |
| PRIORS | 201-204 | Floor sensitivity .0462078 (40 ñ50), .0231039, .0115520 (10 ñ50) | CONFIRMED | arithmetic; run dirs LC26-28, LB1-3, LA4p, LA5 present in ROUTEMARG | |
| PRIORS | 210-216 | capacity_demography: β_a=β_f=β_j direction f; β_K = w_K(r_K f + √(1−r_K²) g) | CONFIRMED | `habitat_geometry.py:9-30`; `age_priors.py:772-809`; trace `manifold_loadings` [1,0,1,0,.787,.617,1,0] | |
| PRIORS | 222-223 | ξ_r ~ N(1.7346, .15); LA1 r_K .787 | CONFIRMED | trace `capacity_correlation_logit`, `capacity_demography_correlation` | |
| PRIORS | 224-225 | Var(β_ℓ) = w_ℓ² for every field | CONFIRMED | unit-norm loadings `habitat_geometry.py:12-14` | |
| PRIORS | 231-234 | rank2/exchangeable keys ignored by capacity_demography | CONFIRMED | `age_priors.py:772-809` reads only capacity_correlation_* / fixed_capacity_correlation; EXECUTED_CODE §4 | |
| PRIORS | 240-243 | "mean .9 and sd .1 on the atanh scale" | AMBIGUOUS | `age_priors.py:1674-1680`: d = (atanh χ − atanh .9)/.1 | The location is atanh(.9) = 1.472 on the atanh scale (i.e. .9 on the correlation scale); only the sd .1 is on the atanh scale. Formula on L243 is correct. |
| PRIORS | 245-248 | χ_coef .598 → −30.57; χ_map .939 → 0; χ_map = cosine of land maps (Z̄−⟨Z̄⟩)β | CONFIRMED | `age_priors.py:1684-1697`; `data_loading.py:479-485` (Z_ref = 1996–2025 mean); trace `coupling_coef_prior` −30.565, `coupling_map_prior` 0 | |
| PRIORS | 250-253 | No Jacobian; VWC sd .05 no consistent gain | CONFIRMED | `numpyro.factor` on derived quantities; SENSITIVITY_SET (S:10381-10433, quoted not re-measured) | |
| PRIORS | 254-255 | Four-free = exchangeable, pairs all; no four-free run at ×.2 | CONFIRMED | ROUTEMARG listing: free4 runs only at pin/×.5/×.33 (FR*, FRA*, LC7, LC14); none `*free4*n50x020` | Exact four-free overlay keys are not diffed anywhere (SENSITIVITY_SET says so). |
| PRIORS | 259-266 | readout {split, 1996–2025, rank 64, within_shared demographic}; H_ℓ = (Z̄−z_c)·β_ℓ + (Z_t−Z̄)·w_ℓ f | CONFIRMED | ovl; `age_fields.py:226-231,247-258`; `age_priors.py:950-962`; trace `w_scale_within` = `w_scale` | |
| PRIORS | 262 | "Both [level and deviation] are centred at z_c" | WRONG | `age_fields.py:250-251`: H = (Z_ref·β − z_c·β) + (z_t − Z_ref)·β_w | Only the level is centred at z_c; the deviation is centred at the cell's own Z̄(x). z_c = mean Z over distinct cell-years with a positive BBS count (`data_loading.py:21-41`). |
| PRIORS | 268-270 | a,f,j reduce to w_ℓ h; K level own direction, change via f | CONFIRMED | as above (bs_w = w_a f since β_a = w_a f) | |
| PRIORS | 271-272 | "Space-for-time holds exactly for Sa, F, Sj. For K it holds only in time." | AMBIGUOUS | K: spatial coefficient β_K, temporal coefficient w_K f | Clearer: space-for-time holds for Sa, F, Sj; it does not hold for K, whose temporal response uses the demographic direction f while its spatial level uses β_K. |
| PRIORS | 273-274 | No separate within amplitudes; within_ratio_*, within_corr, within_tied not read | CONFIRMED | `age_priors.py:950` branch taken first; trace | |
| PRIORS | 280-288 | amplitude prior; loc unchanged; candidate scale [1.0263,1.0263,1.0227,1.0263] vs base [.5628,1.0263,1.0227,.5628]; LA1 w (2.742,5.399,6.155,.222) | CONFIRMED | trace `w_scale_raw`, `w_scale`; base config :97-108 | |
| PRIORS | 290-296 | slopes fixed at 1; softplus(.4778) = .960 = .687×1.399 | CONFIRMED | `age_priors.py:1267-1284`; arithmetic | |
| PRIORS | 314-322 | Sj = Sa·expit(α_j+H_j), α_j ~ N(.405,.5); Q_k = expit(α_a+w_a h^d)·expit(α_j+w_j h^d) at arrival cell; adults no journey mortality | CONFIRMED | `age_fields.py:325-326,353-404`; Q applied after convolution `age_forward.py:229-232` + juvenile_dispersal_vectorized; adult path `age_forward.py:355-361` has no Q; trace `alpha_j` | |
| PRIORS | 341-358 | dispersal_random false; resolved switches list | CONFIRMED | ovl; EXECUTED_CODE §3 | |
| PRIORS | 364-373 | 𝒜(N) = 1 − (1+γN/3)^−4 "on N′ before survival", multiplies F; γ = s ln2/(n50+1e-6) = 600.0; ñ50 .0011552 | WRONG (minor; formula and values CONFIRMED) | `age_forward.py:1105-1117,1175`; `age_priors.py:361-363,1322,1506`; trace `allee_gamma` 600.00004; `age_forward.py:751` vs `:1170` | 𝒜 is evaluated on A′ + J^s + (1−m)J^m (arrivers already reduced by disease), not on N′; identical before 1993. |
| PRIORS | 374-375 | n50_raw_loc −3.4915 / scale .25 inert under the pin | CONFIRMED | trace `n50_raw` obs NOTCOORD, lp −0.0931 | |
| PRIORS | 377-379 | 𝒜 = ½ at C½ = .819 n50 (.01892 RC) | CONFIRMED | 3(2^{1/4}−1)/ln2 = .8189; ×.0231039 = .01892 | |
| PRIORS | 383-385 | Retention table .18/.32/.52/.80/.94/.993/.999/≈1 | CONFIRMED | recomputed from the formula at n50 = .0231039 | |
| PRIORS | 401-404 | Pins −2.1 → .1155 (.00578); E055 −2.797 → .059; ×.2 → .0231 (.00116) | CONFIRMED | softplus arithmetic | |
| PRIORS | 406-411 | Bridge: 1 mean-route count = .735 [.367] birds/km²; 1 internal = 41.7 mean-route counts | CONFIRMED (as records) | S:10185-10188 (LC3); `abs_calib.py:20-34` | Computed on LC3, not LA1. |
| PRIORS | 411 | 41.7 = "pop_scalar × Jensen factor e^{σ²/2} 2.06" | AMBIGUOUS | 20 × 2.06 = 41.2, not 41.7. `abs_calib.py:27`: q = median(expected_obs_before_observer / simulated_density)·e^{σ²/2}; the median ratio also carries N_{t−1}/N_t (and ρ_fl J) ≈ 1.012 | q = 20 × median ratio (≈1.012) × e^{σ²/2} (2.06, LC3 σ). |
| PRIORS | 412 | 1 median-route count ≈ 1.53 [.77] birds/km² | CONFIRMED | 41.7 × .735 / 20 = 1.532; ×.367/20 = .765 | |
| PRIORS | 413-416 | V&L r/2 is where their mate-finding factor is ½ (not re-checked) | CONFIRMED [L] | The code's own V&L reproduction form: P = B²/(r + 2B), mated share 2P/B = ½ at B = r/2 (`age_forward.py:1196-1203`) | Paper not re-read. |
| PRIORS | 418-423 | Bridge table (n50 and half-point in birds/km², relation to V&L) | CONFIRMED | recomputed (×1.5325 and ×.819) | |
| PRIORS | 424 | Median occupied K 2.4 [1.2] | AMBIGUOUS | S:10189 (LC3 fit) | Value is LC3's, not LA1's. |
| PRIORS | 431-432 | floor .71 [.35]; .0462 = 1.42 [.71] | CONFIRMED | arithmetic | |
| PRIORS | 442-451 | S_a^min = clip(1−S_jF,.05,1); c_s, S_a^K, c formulas; θ ~ Beta(1,1), LA1 1.19e-4 | CONFIRMED | `age_priors.py:697-708,1797-1803`; trace `dd_survival_share` | |
| PRIORS | 452-455 | F^eff = F/(1+cu²)𝒜(N′), u = N′/K; S̃_a = S_a(1−m)/(1+c_s u²); J_t = A_t F^eff | WRONG (minor) | `age_forward.py:751` multiplies arrivers by (1−m) before `reproduction_age_structured` sums N_total_post (`:1170`) for crowding (`:1173,1190`) and Allee (`:1175`) | u and 𝒜 use A′ + J^s + (1−m)J^m; the Hill term uses N′ = A′ + J^s + J^m. Rest of the row CONFIRMED. |
| PRIORS | 456-457 | Age-structure prior 100 × mean log Beta(1.01,1.01) on ω_K, same constants | CONFIRMED | `age_priors.py:619-629,1809-1819` | |
| PRIORS | 466-469 | Adult kernel area density exp(−(d/b_A)^.859), b_A from 2-D mean 100 km | CONFIRMED | `build_kernels.py:66-102`; `dispersal_spec.py:22-29` | |
| PRIORS | 471-479 | J200 V&L form, a .468, b 24.937; 1/d regularised below 54/62 km; 4×3 bands; own-cell .210, P(>155) .368, P(>483) .109 (63/26/11%), grid mean 202.7; 31 subsamples | CONFIRMED | `_vlF/KERNEL_DISCRETISATION.json`; `build_vl.py:95-112` (rmin = cell/(2S)); DESK_PROVENANCE §b rows 49-50 (repro.out [E]); ovl kernel_discretization | Mass figures are from the `_vl` build log, same kernel. |
| PRIORS | 480-483 | Q_k from land-conditioned Z^d (20 steps); linear FFT, zero pad, edge corrections, smooth floor, complex64 | CONFIRMED | ovl path_integration_steps 20, numerics; EXECUTED_CODE §3 | |
| PRIORS | 485-489 | metadata `dispersal_spec` records mdd 200 / shape .468 / realised MDD 202.22; guard passes | AMBIGUOUS | meta `dispersal_spec` = {mdd 200, shape .468, splits,...}; 202.22 is in meta `path_feature_meta.realized_discrete_mdd_km`, not in `dispersal_spec` | Content correct; the 202.22 sits in `path_feature_meta`. |
| PRIORS | 491-495 | J330/J110 b 41.056/13.715; J60A30 lower bound | CONFIRMED | SENSITIVITY_SET; DESK_PROVENANCE row 49 | |
| PRIORS | 501-511 | Veit p = clip(u⁰ − softplus(10(u⁰−1))/10, 1e-6, 1)·(1−ν w_nat), both ages; ν ~ Beta(1.5,1.5) = .989; logit sites inert at 2.0/4.0 | CONFIRMED | `age_forward.py:333-345`; `age_priors.py:1844-1849`; trace `native_departure_r`, `dispersal_logit_*` NOTCOORD; LF:269 | |
| PRIORS | 512-513 | Overlay `_final` says Beta(17,3); Beta(1.5,1.5) runs | CONFIRMED | ovl `_final`; trace Beta(1.5,1.5) | |
| PRIORS | 519-527 | One release 1951 (t 49) at (27,89); R = .6852/20 = .03426; no exemption; 4 candidate cells, first used | CONFIRMED | meta inv_timestep 49, inv_locations [[27,89],[27,88],[26,89],[28,88]]; `age_priors.py:1465-1485`; trace `inv_pop_relative` .03426; ovl release_allee_exemption_years 0 | |
| PRIORS | 528-539 | q50 0.6852 (54 km) vs base 0.7083 (27 km, n 3791); inert pulse keys | CONFIRMED | ovl; base config; `age_priors.py:1465-1485` | |
| PRIORS | 549-557 | m = Δ·hill(N′); Δ formula; acts on S̃_a, S_j and arrivers; not on fecundity | CONFIRMED | `age_fields.py:100-120`; `disease_onset.py:54-77`; `age_forward.py:744-751`; `age_priors.py:1716-1720` | |
| PRIORS | 560-568 | m_max .8; t ≥ 91; o* = o+δ₀+B·ω_lag; o_dec = (o−91)/10; 24-function land-centred basis shared by severity and lag; compact 3-yr gate; disease_tau unused; Hill on N′ | CONFIRMED | meta disease_timestep 91; meta `disease_onset_decades` = (onset−91)/10 exactly; meta `disease_sev_basis` == `disease_lag_basis` (24×4318, max diff 0, land mean ≤3e-8); trace `disease_tau` 1.0 | |
| PRIORS | 570 | 56 free disease parameters | CONFIRMED | trace: 1+1+1+24+1+24+1+1+1+1 | |
| PRIORS | 577-579 | `disease_severity_map` = m_max·expit(·)·hill(K⁰ in 2025), diagnostic | CONFIRMED | `age_priors.py:1781-1782` (`Kbase_flat[-1]`); ceiling .8 `age_priors.py:162` | |
| PRIORS | 589-598 | log k_hᶜ ~ N(log 8,.7), LA1 .0136 (.272 RC); n_h = .5+2.05·expit(N(−.04879,.8)); bound .85(4/.8−2) = 2.55; median 1.5; LA1 2.18 | CONFIRMED | `age_priors.py:1126-1166`; trace | |
| PRIORS | 607-615 | μ_sev ~ N(1.386,.5), median modifier .8, peak .64; LA1 −1.050, max Δ .431 | CONFIRMED | trace; `disease_mortality_amp` max .43094 | |
| PRIORS | 631-634 | b_late ~ N(−.15,.15), LA1 −.401 | CONFIRMED | trace | |
| PRIORS | 640-642, 677-678 | I₀ = 1 in core, margin_fraction (.5) at fringe, 0 elsewhere | WRONG (minor) | meta `initpop_latent`: 350 cells at 1, 935 at .5, 124 at other coarsened fractions (.125–.875), 6,095 zero; `native_hull_build.py:10-11` ("coarsened margins kept") | On the 54 km grid I₀ also takes coarsening fractions (.125–.875) in 124 cells. |
| PRIORS | 645-651 | b_nat = −softplus(raw), raw ~ N(0,.7); LA1 −.570 | CONFIRMED | `age_priors.py:1097-1102`; trace | |
| PRIORS | 657-668 | δ₀ ~ N(0,2); ω_lag sd 1.5/√24; ω_sev sd .5/√24; ζ prior median 38%; log τ_rec ~ N(log 12,.5); LA1 ζ .0146, τ_rec 19.2; delay inert | CONFIRMED | trace (.30619, .10206); `disease_onset.py:54-57` | |
| PRIORS | 674-681 | Seed = I₀·K⁰(1902); `_nativeMX` hull = union of ≤1970 western BBS hull and pre-1940 Mexican specimen hull; 299 margin cells at .5 | CONFIRMED | `age_priors.py:1776`; meta `native_hull_variant` (tag "bbs<=1970 + specimens<1940", newly_seeded 299, margin .5); `native_hull_build.py:1-15,81-86` | |
| PRIORS | 683-688 | 40-yr spin-up at 1902–06 means, no disease/release; total split 50/50; 1902 BBS uses N^sp | CONFIRMED | `age_priors.py:1970-1986` (tile of first 5 years, disease zeros, inv_t = S+1000); `age_forward.py:544-545`; `age_priors.py:2058-2059` | |
| PRIORS | 718-724 | μ_i = max(s[N_{t−1}+ρ_fl J_t],1e-6)·exp(η FY + σ e_r) | CONFIRMED | `age_priors.py:2056-2061,2079-2082`; LF:618-622 | |
| PRIORS | 725-732 | ρ_fl Beta(1,4) (3.7e-4); η N(0,1); φ = φ₀ q^tier max(μ,1e-3)^κ; φ₀ Exp(1), q Beta(2,2) (236 tier-1 rows), κ HalfNormal(.5) | CONFIRMED | trace; `age_priors.py:2098-2113`; LF:618-622,670-671; meta obs_quality tier1 236 (2008–18) | |
| PRIORS | 733-740 | e_r N(0,1) over 19,127 routes, Laplace 30 clipped Newton steps from 0; σ free HalfNormal(1), LA1 1.196; sigma_u/sigma_mode inert; ZeroSumNormal held at 0; no zero-sum | CONFIRMED | LF:616,638-659,264; meta 19,127 routes; LA1/result.json sigma 1.19578 | |
| PRIORS | 741-742 | All 191,895 rows weight 1, no holdout | CONFIRMED | LA1 log `rows` n = train = 191,895; obs scale 1 (trace) | |
| PRIORS | 744-749 | 68,704 pseudo-zero rows at route effect 0 on top of absences; ~1–2 units; `--drop_pseudozeros` | CONFIRMED | LA1 log train_with_route 123,191; LF:640-655, 714-715 (`pass`); LF:122-124; S:10574 | |
| PRIORS | 753 | NestWatch 1,089 attempts in 330 cells, 1902–2020 | AMBIGUOUS | recomputed `nest_arrays` filter: 1,089 attempts, 330 cells, years 1968–2020 | 1902–2020 is the `year_range` filter; kept attempts span 1968–2020. |
| PRIORS | 755-768 | m_n = 2F^eff/n_br e^{Xδ_m}, n_br 2.5; covariates; 32-day cycle; NB2 on fledged−1; δ_m N(0,.1), δ_s N(0,.5), a_s N(3.5,1), λ_c N(2.3,1) | CONFIRMED | `age_priors.py:649-667,2015-2046`; ovl; trace | |
| PRIORS | 772-774 | "Every term has weight 1" | AMBIGUOUS | Term weights are 1 (bbs/cbc/bbl/nest/natal defaults), absences w = 1 (npz), but specimen rows carry 0/1 collector weights: 571 of 5,522 rows weight 0 (`specimens_cache_collectors.npz` w sum 4,951) | Say: all term weights 1; specimen rows have a 0/1 collector filter. |
| PRIORS | 776-780 | CBC: latent effort, BBL source, linear drift; 10,511 / 1,823; priors s N(0,5), p N(.6,.3), log φ N(0,1), drift N(0,1) | CONFIRMED | LF:345-424 (prior line 423); LA1 log | |
| PRIORS | 782-784 | BBL m-array with disease; 8,778 / 543; 3 regions × 4 eras, logit N(logit .005, 1.5) | CONFIRMED | LF:497-532; `bbl_adult_marray.npz` grp ∈ {0,1,2}; digitize [1980,1995,2006]; LA1 log | |
| PRIORS | 786-790 | Specimen counts 5,522 cell-decades, NB2; log s_e N(0,3); β_E Uniform(0,1) on the HMC target; log φ N(0,2); comment "Beta(2,2) + logit Jacobian" mismatches | CONFIRMED | LF:752-762: cost adds −log(β(1−β)) once on the logit coordinate → density uniform in β; run_hmc adds no Jacobian for extra blocks (LF:40-42) | |
| PRIORS | 792-794 | Absences 74,306 cell-years t ≤ 1965, cost −log NB2(0 \| sN, .43724) | CONFIRMED | npz t max 63, w all 1; LF:711-714; `--presence_k` default .43724203 | |
| PRIORS | 796-798 | Native viability: one-sided on mean log λ^pot (S_a+S_jF, 1902) in Mexico / US native regions | CONFIRMED | LF:589-598; `native_viable_cells.npz` (88 / 257) | |
| PRIORS | 809-817 | Splits [0,155.36,482.74,1e9] not quantiles; J200 bands ~63/26/11%; inputs built under these literals | CONFIRMED | ovl; `_vlF/KERNEL_DISCRETISATION.json` splits_km | |
| PRIORS | 827-849 | rank2 not active | CONFIRMED | EXECUTED_CODE §5: `_exchangeable_w_env` not executed, rank2 branch untaken; ovl form | |
| PRIORS | 856-867 | exchangeable not in candidate; variant adds coupling_priors on 6 pairs, fixed_coupling null, coupling_logit_scale 1.0 | UNSUPPORTED (variant keys) | SENSITIVITY_SET states the four-free overlay was not diffed; no four-free overlay checked | "Not active" part CONFIRMED. |
| PRIORS | 871 | k_trend disabled, w_k_trend deterministic 0 | CONFIRMED | trace | |
| PRIORS | 886-897 | invasion_sites 9 base; 4 cells in candidate inputs; exemption 0 | CONFIRMED | base config; meta inv_locations; ovl | |
| PRIORS | 906-907 | median_fraction .1 ⇒ prior-median total pulse across 9 sites ≈ .58 counts/yr | WRONG (minor, historical) | base q50 = 0.7083: 9 × .1 × .7083 = .64; the .58 (= 9×.1×.64) used an older q50 | ≈ .64 route counts per year at the base q50 .7083. |
| PRIORS | 911-933 | Disease on K (base) not active in candidate | CONFIRMED | EXECUTED_CODE §5: `disease_k_fraction` not executed; `age_fields.py:367-368` | |
| PRIORS | 937-947 | Logistic departure / net-exporter not active | CONFIRMED | `age_forward.py:333-339` (veit overrides) | |
| PRIORS | 952-966 | Unclumped Allee not active; old retention numbers; base prior N(−2,1); overlay N(−3.4915,.25) inert | CONFIRMED | ALLEE_AGG_K = 3; recomputed 42/75/94/99.6% at n50 .127; base config; ovl | |
| PRIORS (notation) | 788 | specimen era rate written s_e | AMBIGUOUS | METHODS_DRAFT §13 renames it s^S_era because "e is the route effect" | Use s^S_era as in METHODS_DRAFT. |
| PRIORS (notation) | 552 / 756 / 759 | m (disease mortality) vs m_n (nest mean); s (gauge) vs s_n (daily nest survival) | AMBIGUOUS | METHODS_DRAFT uses the same pairs while §13 claims double meanings are removed | Distinguished only by subscript; consider renaming m_n / s_n. |
| PRIORS (notation) | 126 / 290 / 367 | γ = Allee rate and γ_a…γ_K = slopes | AMBIGUOUS | NOTATION V4 proposed dropping the slopes; PRIORS still names them | Minor. |
| PRIORS (notation) | 457 / 659-661 | ω_K (juvenile fraction at K) vs ω_sev, ω_lag (basis weights) | AMBIGUOUS | both docs | Minor reuse of ω. |
| TEMPORAL | 20 | invasion_year 1940 base, 1951 candidate | CONFIRMED | `config/data_config.json:25`; `config/overlays/grid54_data.json` timeline.invasion_year 1951 | |
| TEMPORAL | 71-72 | Release 1951, index 49, one cell (27,89) | CONFIRMED | meta inv_timestep 49, inv_locations[0] | |
| TEMPORAL | 73-77 | established1951 derived by `kernel_sens.py derive`, only inv_timestep 38→49 | CONFIRMED | `INPUT_CONTRACT.json` (changed_metadata_keys [inv_timestep], 38→49); `kernel_sens.py:38-55`; DESK_PROVENANCE row 6 | |
| TEMPORAL | 78-82 | grid54_data overlay 1951; base 1940; fresh base ingest would release 1940 | CONFIRMED | as above | |
| TEMPORAL | 89-94 | disease index 91; Δ ≡ 0 before it; no K depression; data_config `timeline._comment` stale | CONFIRMED | meta disease_timestep 91; `age_priors.py:1716-1720`; `age_fields.py:367-368`; `data_config.json:22` | |
| TEMPORAL | 96-98 | onset = map + δ₀ + regional slack, 3-yr smoothstep | CONFIRMED | `age_fields.py:123-134`; `disease_onset.py:63-77` | |
| TEMPORAL | 104-106 | t = 0…123 ↔ 1902…2025; 40-yr spin-up at 1902–06 means, no disease/release | CONFIRMED | meta years; `age_priors.py:1970-1986` | |
| TEMPORAL | 107-109 | Z EMA half-life 10.9 yr + field correction; distinct from τ = 2 state-stream EMA "(docs/DATA_SOURCES.md)" | WRONG (citation) | Half-life: DESK_PROVENANCE §b row 43 [E]. The τ = 2 yr EMA is described in TEMPORAL.md:59-60 itself; DATA_SOURCES.md never mentions it | Cite TEMPORAL.md "Climate" section (states.streams[].ema_tau), not DATA_SOURCES.md. |
| TEMPORAL | 110-112 | Z̄ = 1996–2025 mean; BBS year t uses N_{t−1} | CONFIRMED | `data_loading.py:479-485`; `age_priors.py:2058-2059` | |
| TEMPORAL | 114-119 | absences t ≤ 1965; CBC 1902–65; specimens 1880–1940 with 1880–1901 read against 1902; pseudo-zeros pending | CONFIRMED | absence npz t max 63; LF:349 (≤ 1965); `specimen_lik.py:36,73-74` | |
| DATA_SOURCES | 35-36 | paths relative to `${HOUFIN_PROCESSED}/grid54` = E:\Datasets\houfin\grid54 | CONFIRMED | ovl `_paths_comment`; files found there | |
| DATA_SOURCES | 40 | Z = FIELD54 field54prod_ema s64 (v1), chain DESK EMA 10.9 yr → field_d → dims 0–63; 124×4,318×64, rank 64 | CONFIRMED (provenance) | DESK_PROVENANCE §a rows 26-28, §b row 43 [E]; ovl latent_dim/rank 64 | Not re-executed here. |
| DATA_SOURCES | 41 | Z^d current `_vlF` (build_vl.py --land_share_floor 1e-4), superseded `_vl`; ~170 pairs, max .31; 124×4,318×12×64 | CONFIRMED | ovl input_dir; KERNEL_DISCRETISATION.json; S:10584-10587; Z_disp file 1,644,847,104 B = 124·4318·12·64·4 | `build_vl.py` is untracked in the main repo (`git status` `??`). |
| DATA_SOURCES | 42 | "BBS US/Canada + Mexico … 191,895 rows" | WRONG | meta: 191,895 rows include the 68,704 route-less pseudo-zero rows of the next table row; surveyed rows = 123,191 (LA1 log train_with_route) | 123,191 survey rows (US/Canada 1966–2025 + 236 Mexico 2008–18); 191,895 including pseudo-zeros. 19,127 routes, 236 tier-1, 228 dropped are CONFIRMED (meta, `native_hull_variant`). |
| DATA_SOURCES | 43 | 68,704 pseudo-zeros = 1,808 cells × 38 yr | CONFIRMED | meta (pseudo cells 1,808; years 1902–1939) | |
| DATA_SOURCES | 44 | absences halo400_nativeMX, 74,306 | CONFIRMED | npz | |
| DATA_SOURCES | 45 | NestWatch 1,089 attempts, 330 cells, 1902–2020 | AMBIGUOUS | attempts span 1968–2020 | as PRIORS 753 |
| DATA_SOURCES | 46 | CBC file (TACC path), 10,511 / 1,823 | CONFIRMED | RESOLVED_CONFIG `cli`; LA1 log | |
| DATA_SOURCES | 47 | BBL pairs "fixes the CBC migrant-destination kernel" (East LN(log 265,.83), 218°, κ 6.24; others LN(log 337,.83)) | AMBIGUOUS | Values CONFIRMED (`cbc_obs.py:24-26`; LA1 log kappa_east 6.2424). But the LogNormal parameters, bearing and migrant shares are hard-coded in `cbc_obs.BBL`; at runtime the CSV only sets κ_east (LF:394-396) | Say the kernel was estimated offline from the pairs (p14_bbl_sw.py) and only κ is computed from the CSV at run time. |
| DATA_SOURCES | 48 | BBL 8,778 cohorts, 543 recoveries, to 2025 | CONFIRMED | npz; LA1 log | |
| DATA_SOURCES | 49 | specimens collector cache, 5,522 (582 positive), 1880–1940; filter rule [U] | CONFIRMED | npz; LA1 log | LA1 itself read the TACC copy `takeover_claude/inits/specimens_cache_collectors.npz` (identity with the local file not checked). |
| DATA_SOURCES | 50 | native viability 88 / 257 | CONFIRMED | npz | |
| DATA_SOURCES | 51 | disease arrival map 54 km tif | CONFIRMED | file present; meta disease_onset 92.7–105.0 (≈1994.7–2007) | |
| DATA_SOURCES | 52 | I₀: +Mexican hull, 299 margin at .5; 1,409 seeded (350 core) | CONFIRMED | meta initpop_latent / native_hull_variant | Fractional coarsened values also present (see PRIORS 640). |
| DATA_SOURCES | 54-57 | v2 states pending, 219 cells | CONFIRMED | DESK_PROVENANCE §d | |
| DATA_SOURCES | 59-63 | Reproducibility notes "(DESK_PROVENANCE §c)" | AMBIGUOUS | coord_code note is §c; the dispersal_spec and stale path_features notes are §d items 5 and 1 | Cite §c–§d. Content CONFIRMED. |
| DATA_SOURCES | 67-71 | base 27 km; grid54_data sets 54000 → 67×112, 4,318 land, ESRI:102003 | CONFIRMED | `grid54_data.json`; `data_config.json:31,34`; meta Ny 67, Nx 112, N_land 4,318 | |
| DATA_SOURCES | 72-73 | "see 'Data entering the current model' below" | WRONG (cross-ref) | that section is at line 32, above | "above". |

---

## Summary

### Verdict counts

| section | CONFIRMED | WRONG | AMBIGUOUS | UNSUPPORTED | rows |
|---|---|---|---|---|---|
| A. METHODS_DRAFT.md | 69 | 6 | 20 | 0 | 95 |
| B. age_model_config.json comments | 17 | 1 | 1 | 0 | 19 |
| C. CONSIDERATIONS G14 / P7 | 5 | 0 | 2 | 0 | 7 |
| D. PRIORS / TEMPORAL / DATA_SOURCES | 101 | 8 | 17 | 1 | 127 |
| **total** | **192** | **15** | **40** | **1** | **248** |

Every equation in METHODS_DRAFT §3–§9 was compared against the code that runs. Eqs. (1)–(13) and (15)–(31) match in functional form, signs, density, age class, time index and scale factors. The exception is the crowding and Allee census in (14) and §5. Every prior location and scale in Table 1 matches `trace_sites_LA1.txt` or the `laplace_fit.py` nuisance priors.

The config edit is clean. The file parses, and all 22 edited keys are `_`-prefixed comments; every non-comment value is identical to HEAD.

### The 10 most important problems

1. **Crowding and Allee census (METHODS eq. 14 and §5; PRIORS 364-373, 452-455; MODEL_SPEC §3.5).**
   - The docs say the crowding ratio u and the Allee factor 𝒜 use N′ = A′ + J^s + J^m.
   - The code applies disease mortality to juvenile arrivers (`age_forward.py:751`) before `reproduction_age_structured` (`:1170`). So u and 𝒜 actually use A′ + J^s + (1−m)J^m.
   - The Hill term uses N′. The two are identical before 1993.
2. **GP kernel statement (METHODS L43-45) is false as executed.**
   - The predictors are centred at z_c, so the kernel for a, f, j is w²(Z−z_c)·(Z′−z_c).
   - H_K mixes the level direction β_K with the change direction w_K f, giving r_K cross-terms.
   - The coupling factors (eq. 29) make the coefficient prior non-iid.
3. **§13's claim that no symbol carries two meanings is wrong.**
   - Bare τ is the BBL interval, although §13 says τ is reserved.
   - q is both q₅₀ and the Mexico concentration multiplier.
   - λ, π, ω and δ each have two or three meanings.
   - The gate's dummies v and a collide with v_ι and ℓ = a.
   - PRIORS still writes s_e.
4. **DATA_SOURCES L42: the BBS row count is wrong.** "BBS US/Canada + Mexico … 191,895 rows" includes the 68,704 route-less pseudo-zeros. The surveyed rows number 123,191.
5. **PRIORS L262: readout centring is wrong.** It says both parts are centred at z_c. Only the level is; the change term is Z_t − Z̄(x).
6. **The native seed shape I₀ is not three-valued (METHODS L148; PRIORS 640-642, 677-678).** 124 land cells hold coarsening fractions between .125 and .875.
7. **Stale run status.**
   - The `_vlF` refit (L0F1) has landed.
   - LC31r is done (201,596.8), and LA5 and LA6 are done.
   - The all-data J200 ×.33 run, LA7 (201,558.5), is missing from the sensitivity table, although it beats LA1p (201,573.2).
   - PRIORS L29-30 names LA1's `_vl` overlay as the candidate's.
8. **The absolute bridge comes from LC3 but is applied to LA1 without saying so (PRIORS, CONSIDERATIONS G14 L10).**
   - The bridge values (41.7 mean-route counts per internal unit, .735 birds/km²) and "median occupied K 2.4" were measured on LC3 (pin −2.1, floor .0462).
   - 41.7 ≠ 20 × 2.06; the script also carries an N_{t−1}/N_t factor of about 1.012.
9. **The `_invasion_budget_comment` logic is wrong.** "Not active when form = single *or log_inv_total is pinned*": the pin cannot switch off the grid form's budget.
10. **Imprecise quantities.**
    - "26.7k more BBS rows" counts survey rows only; 35,050 rows differ in total.
    - "2 × 10⁻⁹ of the TACC value" is absolute nats.
    - "float64" omits that Z and Z^d are stored float32 (and the kernels complex64).
    - The NestWatch data span 1968–2020, not 1902–2020.
    - The CBC migrant kernel is hard-coded from an offline BBL fit; only κ_east is computed at run time.
    - Several [U] items are resolvable:
      - T rows sum to 1 (a decade mean);
      - w^exp and w^cnt are 0/1 indicators;
      - Q at the arrival cell is established by the code.

The only UNSUPPORTED row is PRIORS L866-867, the four-free variant's exact overlay keys.

### Where the code itself is likely wrong (not the documents)

1. **Inconsistent census (`age_forward.py:744-751` vs `:1170`).**
   - Adults and juvenile stayers enter the crowding and Allee census before disease mortality; arrivers enter after it.
   - Their own disease mortality is applied later, through S_a(1−m) and S_j(1−m).
   - Probably unintended. It slightly lowers crowding and mate-finding after 1993 in cells fed by immigration.
2. **Specimen effort-exponent prior (`laplace_fit.py:762`).**
   - The cost adds −log β(1−β) once, which makes the target uniform on β_E.
   - The comment and `specimen_lik.py:19` intend Beta(2,2) plus the logit Jacobian, which needs the factor twice.
3. **Pseudo-zero replacement is a no-op (`laplace_fit.py:715-716`).** The "pseudo-zeros are replaced" branch is `pass`, so 1902–39 pseudo-zeros and range absences double-count early absence. This is the user's known, pending decision.
4. **`abs_calib.py:24-27` mixes years.** It divides the breeding-convention expected count (built from N_{t−1}) by N_t, biasing the bridge by about 1%. This is research tooling, not the model.

Two related points are design properties, not bugs:
- The BBL survival path evaluates the Hill term on the end-of-year N_t, while the forward model uses N′. The flag help documents this ("close").
- The equilibrium identity in eq. (8) fails where c is clipped at 0 (sink cells).

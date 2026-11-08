# Documentation changes, 2026-11-08 (alignment with the LA1 specification)

**Scope.**
- **Source of truth:** `docs/model_spec/` (MODEL_SPEC, NOTATION, RESOLVED_CONFIG, EXECUTED_CODE,
  DESK_PROVENANCE, SENSITIVITY_SET), plus the lead's facts: the `_vlF` input, the two pending
  decisions, and the Allee half-point.
- **What was not touched:** no source code, overlays, experiment records, TACC or other branches.
  Nothing was staged or committed.
- **Line endings preserved:** PRIORS.md, age_model_config.json and CONSIDERATIONS.md are LF;
  TEMPORAL.md and DATA_SOURCES.md are CRLF.
- **Backups** of the pre-edit files: `C:\Users\brend\AppData\Local\Temp\claude\23b2334d1d63b197\1d135948-c54f-4d07-94f0-cc6b964cfcd5\scratchpad\docs_agent_backup\`.

**Row count.** `DOC_DISCREPANCIES.md` has **79** rows (P1–P20, T1–T3, D1–D2, C1–C25, A1–A25,
R1–R4), not 87.

---

## 1. Files changed

### `docs/PRIORS.md` (rewritten in place; LF)

**Structure.**
- New lead section, "Which model this file describes": the candidate in one paragraph, the
  `_vlF` input (`_vl` superseded), the two pending decisions, and a pointer to `docs/model_spec/`.
- Sections for mechanisms that do not run were moved under a final **"Not active in the current
  model"** heading, keeping their history:
  - rank-2 prior;
  - exchangeable form (labelled as the sensitivity variant);
  - K trend;
  - invasion sites and invasion budget;
  - disease on K;
  - logistic departure and the "net exporter" argument;
  - the unclumped Allee form.
- Headings that config comments link to were kept, so every `docs/PRIORS.md#…` anchor in the
  base config resolves. The missing `#dispersal-kernels` anchor now exists (P20 / C14).

**Rows resolved.**

| Section | Rows | Change |
|---|---|---|
| The gauge | P1 | Invariance stated with its exception: `capacity_floor_internal` is in internal units. 54 km q95 = 17.3 added. RULE restated with the floor as the exception. |
| Capacity level, link, calibration | P2, P3 | 2.582 (54 km derivation from the overlay `_grid54_comment`), `alpha_k_scale` 1.0, prior N(−1.98192, 1). K = softplus(α_K + H_K) + K_min, γ_K = 1, no trend, no disease on K. The 27 km figures are kept and labelled. |
| Capacity floor (new) | P9, P19 | K_min = .0231039 internal = .462 RC = 20 ñ50. Not applied to the seed. The units trap. `capacity_floor_n50_multiple` is unread (8 vs 20). Birds/km² .71 [.35]. Sensitivity floors. |
| Manifold prior | P4, P6 | `capacity_demography` equations (MODEL_SPEC §1), r_K prior, LA1 r_K = .787, inert keys listed. Rank-2 moved to Not active. |
| Coupling priors (new) | P19 | One-sided atanh factors (MODEL_SPEC §7), LA1 values, the VWC variant. |
| Split readout (new) | P19, (A16) | `within_shared = demographic`: K's level uses its own direction, while its time change uses **f**. |
| GP amplitude | P5 | Candidate scales [1.0263, 1.0263, 1.0227, 1.0263]. The old percentile bands are labelled as base-only. |
| Juvenile survival (new) | P19 | Ratio form; Q_k. |
| Structural switches | P6, P19 | All four `manifold_prior.form` values. Link. `dispersal_random_enabled` false. List of resolved switches (EXECUTED_CODE §3). |
| Allee effect | P7, P8 | Clumped per-capita form (MODEL_SPEC §3.5). Pinned n50 = .0231 RC, γ = 600, half-point C½ = .819 n50, retention table at the pin. Fold location marked [U]. Pin history. PIF table recomputed with half-points (×1, ×.5, ×.33, ×.2). Floor values. |
| Density regulation (new) | P19, (A12, A14, A18) | c_s, c and θ (MODEL_SPEC §2, §3.5); yearling breeding; age-structure prior. |
| Dispersal kernels (new) | P20, P19 | Adult kernel (2-D area density; its 1-D → 2-D treatment [U]). J200 V&L-faithful juvenile kernels with b_VL, a_VL, bands and mass shares (DESK_PROVENANCE b). Metadata mislabel. Land-share floor and `_vlF`. |
| Departure (new) | P17, P19 | Veit form × (1 − ν w_nat), ν ~ Beta(1.5, 1.5), LA1 .989; inert logit keys; the `_final` Beta(17, 3) error noted. |
| Invasion pulse | P11, P12, P13 | One cell (27, 89), 1951, R = .03426 internal, total pinned, no exemption, 4 candidate cells with only the first used, q50 .6852 (54 km). The 9-site / 90-coefficient design moved to Not active. |
| Disease term, severity, modifier, onset and recovery | P14, P15, P16 | Added mortality on S̃_a, J^s and J^m with Hill on N′, m_max .8, n_h ∈ (.5, 2.55) with its prior construction, k_h in RC → internal, 64% prior-median peak, `disease_severity_map` as a diagnostic only, onset and recovery priors. The K-depression text moved to Not active. Late-era and native-resistance sections reworded for "mortality amplitude". |
| Initial population | P17, P18 | Seed = I_0 K⁰ (pre-floor) with the native-Mexico hull; 40-year spin-up; 50/50 re-split [U]; Veit departure. The current list of evidence for the Plains barrier, including the pending pseudo-zero decision. |
| Observation model (new) | P19, (A15, A21) | Breeding-form mean, ρ_fl, η, NB2 concentration, Laplace route effects with σ free (HalfNormal(1)), `sigma_u` inert, no holdout. **Pending decision 1** described neutrally. |
| NestWatch (new); Terms added by the fitting driver (new) | P19, (A22–A25) | Nest model and priors; CBC, BBL, specimens (β_E is uniform on the HMC target), absences, native viability. |
| Radial splits | — | Kept. Notes that the candidate uses the pinned literals, with the J200 V&L band masses (about 63/26/11%). |

### `docs/TEMPORAL.md` (CRLF)

| Rows | Change |
|---|---|
| T1 | Timeline table: `invasion_year` is 1940 in the base config and 1951 in the candidate. |
| T2 | Invasion section: the candidate fires at index 49 (1951) in cell (27, 89). The index was set by `kernel_sens.py derive` (inputs metadata 38 → 49), not by `temporal.invasion_timestep`. `config/overlays/grid54_data.json` sets 1951. Base `data_config.json` still reads 1940. |
| T3 | `disease_timestep` bounds the **added-mortality amplitude** in the candidate; there is no K depression. The data-config `_comment` wording is flagged as stale. |
| (new) | "How the age model uses the timeline (current candidate)": spin-up, DESK EMA (10.9 yr half-life) vs the state-stream EMA (τ = 2 yr), readout window, BBS N_{t−1} convention [L], pre-1966 evidence including **pending decision 1**. |

### `docs/DATA_SOURCES.md` (CRLF)

| Rows | Change |
|---|---|
| D1 | The grid paragraph now states 27 km (base) vs **54 km** (candidate, `grid54_data.json`; 67 × 112, 4,318 land cells, FIELD54 native 54 km). |
| D2 | New section "Data entering the current model": DESK Z (FIELD54 v1 chain), path features (**current `_vlF`**, superseded `_vl`), BBS rows and Mexico tier, pseudo-zeros (**pending decision 1**), range absences, NestWatch, early CBC, BBL pairs, BBL m-array, GBIF specimens, native-viability regions, disease arrival map, native seed shape, with sizes. **Pending decision 2** (v1 → `field54prod_v2_ema`, 219 cells) described neutrally. Reproducibility notes from DESK_PROVENANCE. |
| note | The audit's line references for this file (129, 108–125) do not match the working copy (the same text is at lines 30 and 9–26). The content matched, so the rows were applied. |

### `config/age_model_config.json` (comment values only; LF; JSON valid)

**Checks.** A script re-parsed the file and checked that all 119 non-comment leaves are unchanged,
in the same order. One comment key was added: top-level `_model_spec_comment`, which points to
`docs/model_spec/` and says that the base comments describe base values.

**Policy.** BASE comments were **correct for the base config's own defaults** (rank2, disease on
capacity, Allee k = None, latent_dim 24; DOC_DISCREPANCIES §4 preamble). Each comment therefore
keeps its base description and adds what the current candidate runs instead. They were not
rewritten to describe only the candidate, which would have made them wrong next to the base values
they annotate.

| Key | Rows |
|---|---|
| `_allee_comment` | C1 |
| `_disease_prior_comment` | C2 |
| `disease_prior._severity_comment` | C3 |
| `disease_prior._ceiling_comment` | C4 |
| `disease_prior._mu_comment` | C5 |
| `_disease_arrival_map_comment` | C6 |
| `_k_range_comment` | C7 |
| `capacity_level_prior._link_comment` | C8 |
| `_capacity_level_comment` | C9 |
| `manifold_prior._form_comment` | C10 (base copy) |
| `kernel_contract._comment` | C13 (base copy) |
| `dispersal._kernel_source_comment` | C14 (anchor now exists) |
| `_invasion_pulse_comment`, `_invasion_sites_comment`, `_invasion_budget_comment` | C15 |
| `_release_allee_comment` | C16 |
| `_initpop_comment` | C17 (base copy) |
| `_scale_comment` | P1 |
| `_manifold_comment` | P4 |
| `_amplitude_comment` | P5 |
| `_k_trend_comment` | P10 |

**Run-identity caveat.** `age_run_map._run_fingerprint` hashes the whole config, comments included.
So any MAP-checkpoint resume (`age_run_map`), or an HMC drift check (`hmc_common.check_model_drift`)
that compares against a fingerprint made from this base config, will see a change.
- The `laplace_fit.py` path used by the candidate and the `_vlF` refit does not use either
  (EXECUTED_CODE §5 coverage; `laplace_fit._resume` only reads the init npz).
- A running job already loaded its config.
- `HOUFIN_HMC_ALLOW_MODEL_DRIFT=1` is the documented override for comment-only edits.

### `research/CONSIDERATIONS.md` (main checkout; LF)

**G14 (R1, R2).**
- The "current pin" bullet is relabelled "2026-11-06 pin" and given its half-point.
- The ×.5 and ×.33 bullet gains its half-points. The user's "bracket" statement is unchanged.
- New bullet: the candidate's ×.2 pin with birds/km² and its half-point.
- The K-floor bullet gains the candidate floor (.71 [.35], ~.3 × median K).
- New "Model fact" bullet: the executed Allee is the clumped per-capita form, half-released at
  .819 n50; compare half-points; the bridge is unchanged.

**P7 (R3).** The pin's half-point (.145 [.072]) was added, plus a "Model fact" sentence: the
candidate runs 0.2 × this pin, with the floor at 20 × n50 in common units, and the label key is
unread. The user's sentence "The decision stands until the user revisits it" was **left
unchanged**. Whether the ×.2 candidate amounts to revisiting it is the user's call, so R3 is only
partly resolved.

**M32.** It was checked and contains no factual statement about the Allee half-point or units, so
it is unchanged.

**R4 (G6).** Left open. It is outside the items this task allowed (G14, M32, P7). Proposed text, for
the lead:
- "The age model consumes the FIELD54 EMA cube (64 dims, rank 64; DESK EMA + field correction)".
- "Pre-1940 data now include range absences, early-CBC counts (1902–65) and specimen counts
  (1880–1940), besides the pseudo-zeros (pending decision)".

### New files in `docs/model_spec/`

- **`METHODS_DRAFT.md`.** Manuscript-style Methods, §1–§13, with 31 numbered equations, a prior
  table, the selection and held-out design, the sensitivity set, the pending decisions, and the
  symbol rename table (§13).
- **`DOCSTRING_PATCHES.md`.** A1–A25 plus X1 (`src/temporal.py:116-123`, not in the audit), with
  current excerpt and proposed text. Appendix B gives overlay-only comment rows; Appendix C gives
  the data-config `timeline._comment`.
- **`DOC_CHANGES.md`.** This file.

---

## 2. Row status summary

| Status | Rows |
|---|---|
| **Resolved in docs** | P1–P20, T1–T3, D1–D2, R1, R2 |
| **Resolved in base config comments** (overlay copies still open) | C1–C10, C13–C17 |
| **Partly resolved** | R3 (model fact added; the decision status is the user's) |
| **Open: overlay-only comments**, proposed text in DOCSTRING_PATCHES App. B (overlays are out of scope) | C11, C12, C18–C25; and the overlay copies of C1–C10, C13–C17 |
| **Open: Python docstrings and comments**, proposed text in DOCSTRING_PATCHES (source edits not allowed) | A1–A25 |
| **Open: outside allowed CONSIDERATIONS items** | R4 |
| **Open: config value, not a comment** | `config/data_config.json` `invasion_year` 1940 (T1; the overlay supplies 1951); `capacity_floor_n50_multiple` 8 vs 20 (P9; unread). Not changed |

---

## 3. Inconsistencies found in the specification itself

1. **Symbol reuse inside NOTATION / MODEL_SPEC.** Each case below is one symbol with two meanings.
   They are resolved in METHODS_DRAFT §13 and used in PRIORS.md.
   - σ: route SD vs the logistic function;
   - r: K–demography correlation vs route index vs kernel distance;
   - x: land cell vs departure ratio (§3.2) vs HMC coordinates (§8);
   - θ: brake share vs model coordinates (§8);
   - u: crowding ratio vs route offset (§4);
   - B: disease basis vs broods;
   - m: mortality vs CBC effort trend m(t);
   - ℓ: field index vs log-likelihoods ℓ_n, ℓ_r, ℓ^B and CBC log effort ℓ_ϱ;
   - c: crowding constant vs CBC circle c(ϱ);
   - e: route effect vs edge corrections e_A, e_k vs nest exposure e^pre vs specimen era e(ϱ);
   - A: adults vs CBC source abundance A_ϱ;
   - g: capacity direction vs detection function g(N);
   - R: founders vs native-viability regions;
   - ϱ: CBC rows and specimen rows;
   - y_ϱ: CBC count vs year;
   - i, k: BBS row / kernel vs BBL cohort / interval;
   - H: habitat predictor vs Laplace curvature;
   - P: migrant density vs BBL cell probability;
   - C: route-count scale vs the costs C_abs, C_nv;
   - f: fecundity label vs coefficient direction.
2. **Undefined nest symbols.** MODEL_SPEC §5.1 uses w^exp_n, w^cnt_n, e^pre_n and d^last_n without
   defining them in NOTATION. METHODS defines the last two from context and marks the weights [U].
3. **Specimen terms.** MODEL_SPEC §5.5 uses an era index e(ϱ) that NOTATION does not list, and
   the normalisation of T is [U].
4. **BBL interval index.** MODEL_SPEC §5.3 does not state whether it starts at 0 or 1, or which
   cell S^B uses. METHODS marks S^B_{bτ} = S^B(x_b, t^band_b + τ) as [L].
5. **PIF bridge rounding** (records, not the spec). The Jensen factor is quoted as 2.06, but
   41.7/20 = 2.085. The factor also carries the σ of the fit it was computed on (LA1 σ = 1.196
   gives e^{σ²/2} = 2.04). abs_calib.py should be rerun for the candidate before birds/km² values
   are published.
6. **Ambiguous "20 n50".** NOTATION §1 writes K_min "= 20 n_50" in a row where both sides are in
   route counts (0.462 RC / 0.0231 RC). That is correct, but it reads like the mixed-unit
   statements V9 warns about. The docs use "20 ñ50 (common units)".

## 4. Observation for the lead (not a doc change)

The candidate's data flags carry **no `--cbc_presence`** (`cbc_presence = None` in
RESOLVED_CONFIG `cli`). Early CBC enters only as counts (`--cbc_counts`). The memory note "Presence
+ absence base" says every fit carries `--cbc_presence` AND range absences, so this may need a
check against the user's decision record.

## 5. Symbols added or renamed

Relative to NOTATION.md (full table in METHODS_DRAFT §13):
- expit(·); r_K; bold **f**, **g**;
- d (kernel distance), a_VL, b_VL, b_A (adult kernel scale), r_VL (V&L mate-finding parameter);
- u⁰ (departure ratio), υ (HMC coordinates), Θ (model parameters), Λ (gauge scale factor in
  PRIORS);
- ε_A, ε_k (edge corrections), τ_FFT, ϖ (dummy argument);
- n_br (broods);
- 𝓛_r, 𝓛_n, 𝓛^B (log-likelihoods) and L_• (NLL totals), with L_abs, L_nv replacing C_abs, C_nv;
- 𝓘_r, 𝓘_ι (Laplace information);
- CBC: ψ_ϱ, ψ̄(t), V^E_ϱ, ι(ϱ), v_ι, N^C_ϱ, μ^C_ϱ, φ^C_ϱ, yr_ϱ;
- specimens: ς, μ^S_ς, s^S_era, 𝒟(N);
- BBL: b, τ, n^B_b, m^B_{bτ}, t^band_b, reg_b, P^B_{bτ};
- E^pre_n (nest exposure days), 𝓡 (regions), 𝒳_abs (absence set), 𝒫 (model density), Π_coup
  (coupling factor), C½ = 0.819 n50.

---

## 6. Fixes after independent verification (VERIFICATION.md, 2026-11-08)

**Scope.**
- Ledger rows are cited as section + line (`A:197-200` = METHODS_DRAFT row 197-200, `D:PRIORS 262`
  = section D, PRIORS line 262, and so on).
- The lead also supplied facts: the census, β_E, the run status, the bridge provenance, the
  DATA_SOURCES numbers and the notation.
- **Same ground rules:** docs only, line endings preserved, config JSON valid, and all 119
  non-comment values unchanged (re-checked by script).
- **MODEL_SPEC.md** was edited only for the census fix (§3.5), as the lead permitted.

### 6.1 WRONG rows (15) — all fixed

| ledger row | fix | file(s) |
|---|---|---|
| A:43-45 (GP kernel w² Z·Z) | Rewritten. The Ružička kernel describes the feature map. The executed demographic predictors have covariance w_ℓ²(Z_t − z_c)·(Z_t′ − z_c). H_K has r_K cross-terms. The coupling factors make the prior non-Gaussian. | METHODS §2 |
| A:148; D:PRIORS 640-642, 677-678 (I₀ three-valued) | I₀: 350 cells = 1, 935 = .5, 124 at coarsening fractions .125–.875, rest 0. | METHODS §4, §6; PRIORS (native resistance, initial population) |
| A:197-200, A:212; D:PRIORS 364-373, 452-455 (census) | Crowding ratio and Allee use N″ = A′ + J^s + (1−m)J^m. The Hill term uses N′. N″ = N′ before 1993. Added an **Open convention question (pending, user)** box, and pending decision 3 in METHODS §12. | METHODS (14), §4 box, §5, §12; PRIORS (Allee effect, density regulation, opening list); **MODEL_SPEC §3.5** (equation, note, Allee bullet) |
| A:515-516 + notation τ (§13 claim; bare τ as BBL index) | BBL rewritten on calendar years t with recovery set 𝒯_b, so there is no τ index (the start interval is [U]). §13 now states an explicit convention: no *bare* symbol has two meanings, and decorated families are listed. | METHODS §7.4, §13 |
| B:`_invasion_budget_comment` | "Not active when invasion_pulse_prior.form = single (current candidate)". | config |
| D:PRIORS 262 (centring) | Only the level is centred at z_c; the deviation is centred at Z̄(x). | PRIORS (split readout); METHODS §2 |
| D:PRIORS 906-907 (.58) | Changed to .64 (9 × .1 × .7083). | PRIORS (Not active, invasion budget) |
| D:TEMPORAL 107-109 (citation) | Now cites this file's "Land use / soil" section (`states.streams[].ema_tau`). | TEMPORAL |
| D:DATA_SOURCES 42 (row count) | 123,191 surveyed rows (236 Mexico); 191,895 together with the 68,704 pseudo-zeros. | DATA_SOURCES; METHODS §7.1; PRIORS (observation model) |
| D:DATA_SOURCES 72-73 ("below") | Changed to "above". | DATA_SOURCES |

### 6.2 AMBIGUOUS rows resolved (where the ledger gave evidence)

| ledger row | fix |
|---|---|
| A:11-12; D:PRIORS 43-44 (refit "being run") | The `_vlF` refits have landed: L0F1 201,573.1, L0F2 201,637.5, L0F3 201,710.5. Inert (< 0.001 nats at fixed parameters). HMC slot A is at L0F1. |
| D:PRIORS 29-30 (overlay) | Both overlays named: LA1 = `J200VW_ALL_local.json` (`_vl`; TACC `ALL_J200VW_norout_dds_spin_natdepweak_kf4_pc3.json`); `_vlF` candidate L0F1 = `J200VWfl_ALL_local.json` (TACC `ALL_J200VWfl`). |
| A:26-30 note | C is defined for an experienced observer (FY = 0) as well as route effect 0. |
| A:106-107 | The Q_k arrival-cell statement is now unqualified (the ledger confirmed it). |
| A:122 | The replacement identity holds only where F S_j > 1 − S_a^K. In sink cells c is clipped at 0. |
| A:267-275 | φ_i = φ_i(μ_i); 𝓘_r uses the total derivative. |
| A:285; D:PRIORS 753; D:DATA_SOURCES 45 | NestWatch data span 1968–2020 (filter window 1902–2020). |
| A:296-301 | w^exp = 1[exposure record], w^cnt = 1[success with ≥ 1 fledgling]; 𝟙^fail defined. |
| A:315-320; D:DATA_SOURCES 47 | The CBC migrant kernel is hard-coded from the offline BBL fit (`p14_bbl_sw.py`). Only κ_bear is computed from the pairs CSV at run time. Pacific means west of 117°W. |
| A:360 | T rows sum to 1 (decade mean). The 1880–1901 row reads the 1902 state. |
| A:435-437 | log 𝒫_model also includes the constant log-probs of held or conditioned sites (`observer_route_eps` ≈ −17,575.6, n50_raw −0.093, departure logits). |
| A:456 | The match is 2 × 10⁻⁹ **nats** absolute. |
| A:463 | Z and Z^d are stored in float32. The complex64 kernels make convolutions effectively single precision. |
| A:465-466 (note) | The Hessian pre-check was on LC19 and LC11, not LA1. |
| A:485-487 | 26,728 more surveyed rows and 35,050 more rows in total (the twins also withhold 8,322 pseudo-zeros). |
| A:491-499 (stale sensitivity) | Table updated: LC31r 201,596.8; LA7 / LFA7 201,558.5; quarter-floor ×.2 all-data J200 201,571.9, J330 201,622.5, J110 201,675.5; `_vlF` L0F1–3. The candidate stays at ×.2 (user, HMC navigability); ×.33 is the likelihood-preferred neighbour. Also added to PRIORS (Allee effect). |
| A:470-474 | The withheld set is stated as 2021–25 plus 31 pinned 216 km blocks (S:2418, confirmed by the ledger); the [L] tag is removed. |
| A:417-427 note | χ is clipped to ±.999999. χ_map is a Pearson correlation (the maps are land-mean-centred). |
| notation q, λ, π, ω, δ, gate v / a, s_e vs s^S_era, undefined y_i / t_ϱ / 𝟙^fail / ε_mach | q → q_M; λ^B → logit π^B; λ_c → log φ_n; π_n removed ((S^d_n)^32 inline); ω_sev, ω_lag → **β**_sev, **β**_lag (ω_K is now the only ω); δ_m, δ_s → **β**^N_m, **β**^N_s and a_s → α_dsr (δ₀ is now the only δ); m_n → μ^N_n; s_n → S^d_n; G written with smoothstep(ϖ); s_e → s^S_era in PRIORS; y_i, y_ϱ, y_ς, t_ϱ, 𝟙^fail_n and ε_mach defined. BBL recoveries m^B → y^B. Euler's e is replaced by exp(·) throughout METHODS. Same renames applied in PRIORS (nest, disease, observation). |
| notation: decorated families (m, s, γ, κ, n, p) | The §13 convention paragraph now lists them explicitly as decorated families. Each decorated symbol has one meaning, and no bare symbol is reused. |
| B:`_link_comment` | Now describes K_base and states K = K_base·(1 − disease fraction) [+ floor] under this file's default, and K_base [+ floor] under target = survival. |
| B:`_disease_arrival_map_comment` (note) | The 3-yr smoothstep is attributed to the candidate overlay's onset_gate. |
| C:10, C:5-6; D:PRIORS 197-199, 411, 424 (bridge provenance) | The bridge (q = 41.7, .735) and the median occupied K (2.4) are stated as measured on **LC3** (pin −2.1, floor .0462, held-out-masked). q = 41.7 is the measured quantity; the "20 × 2.06" decomposition is rounded (median N_{t−1}/N_t ≈ 1.012); LA1's σ gives about 2.04. **Recommend rerunning `abs_calib.py` on the candidate before publishing birds/km².** Applied in PRIORS and as a model-fact bullet in CONSIDERATIONS G14. |
| D:PRIORS 240-243 | Coupling location: correlation .9 = atanh 1.472; sd .1 on the atanh scale. |
| D:PRIORS 271-272 | Space-for-time holds for Sa, F, Sj and not for K. |
| D:PRIORS 485-489 | 202.22 km is in the embedded `path_feature_meta`, not in `dispersal_spec`. |
| D:PRIORS 772-774 | Term weights are 1. Specimen rows carry a 0/1 collector weight (571 of 5,522 at 0). |
| D:PRIORS 788 | s_e → s^S_era. |
| D:PRIORS 126/290/367 (γ slopes) | The slopes are now named by code names (`gamma_a`…), not γ. |
| D:DATA_SOURCES 59-63 | Now cites DESK_PROVENANCE §c–§d. |

### 6.3 Lead-supplied fact: specimen effort exponent β_E

The prior is now documented as **Beta(2, 2)** on β_E, sampled on the logit scale.
- The MAP objective carries the Beta(2, 2) density with no Jacobian (MAP convention).
- HMC mode adds the logit Jacobian (lead's 2026-11-08 fix in `laplace_fit.py`).
- Eq. (31) gains the term −log[β_E(1 − β_E)], which is about +3.0 at LA1. The quoted U_HMC
  = 201,703.58 predates the fix.

This was applied in METHODS (Table 1, note a, eq. 31), PRIORS (driver terms) and DOCSTRING_PATCHES
A25. The ledger rows A:408-409 and D:PRIORS 786-790 (CONFIRMED for the pre-fix code) are
superseded by this change.

### 6.4 UNSUPPORTED row (1)

| row | action |
|---|---|
| D:PRIORS 856-867 (four-free variant keys) | Now attributed to the records (S:9778-9781) and tagged [L]: the overlay was not diffed. It stays unverified until a four-free overlay is diffed. |

### 6.5 Still open

| item | why |
|---|---|
| `MODEL_SPEC.md` §6 β_E row ("Uniform(0, 1) on the HMC target") and NOTATION | Edits to MODEL_SPEC were permitted for the census fix only. That row describes the LA1-time code. The lead should add a dated note, or let METHODS (current) and the spec (as of LA1) differ explicitly. |
| MODEL_SPEC §5.3 / NOTATION symbol collisions | MODEL_SPEC and NOTATION are the source-of-truth files and keep their original symbols. METHODS §13 maps them. |
| BBL first-interval convention | [U]: whether the banding year is the first m-array interval was not established. METHODS writes the recovery set 𝒯_b to avoid asserting it. |
| Specimen collector-filter rule; adult-kernel 1-D → 2-D question; held-out scoring script; trough D definition | No evidence in the ledger or spec ([U] kept). |
| Overlay comments (Appendix B), Python docstrings (A1–A25), `data_config.json` `timeline._comment`, R4 (CONSIDERATIONS G6) | Unchanged scope restrictions. DOCSTRING_PATCHES A18 now also notes the census asymmetry. |
| Code-level issues listed by the ledger (census asymmetry, pseudo-zero no-op, `abs_calib.py` year mixing) | These are code, not docs. They are documented as pending user questions or recommendations. |

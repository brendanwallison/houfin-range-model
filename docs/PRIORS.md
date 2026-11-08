# Priors and parameterizations of the age-structured model

The reasoning behind each prior and parameterization in `config/age_model_config.json`.
Every section is named for the config path it documents, so the pointer in the config is
unambiguous.

**Why this lives here and not in the config.** A JSON `_comment` is never executed,
imported, rendered, or reached by a test, which is where documentation drift accumulates
unnoticed — two fields in that file had gone stale exactly this way. And
`age_run_map._run_fingerprint` embeds the whole config dict, comments included, so editing
a comment invalidates MAP-checkpoint resume: prose should not be load-bearing on run
identity. Short one-line knob descriptions stay in the config beside their values; the long
derivations are here.

**These are the current values' rationale, not a changelog.** Where a number was chosen
against a measurement, the measurement is named (usually
`scripts/diagnostics/bbs_abundance_quantiles.py`). Where an earlier parameterization is
mentioned, it is because the current one cannot be understood without the contrast.

## Which model this file describes (read first)

**Revised 2026-11-08 against the executed specification in [`docs/model_spec/`](model_spec/).**
The model facts below are taken from `docs/model_spec/MODEL_SPEC.md` (equations),
`NOTATION.md` (symbols and units) and `RESOLVED_CONFIG.json` (resolved values). Symbols follow
`NOTATION.md`, with the renames listed in `docs/model_spec/METHODS_DRAFT.md` ("Symbols added or
renamed"): `expit` is the logistic function (σ is reserved for the route-effect SD), r_K is the
capacity–demography correlation, and bold **f**, **g** are coefficient directions.

- **Current candidate (HMC).** It runs the base config with an experiment overlay on top. The
  overlay changes many values, so the base config's own values are **not** what runs. Where the
  two differ, this file gives the candidate's value and says so.
  - The current (`_vlF`) candidate is **L0F1**, the HMC slot-A init. Its overlay is
    `J200VWfl_ALL_local.json` (TACC `ALL_J200VWfl`).
  - The specification in `docs/model_spec/` was measured on **LA1**. Its overlay is
    `J200VW_ALL_local.json` (`_vl` input; TACC `ALL_J200VW_norout_dds_spin_natdepweak_kf4_pc3.json`).
  - The two overlays differ only in `input_dir` (and in the unread label
    `capacity_floor_n50_multiple`).
- **Candidate in one paragraph.** 54 km grid; FIELD54 DESK latent Z (rank 64); J200 juvenile
  dispersal kernel (V&L-faithful, 1-D mean 200 km); `capacity_demography` habitat form with
  one-sided Sa–K coupling priors; clumped mate-finding Allee with n50 pinned at 0.2 × the
  2026-11-06 pin (`n50_raw` −3.7562); absolute K floor 0.0231039 internal; epizootic as added
  mortality; Veit departure with a native-range reduction; one 1951 release; 40-year spin-up;
  BBS route effects integrated out by a per-route Laplace approximation; all data (no held-out
  mask) for final inference.
- **Habitat-path input.** The current input is the `_vlF` build
  (`grid54/inputs_shape/J200_established1951_vlF_nativeMX`). It restores the FIELD54 land-share
  floor (1e-4). The earlier `_vl` build lacked that floor and is **superseded**: the two differ
  only at about 170 coastal cell-band pairs (maximum absolute difference .31). At fixed
  parameters the objectives differ by < 0.001 nats, so the change is inert.
  - The `_vlF` all-data refits have landed: L0F1 (J200) 201,573.1, L0F2 (J330) 201,637.5 and L0F3
    (J110) 201,710.5.
  - The LA1 point values quoted in this file come from the `_vl` fit.
- **Pending decisions (user).** Three questions are open. Each is described in the sections it
  touches and marked *pending*:
  1. whether the 1902–39 pseudo-zero BBS rows are dropped (see Observation model);
  2. whether to move from FIELD54 v1 states to `field54prod_v2_ema` (see `docs/DATA_SOURCES.md`);
  3. the census convention for crowding and the Allee factor after 1993 (see Density regulation).
- **Sections that describe mechanisms that do not run** in the candidate are kept for history,
  at the end of this file, under **"Not active in the current model"**.

## The gauge

`population_model._scale_comment`

THE GAUGE. `population_scale_route_counts_per_relative_unit` (s = 20) converts model density N
(internal units) to the route-count scale C = sN: the expected BBS count of the median
observer×route, with route effect 0. It is a SAMPLER-CONDITIONING choice, not biology.

**Invariance.** The likelihood would be invariant under (K → ΛK, initpop → Λ·initpop,
founders → Λ·founders, γ → γ/Λ, s → s/Λ) for any factor Λ > 0, because:
- K enters the forward simulation only through N/K;
- the Allee term enters only through γN = ln2·C/n50.

**Exception in the candidate.** The K floor `capacity_floor_internal` (K_min = 0.0231039) is
declared in INTERNAL units and added to K with no gauge conversion. A change of s would
therefore change what the floor means, and the invariance and the rule below both fail for it.
Keep s = 20 while the floor is stated in internal units, or restate the floor in route counts
first.

**Where 20 comes from.** It was set from the q95 of MODERN (>2000) per-GRID-CELL mean House
Finch route counts. These are computed over cells rather than observations, because BBS sampling
density is spatially biased.
- On the 27 km grid, q95 = 19.4, rounded to 20.
- On the 54 km grid the same statistic is 17.3. The gauge was left at 20 (overlay
  `_grid54_comment`).
- The previous value, 210, was roughly the MAX per-cell mean, which is why it conditioned badly:
  normalizing by an extreme put the typical occupied cell at density 0.010 and the median cell
  at 0.003, i.e. all the mass in the bottom 1% of [0,1].
- At 20 the occupied bulk lands in [0.1, 1], with about 5% of cells above 1. That is harmless,
  since density is not a probability.

**RULE.** Because s is a free gauge choice, every absolute-scale prior should be declared in
ROUTE-COUNT units and divided by s at one boundary, so that changing s cannot change any prior's
meaning. These follow the rule: n50, the capacity level, the founder reference q50 and the
disease Hill half-density k_h. The K floor does not (see above).

## Capacity level

`population_model._capacity_level_comment`

The prior on K's continental LEVEL is stated in expected BBS route counts. It is divided by the
gauge in code, so the number means the same thing whatever pop_scalar is.

**Candidate value: `target_level_median_route_counts` = 2.582 (54 km grid).** The overlay
derives it as 3.0 × (54 km occupied median 2.2537 / 27 km occupied median 2.6183), following the
earlier shared-readout fit's target of 3.0 (overlay `_grid54_comment`). The prior is then
- α_K ~ Normal(−1.98192, 1.0), where −1.98192 = softplus⁻¹(2.582/20);
- so the pre-floor capacity K⁰ at H_K = 0 has prior median exactly 2.582 counts. The realised
  capacity K adds the floor (see Capacity floor).

**Base config value: 2.6183**, the 27 km occupied median. It was computed by
`scripts/diagnostics/bbs_abundance_quantiles.py` with:
- modern era ≥ 1990;
- occupied = ≥ 2 distinct modern years with a positive count;
- per-cell value = mean route count over that cell's modern records;
- n = 2424 of 17,209 land cells.

Re-run that script after any change to the BBS release, the grid, or the occupancy rule: this
anchor does not update itself.

**Note the scope.** The anchor uses occupied cells only. It constrains the AVERAGE capacity and
says nothing about which cells sit below it. Empty cells are INTENDED to be empty because
λ < 1 (survival/fecundity), not because K is small there, but nothing in this prior enforces
that. See `_k_range_comment` and figure 07's Allee-dead class.

## Capacity link

`population_model.capacity_level_prior._link_comment`

**What runs (candidate).** A SOFTPLUS link applied in DENSITY space (internal units), with an
additive floor:

    K⁰ = softplus(α_K + H_K),    K = K⁰ + K_min,    K_min = 0.0231039 internal

- There is no slope (γ_K is fixed at 1).
- There is no time trend (`k_trend.enabled` = false; see "K trend" under Not active).
- There is no disease term on K (the epizootic acts on survival; see Disease term).

The previous parameterization used exp: K = k_level · exp(γ_K·H_K + trend). Only the
transformation changed.

**Why density space.** softplus does not commute with scaling (softplus(ϖ)/Λ ≠ softplus(ϖ/Λ)),
whereas exp does (exp(ϖ)/Λ = exp(ϖ − log Λ)). A softplus link therefore has to be pinned to a
space, and density is the one the forward simulation uses. Applying it in route-count space
instead would put the argument near +2.8, softplus's LINEAR regime, and make the link's shape
depend on `population_scale_route_counts_per_relative_unit`.

**What it buys.** At typical densities the argument is near −2.1. That is deep enough in
softplus's exponential regime that the link closely tracks exp, but it becomes progressively
sub-exponential as density rises. On the 27 km anchors, softplus/exp is:
- 0.98 at 1 route count;
- 0.93 at 2.9 (the typical occupied cell);
- 0.86 at 7;
- 0.74 at 16 (q90 of occupied cells);
- 0.28 at 73 (q99).

So it weakly regularizes the long upper tail of abundance without any hard bound, which is the
point. Note it is a graded shrinkage everywhere, not identity-then-threshold.

**Historical note.** The original model was also softplus in density space, but with pop_scalar
at 210 the argument sat between −3 and 0, so the link was effectively exp. That is why a penalty
SUBTRACTED from that argument acted multiplicatively and without a floor, and annihilated
eastern carrying capacity.

## Capacity calibration

`population_model.capacity_level_prior._calibration_comment`

`target_level_median_route_counts` is the belief. It is stated in route counts so that it
survives a change of pop_scalar.

- **Solving the location.** α_K's prior LOCATION is solved at import by
  `age_priors._softplus_loc_for_median` as softplus⁻¹(target / pop_scalar). This is exact and
  closed-form: softplus is monotone, so median(softplus(X)) = softplus(median(X)) whatever
  `alpha_k_scale` is.
- **Why a median.** The target is a MEDIAN so it can be matched exactly. A mean target would have
  to be converted through a lognormal ratio that softplus only approximates.
- **Candidate values.** target 2.582, `alpha_k_scale` = 1.0, so α_K ~ N(−1.98192, 1).
- **Base config values.** target 2.6183, scale 0.8.

The scale sets the spread around the level and does not affect where the level sits. The median
statement refers to K⁰. The floor K_min is added after the link and is not part of this
calibration.

## Capacity floor

`population_model.capacity_floor_internal` (candidate overlay; not in the base config)

**Definition.** K = K⁰ + K_min, with K_min = 0.0231039 internal units = 0.462 route counts.
- K_min is added after everything else.
- It is **not** applied to the native seed, which uses K⁰ (see Initial population).
- In the LA1 fit, min K⁰ is about 6e-11 while min K = 0.0231. The floor therefore bounds the
  crowding ratio N/K and the departure ratio where K⁰ vanishes.

**Units trap** (NOTATION §1). K_min and n50 have the same number, 0.0231039, in different units:
- K_min is internal;
- n50 is in route counts.

In common units K_min = 20 ñ50 = 20 n50/s. The equality is arithmetic: the floor was set by
halving the absolute floor 0.0462078, and n50 by taking 0.2 × softplus(−2.1).

**The multiple label is not a setting.** `capacity_floor_n50_multiple` is not read by any code.
It reads 8.0 in the local overlay and 20.0 in the TACC overlay, and the objectives agree. Treat
K_min (internal) as the only floor setting.

**Absolute scale.** On the Partners-in-Flight bridge (see Allee effect), K_min = .71 [.35]
birds/km², roughly 0.3 × the median occupied K. The earlier floor, 0.0462078, was 1.42 [.71]
birds/km². Both the bridge and the median occupied K were measured on LC3 (the n50 pin, floor
.0462, held-out-masked), not on the candidate (see Allee effect).

**Sensitivity runs** (`docs/model_spec/SENSITIVITY_SET.md`):
- floors .0462078 (40 ñ50);
- the candidate .0231039 (20 ñ50);
- .0115520 (10 ñ50).

## Manifold prior

`population_model._manifold_comment` (base: rank-2) · candidate: `manifold_prior.form = capacity_demography`

**What runs (candidate).** Fields are ordered ℓ ∈ (a, f, K, j): adult survival, fecundity,
capacity, juvenile survival. This is the column order of `w_env`. Juvenile survival is APPENDED
so that `w_env[:,0..2]` keeps its meaning for positional readers. The coefficient vectors are:

    w_ℓ = softplus(ξ_ℓ),   r_K = expit(ξ_r),
    β_a = w_a f,   β_f = w_f f,   β_j = w_j f,
    β_K = w_K (r_K f + sqrt(1 − r_K²) g),     f, g ~ N(0, I_64) iid

- **One shared direction.** Adult survival, fecundity and juvenile survival read ONE shared
  direction **f**. Their coefficient correlation is exactly 1, and they differ only by amplitude
  and intercept.
- **K can depart.** K mixes **f** with its own direction **g** at correlation r_K.
- **Prior on r_K.** ξ_r ~ N(logit .85 = 1.7346, .15), so r_K has prior median .85. At LA1,
  r_K = .787.
- **Variances.** Var(β_ℓ) = w_ℓ² for every field, including K, so the uncentred-Ružička GP
  contract (iid coefficients across features) holds for every field.

The rationale for this form (K-only departure vs four free fields) is recorded in the research
records (`SANDBOX_INVASION_SHAPE_2026-11.md`, "FOUR FREE FIELDS"; comparison runs in
`docs/model_spec/SENSITIVITY_SET.md`). It is not restated here.

**Keys ignored by this form.** `fixed_coupling` (1.0), `coupling_target`,
`coupling_logit_scale`, `target_corr_*`, `communality_logit_scale` and `angle_scale` are present
in the resolved config but are not read by `capacity_demography`. They belong to the rank-2 and
exchangeable forms (see Not active).

## Coupling priors

`population_model.manifold_prior.coupling_priors` (candidate)

These are two one-sided factors on REALISED adult-survival–capacity coupling. Each is centred at
a correlation of .9 (atanh .9 = 1.472) with sd .1 on the atanh scale, and only penalises coupling
below .9 (χ is clipped to ±.999999):

    log factor = −½ · min( (atanh χ − atanh .9) / .1, 0 )²

- **χ_coef** is the cosine between the coefficient vectors β_a and β_K. At LA1, χ_coef = .598,
  which gives a factor of −30.57.
- **χ_map** is the uncentred cosine of the land-cell maps (Z̄ − ⟨Z̄⟩)β_a and (Z̄ − ⟨Z̄⟩)β_K. At
  LA1, χ_map = .939, which gives 0.

These factors sit on derived quantities and carry no Jacobian, so under HMC the effective prior
on (r_K, **f**, **g**, w) is their product with the coefficient priors above.
- **Tighter variant.** A variant with sd .05 ("VWC") was tried. The records report no consistent
  gain (`docs/model_spec/SENSITIVITY_SET.md`).
- **Four-free variant.** The variant uses `exchangeable` with `pairs = all`. There is no
  four-free run at the candidate Allee setting.

## Split readout

`population_model.readout` = {split: true, ref_window: [1996, 2025], rank: 64, within_shared: "demographic"} (candidate)

Each cell's latent Z_t(x) is split into a long-run level Z̄(x) (the 1996–2025 mean) and a
within-cell deviation Z_t(x) − Z̄(x). Only the level is centred at z_c, the mean Z over the distinct
cell-years with a positive BBS count. The deviation is centred at the cell's own Z̄(x).
With `within_shared = demographic`, every field's deviation reads the demographic direction **f**
with that field's amplitude:

    H_ℓ(x,t) = (Z̄(x) − z_c)·β_ℓ + (Z_t(x) − Z̄(x))·w_ℓ f

- **Demographic fields.** For ℓ ∈ {a, f, j} this is H_ℓ = w_ℓ h(x,t), with h = (Z_t − z_c)·f.
- **Capacity.** For K, the *spatial level* uses K's own direction (r_K f + sqrt(1−r_K²) g), and
  the *temporal change* uses **f**.
- **Space-for-time.** Space-for-time therefore holds for Sa, F and Sj. It does not hold for K,
  whose temporal response uses the demographic direction **f** while its spatial level uses β_K.
- **No separate amplitudes.** There are no separate within-cell amplitudes (`w_scale_within` =
  `w_scale`). The keys `within_ratio_*`, `within_corr` and `within_tied` are not read.

## GP amplitude

`population_model._amplitude_comment`

There is one amplitude per field, in the order [survival_adult, reproduction, capacity,
survival_juv]: w_ℓ = softplus(ξ_ℓ), ξ_ℓ ~ Normal(amplitude_loc_ℓ, amplitude_scale_ℓ). Since
Var(β_ℓ) = w_ℓ², H_ℓ is conditionally a GP with kernel w_ℓ² Z(x)·Z(x'). The lengthscale is
fixed inside the Ružička features; this is the magnitude parameter that pairs with it.

**Candidate values.**
- amplitude_loc = [−.0134, .0025, −.0034, .4778] (unchanged from base).
- **amplitude_scale = [1.0263, 1.0263, 1.0227, 1.0263]** (base: [.5628, 1.0263, 1.0227, .5628]).
- LA1 values: w = (2.742, 5.399, 6.155, 0.222).

**Why all four slopes are fixed at 1.** w_ℓ is the SOLE amplitude: all four per-rate slopes
(code names `gamma_a`, `gamma_j`, `gamma_f`, `gamma_k`; not the Allee rate γ) are fixed at 1. A free slope multiplying a unit-variance latent alongside an
amplitude forms an exactly flat ridge, which is ruinous for HMC step-size adaptation.

**Juvenile location.** amplitude_loc[3] = 0.4778 gives softplus(0.4778) = 0.960. This reproduces
the juvenile amplitude implied by the earlier γ_j·w product (0.687 × E[1 + HalfNormal(0.5)] =
0.687 × 1.399).

**Historical percentiles (base scales only).** The 5–95% amplitude bands quoted in earlier
versions of this file were:
- survival 0.69 [0.33, 1.25];
- reproduction 0.70 [0.17, 1.86];
- capacity 0.69 [0.17, 1.85].

They describe the base scales and do not hold for the candidate's survival entries, whose scale
is 1.0263 rather than .5628. Correlation is invariant to positive scaling, so the coupling
statements above do not depend on these scales.

Deliberately softplus(Normal) and NOT lognormal. Lognormal's tails are about 2× heavier at the
99.9th percentile for reproduction and capacity, and that is what destabilizes this model under
HMC.

## Juvenile survival

`population_model.juvenile_survival` = {form: ratio, ratio_logit_loc .405, ratio_logit_scale .5} (candidate)

Juvenile survival of non-dispersers is a fraction of adult survival:
S_j = S_a · expit(α_j + H_j), with α_j ~ N(.405, .5). Juvenile *journey* survival through kernel k
uses the path habitat index h^d_k at the arrival cell:

    Q_k = expit(α_a + w_a h^d_k) · expit(α_j + w_j h^d_k)

Juvenile arrivers survive with Q_k in place of S_j. Adult dispersers have no journey mortality.

## Structural switches

These switches choose between alternative model structures. The values below are resolved
values for the candidate; base-config defaults are given where they differ.

**`manifold_prior.form`.** The forms are `rank2` (base default), `exchangeable`,
`capacity_demography` (**candidate**) and `survival_breeding`.
- `capacity_demography` is described under Manifold prior.
- `exchangeable` is the four-free sensitivity variant: one coupling ρ for all pairs, with
  `coupling_priors` on all six pairs.
- `rank2` is described under Not active.

**`capacity_level_prior.link`.** The options are `softplus` (**candidate**) and `exp`. See
Capacity link: in density space the softplus argument sits near −2.1, where softplus ≈ exp, so
the LOW tail of K is equally cheap under both links, and exp removes the shrinkage of the HIGH
tail. α_K's location is re-solved per link, so the level median is the target under either.

**`dispersal_random_enabled`.** This is **false** in the candidate. Per-year dispersal-logit
noise at sd 0.001 is inert, and the logistic departure it would perturb does not run under Veit
departure (see Departure).

**Other switches resolved in the candidate.** These are recorded in
`docs/model_spec/EXECUTED_CODE.md` §3:
- `allee_prior.form = absolute`, `aggregation_k = 3`, `per_capita = true`;
- `disease_prior.target = survival`;
- `dispersal_departure.form = veit`;
- `native_departure.enabled`;
- `reproduction.yearling_breeding`, `crowding_power = 2`;
- `adult_survival_dd.enabled`;
- `spinup.enabled`;
- `observation.form = breeding`, with the observer effect and `conc_density_power`;
- `nest_likelihood.enabled`;
- `k_trend.enabled = false`;
- `invasion_pulse_prior.form = single`;
- `habitat_centering = occupied`.

## Allee effect

`population_model._allee_comment` · candidate: `allee_prior` = {form: absolute, aggregation_k: 3, per_capita: true}; `pinned_sites.n50_raw` = −3.756179

**What runs (candidate): the clumped, per-capita mate-finding factor.** It multiplies fecundity
and is evaluated before survival on the census N″ = A′ + J^s + (1 − m)J^m. Arrivers enter after the
year's added mortality, while adults and stayers enter before it (see Density regulation; N″ = N′
before 1993):

    𝒜(N) = 1 − (1 + γN/k_A)^−(k_A+1),   k_A = 3,   γ = s·ln2 / (n50 + 1e-6)

- **Gauge.** γN = ln2·C/n50 (up to the 1e-6), a function of the route-count scale C = sN alone,
  so the gauge cannot change what n50 asserts.
- **Value of n50.** n50 = softplus(n50_raw) is in **route counts** (median observer×route). In
  the candidate it is PINNED: n50_raw = −3.756179 gives n50 = 0.0231039 RC
  (ñ50 = n50/s = 0.0011552 internal) and γ = 600.0 per internal unit.
- **Prior keys.** The prior keys `n50_raw_loc` (−3.4915 in the overlay) and `n50_raw_scale`
  (.25) are inert while n50 is pinned: the conditioned site contributes a constant.

**n50 is not the half-point.** The factor is ½ at **C½ = 0.819 n50** (0.01892 RC at the
candidate pin). The formula 1 − 2^(−C/n50), which IS ½ at C = n50, is the unclumped
`aggregation_k = None` form. It is not what runs; see Not active.

**Retained fecundity 𝒜 at the candidate pin** (n50 = 0.0231 RC):

| C (route counts) | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.25 | 0.462 (= K_min) | 1 |
|---|---|---|---|---|---|---|---|---|
| 𝒜 | .18 | .32 | .52 | .80 | .94 | .993 | .999 | ≈1 |

**Location of the fold.** The location of the Allee fold (the density below which a cell cannot
grow) for the clumped form is not derived in the spec [U]. The earlier rule of thumb
"fold ≈ 1/γ = n50/ln2" was stated for the unclumped form.

**Two consequences to watch.**
1. n50 sets where a cell becomes Allee-dead (suitable but holding no population), which can
   dominate `07_realized_source_sink.png`.
2. The Allee term is what makes a low-density range front expensive to push across unsuitable
   ground, so a weaker brake buys cheaper barrier crossing. Check `17_barrier_crossing.png` and
   the Great Plains pause whenever n50 moves.

**Pinned n50 and literature units (revised 2026-11-08).** Experiment overlays pin n50
(`pinned_sites.n50_raw`) rather than sampling it.
- **Pin history.**
  - The 2026-11-06 user decision ("stronger Allee") pinned n50_raw at −2.1, i.e. n50 = .1155
    median-route counts (internal .00578).
  - E055's earlier pin was −2.797 (.059).
  - **The current candidate runs 0.2 × the 11-06 pin**: n50_raw −3.756179, n50 = .0231 RC,
    internal .00116.
- **The bridge.** To compare with Veit & Lewis 1996, use the ABSOLUTE bridge, not the old
  K-ratio transfer (.0197 × our median K). V&L work in birds/km²: their half-saturation is r_VL/2 = .025 / .075 / .125
  for mate-finding parameter r_VL = .05 / .15 / .25, with .15 chosen. The bridge calibrates the model's 2010–19 US + Canada
  total to Partners in Flight, about 40 M birds (20 M low):
  - 1 mean-route count = .735 [.367] birds/km²;
  - 1 internal unit = q = 41.7 mean-route counts. q is the measured quantity, and its
    decomposition as pop_scalar × Jensen factor e^{σ²/2} (2.06) is rounded: `abs_calib.py` also
    carries the median ratio N_{t−1}/N_t ≈ 1.012.
  - **Provenance.** The bridge was measured on **LC3** (n50 pin −2.1, floor .0462,
    held-out-masked), not on the candidate. LA1's σ (1.196) gives a Jensen factor of about 2.04.
    **Rerun `abs_calib.py` on the candidate before publishing birds/km² numbers.**
  - Hence 1 median-route count ≈ 1.53 [.77] birds/km².
- **Compare half-points with half-points.** The model's mate-finding factor is ½ at
  0.819 n50, not at n50. The V&L column below assumes, as research/CONSIDERATIONS.md G14 does,
  that V&L's r_VL/2 is the density at which their mate-finding factor is ½. That reading of V&L was
  not re-checked here.

| Setting | n50 (RC) | n50, birds/km² at PIF 40 M [20 M] | half-point C½ = .819 n50, birds/km² | Relation to V&L (half-points) |
|---|---|---|---|---|
| 11-06 pin (−2.1) | .1155 | .177 [.089] | .145 [.072] | above V&L's strongest (.125) at 40 M |
| ×.5 (−2.822) | .0578 | .089 [.044] | .072 [.036] | at V&L's chosen .075 (40 M) |
| ×.33 (−3.248) | .0381 | .058 [.029] | .048 [.024] | below V&L's chosen .075 |
| **×.2, candidate (−3.756)** | .0231 | .035 [.018] | .029 [.015] | near V&L's weakest (.025) at 40 M |
| Median occupied K (LC3 fit) | — | 2.4 [1.2] | — | V&L K* 3.8 |

V&L's numbers come from a 1-D spread fit. In 2-D, front curvature and the critical founding
radius (Lewis & Kareiva 1993) mean the same history needs a weaker Allee effect. Only the
direction of that correction is known, not a factor (research/CONSIDERATIONS.md M32).

**All-data likelihood profile** (J200, floor .0231):

| setting | objective | runs |
|---|---|---|
| ×.5 | 201,596.8 | LC31r |
| ×.33 | 201,558.5 | LA7; `_vlF` LFA7 201,558.5 |
| ×.2 | 201,573.2 | LA1p; `_vlF` L0F1 201,573.1 |

- At the quarter floor (.0116) and ×.2, the all-data objectives are J200 201,571.9, J330 201,622.5
  and J110 201,675.5.
- The candidate stays at **×.2**, chosen by the user for HMC navigability.
- ×.33 is the likelihood-preferred sensitivity neighbour, about 15 nats better at J200.

**The K floor does NOT follow n50** (see Capacity floor):
- the candidate floor is 0.0231039 internal = .71 [.35] birds/km²;
- the earlier .0462 floor was 1.42 [.71] birds/km².

Script: `research/exp/invasion_sandbox/abs_calib.py`. Rationale: research/CONSIDERATIONS.md
G14, M32, P7. Rerun abs_calib.py after any change to pop_scalar, route σ or the readout: the
41.7 factor carries the σ of the fit it was computed on.

## Density regulation

`population_model.reproduction` = {form: standard, yearling_breeding: true, crowding_power: 2}; `adult_survival_dd` = {enabled: true, theta_a: 1, theta_b: 1} (candidate)

**Constants.** Two crowding constants are computed from the disease-free rates so that, at
N = K and ignoring the Allee and disease terms, yearling-breeding replacement holds:
S_aᴷ + S_j F/(1 + c) = 1. With S_a^min = clip(1 − S_j F, .05, 1):

    c_s = θ · max(S_a/S_a^min − 1, 0),   S_aᴷ = S_a/(1 + c_s),
    c   = max(F S_j / (1 − S_aᴷ + 1e-6) − 1, 0)

- θ (`dd_survival_share`) ~ Beta(1, 1) is the share of the density brake carried by adult
  survival. At LA1 θ = 1.19e-4, so the survival brake is negligible and fecundity carries the
  brake.
- **Applied each year** with crowding ratio u = N″/K. The census is
  N″ = A′ + J^s + (1 − m)J^m: arriving juveniles are reduced by the added mortality m before the
  census (`age_forward.py:751` vs `:1170`).
  - F^eff = F/(1 + c u²) · 𝒜(N″);
  - S̃_a = S_a(1 − m)/(1 + c_s u²).
  - The Hill term of the disease uses N′ = A′ + J^s + J^m. N″ = N′ before 1993.
- **Open convention question (pending, user).** Adults and stayers enter the census before
  disease mortality, and arrivers after it. This is possibly unintended, and it matters only from
  1993 onward. The alternatives are to census every class before mortality (N′) or every class
  after it.
- **Yearling breeding.** All winter survivors breed: J_t = A_t F^eff.
- **Age-structure prior.** The local age-structure regulariser (`age_structure_prior`: Beta(1.01,
  1.01) on the juvenile fraction at K, ω_K, × effective_sites 100) uses the same constants.

## Dispersal kernels

`dispersal` (base: `_kernel_source_comment`) · candidate: J200 inputs (`_vlF` build)

The adult and juvenile kernels are both from Veit & Lewis 1996, but their SHAPES DIFFER
(0.859 vs 0.468), and in the candidate so do their 2-D forms.

- **Adult kernel 𝒦_A.** Isotropic, mdd 100 km, shape 0.859. Its 2-D form is the area density
  exp(−(d/b_A)^{0.859}), with b_A the scale implied by the mdd. Whether the same 1-D → 2-D translation issue found for the juvenile kernel
  applies to it is **open** [U] (DESK_PROVENANCE open question). Adult dispersers suffer no
  journey mortality.
- **Juvenile kernels 𝒦_k (J200, "V&L-faithful").**
  - Form: the 2-D density ∝ exp(−(d/b_VL)^{a_VL})/d, with a_VL = 0.468 and b_VL = 24.937 km. The
    1-D V&L mean is 200 km. V&L's own b, 41.056 km, gives a 329.3 km mean, which is the J330
    variant.
  - Regularisation: the 1/d factor is regularised below 54/62 km.
  - Structure: 4 directions (raised-cosine wedges) × 3 radial bands, [0, 155.4), [155.4, 482.7)
    and [482.7, ∞) km, with logistic band edges, giving n_ker = 12 kernels.
  - Mass: own-cell mass .210. P(d > 155) = .368 and P(d > 483) = .109, so the three bands hold
    about 63 / 26 / 11% of the mass. The realised 2-D grid mean is 202.7 km.
  - Kernels are cell-integrated (31 subsamples).
- **Journey habitat.** Juvenile arrivers via kernel k survive with Q_k, computed from the
  land-conditioned path features Z^d_{t,k} (20 integration steps). See Juvenile survival.
- **Numerics.** Linear FFT convolution with zero padding, divided by the land-fraction edge
  corrections ε_A, ε_k. A smooth round-off floor is applied. FFT kernels are stored complex64.

**The metadata does not describe the kernel truthfully.** The input metadata's `dispersal_spec`
still records the older area-density juvenile kernel labels (mdd 200, shape .468), and the
embedded `path_feature_meta` still records a realised MDD of 202.22 km. The guard `require_inputs_match_config` therefore passes while the kernel differs
(DESK_PROVENANCE inconsistency 5). Only `KERNEL_DISCRETISATION.json` in the input directory
records the kernel that runs.

**Land-share floor.** The FIELD54 lineage applies a land-share floor of 1e-4 to the path
features.
- The `_vl` build omitted it; `_vlF` restores it (see "Which model this file describes").
- J330 and J110 (b_VL 41.056 and 13.715 km) are the kernel sensitivity axis.
- J60A30 (no V&L-faithful rebuild) is a lower bound only.

## Departure

`population_model.dispersal_departure.form = veit`; `native_departure` = {enabled: true, beta_a: 1.5, beta_b: 1.5} (candidate)

Both age classes leave with the same probability. It is computed from the start-of-year crowding
ratio u⁰ = N⁰/max(K, 1e-6):

    p = clip(u⁰ − softplus(10(u⁰ − 1))/10, 1e-6, 1) · (1 − ν w_nat)

- **Shape.** The first factor is a smoothed min(N/K, 1) (Veit & Lewis 1996).
- **Native reduction.** w_nat = I_0/max I_0 is the native seed shape. ν ~ Beta(1.5, 1.5) reduces
  departure in the native range. At LA1 ν = .989, so native-core cells emit about 1% of the Veit
  rate.
- **Inert keys.** `dispersal_logit_intercept` and `dispersal_logit_slope` (held at 2.0 and 4.0,
  outside the HMC coordinates) and `dispersal_target_capacity_fraction` (.8) have no effect.
- **Stale overlay comment.** The overlay comment `_final` says ν ~ Beta(17, 3). What runs is
  Beta(1.5, 1.5).

## Invasion pulse

`population_model._invasion_pulse_comment` · candidate: `invasion_pulse_prior` = {form: single, n_sites: 1, n_years: 1}; `pinned_sites.log_inv_total` = 0

**What runs (candidate).** There is ONE release, in **1951** (timestep 49), into ONE cell,
(27, 89). The founders, in internal units, are added half to adults and half to juveniles:

    R = (q50/s) · exp(log R_tot) = 0.6852/20 · e⁰ = 0.03426 internal (= 0.685 route counts)

- **Founder count.** `log_inv_total` is PINNED at 0, so the founder number is not fitted.
- **No exemption.** There is no Allee exemption (`release_allee_exemption_years` = 0).
- **Unused candidates.** The inputs carry 4 candidate release cells, (27,89), (27,88), (26,89)
  and (28,88). Only the first is used.
- **The reference q50.** `global_q50_route_counts` is a FIXED, EMPIRICALLY OBSERVED,
  K-INDEPENDENT reference: the median, over all SURVEYED cells, of the modern per-cell mean route
  count.
  - Surveyed cells include cells whose modern counts are all zero, which are genuine observed
    absences.
  - The value is 0.6852 on the 54 km grid; the base config's 0.7083 is the 27 km value (n = 3791
    cells).
  - It deliberately does not use k_level or local K, either of which would tie a historical
    release to a fitted, moving quantity. It deliberately uses the all-surveyed median rather
    than the occupied-only one, since a release event is not an equilibrium population.
- **Inert keys.** While the total is pinned, these keys have no effect:
  `median_fraction_of_global_q50`, `log_budget`, `total_median`, `total_log_sd`.

The 1940 release date, nine candidate sites and per-(site, year) coefficients described in
earlier versions are under Not active ("Invasion sites", "Invasion budget"). The release year
moved from 1940 to 1951 with the `established1951` inputs (`docs/TEMPORAL.md`).

## Disease term

`population_model._disease_prior_comment` · candidate: `disease_prior.target = survival`

**What runs (candidate): the epizootic is ADDED MORTALITY, not a reduction of K.** K has no
disease term. Each year, after dispersal, the added mortality fraction is:

    m = Δ(x,t) · (N′ + 1e-12)^{n_h} / ((N′ + 1e-12)^{n_h} + k_h^{n_h})
    Δ(x,t) = 1[t ≥ 91] · m_max · expit(μ_sev + b_late o_dec(x) + b_nat I_0(x) + B(x)·β_sev)
             · G(t − o*(x)) · [1 − ζ(1 − exp(−max(t − o*, 0)/τ_rec))]

**Where it acts.** It multiplies the overwinter survival of adults (S̃_a), of juvenile stayers
(S_j) and of juvenile arrivers (after Q_k). It does not act on fecundity.

**Terms.**
- m_max = `mortality_ceiling` = 0.8.
- t ≥ 91 means 1993 on (`disease_timestep`).
- o(x) is the mapped arrival (`disease_arrival_map`). o*(x) = o + δ_0 + B·β_lag, and
  o_dec = (o + 1902 − 1993)/10.
- B(x) is a 24-function land-centred spatial basis, shared by severity and lag
  (`severity_/lag_space_frequencies` = 4). β_sev and β_lag are its coefficient vectors.
- G is a 3-year smoothstep onset gate (`onset_gate.form = compact_start`). `disease_tau` = 1.0 is
  a constant and is not used by this gate.
- **The Hill term acts on CURRENT post-dispersal density N′**, not on capacity.

**Structure.** The design is a STRUCTURED hypothesis: 56 free parameters in the candidate, each
stating a testable claim. It replaced a generic 967-coefficient spatiotemporal cosine field subtracted from
K's pre-softplus argument. That field annihilated eastern K and absorbed every kind of spatial
misfit. Frequencies are per-AXIS over the grid extent: space = 4 gives about 900 km N–S and
1500 km E–W half-wavelengths. This is deliberately coarse regional structure (east vs west vs
plains), not habitat-scale detail.

**The `disease_severity_map` deterministic** evaluates m_max · expit(·) · hill(K⁰ in 2025). It
is a diagnostic only. It has no gate or recovery term and uses K⁰ rather than N′, so it is not
the mortality that runs.

## Disease severity

`population_model.disease_prior._severity_comment` · candidate meaning: the Hill (density) term of the added mortality

**DENSITY-DEPENDENT mortality.** hill(N′) = N′^{n_h} / (N′^{n_h} + k_h^{n_h}) is evaluated on the
current post-dispersal density. The reasoning: epidemics need hosts. Mass-action transmission
makes R0 scale with density, and below a threshold density the pathogen cannot persist.

**Priors (candidate).**
- **k_h** (half-density). log k_hᶜ ~ N(log 8, .7) in **route counts** (`k_half_median_route_counts`
  8, `k_half_log_sd` .7), and k_h = k_hᶜ/s in internal units. LA1: k_h = .0136 internal (.272 RC).
- **n_h** (steepness). n_h = 0.5 + 2.05·expit(n_raw), with n_raw ~ N(−.04879, .8).
  - n_h is therefore bounded to (0.5, 2.55). The upper bound .85(4/m_max − 2) = 2.55 follows the
    mortality ceiling.
  - The location −.04879 puts the prior median at `hill_n_median` 1.5.
  - The scale .8 is `hill_n_log_sd` (.4) used as a logit-scale sd of 0.8, not as a log sd.
  - n_h ≈ 3 cannot occur.
  - LA1: n_h = 2.18.

The 27 km-era severity percentiles and the K_base reasoning of earlier versions are under Not
active ("Disease on carrying capacity").

## Disease severity modifier

`population_model.disease_prior._mu_comment`

μ_sev is the logit of the MODIFIER on the mortality amplitude:
Δ = m_max · expit(μ_sev + …) · gate · (1 − recovery).
- **Prior.** μ_sev ~ N(1.386, .5). mu_loc = logit(0.8) = 1.386 puts the prior median modifier at
  0.8.
- **Implied mortality.** With the other logit terms at 0, the prior-median PEAK added mortality is
  .8 × .8 = 64% of survivors at saturating density, before the gate and recovery terms.
- **Plain sigmoid.** The modifier is a PLAIN sigmoid. An earlier min(1, 2·expit(·)) form put a
  zero-gradient clip boundary exactly at the centre of μ's prior.
- **LA1 values.** μ_sev = −1.050. The maximum of Δ over (x, t) is .431.

## Disease late era

`population_model.disease_prior._late_comment`

b_late is a coefficient, per DECADE, on decades since the EPIDEMIC'S OWN historical start
(`disease_start_year` = 1993). It is NOT measured from any individual cell's local arrival, and
it is NOT a west/east genetics claim (that is native resistance below). It enters the logit of
the mortality amplitude Δ.

**Interpretation.** Mycoplasmal conjunctivitis in House Finches is documented to have hit
hardest in its first years and grown milder since, as pathogen and host co-evolved. A negative
b_late means cells reached later in that history were hit less hard. The reason is calendar time,
and it applies equally to any cell reached in a given decade regardless of geography.

**Prior.** b_late ~ N(−.15, .15). It was tightened from scale .3 because it is strongly
COLLINEAR with the density-dependent term: early-arriving cells are also dense, and late-arriving
ones also sparse. It is kept because epidemic-history timing and density are distinct
mechanisms. LA1: b_late = −.401.

## Disease native resistance

`population_model.disease_prior._native_comment`

This is NATIVE-LINEAGE suppression of the mortality amplitude. It uses the SAME dimensionless
core/margin shape I_0 that seeds the native range: 1 in the native core,
`margin_fraction_of_core` (.5) at the fringe, coarsening fractions (.125–.875) in 124 cells on the
54 km grid, and 0 elsewhere.
- **A distinct claim.** It concerns population origin (a genetically diverse ancestral range vs
  the east's single-founder introduction), not arrival timing.
- **Sign-constrained.** b_nat = −softplus(raw), raw ~ N(0, .7), so b_nat ≤ 0: native lineage can
  only suppress mortality. The prior median suppression is softplus(0) = .69 logit units in the
  core.
- **Collinear by construction** with late arrival and with density (the west is native,
  late-arriving and sparse). It is kept because it is a mechanistically distinct, falsifiable
  claim.
- LA1: b_nat = −.570.

## Disease onset and recovery

`population_model.disease_prior._field_comment`, `._recovery_comment` (candidate: `onset_gate.form = compact_start`, 3 years; `recovery.form = exponential`)

**Onset.** The gate G is a smoothstep over 3 years from o*(x) = o(x) + δ_0 + B(x)·β_lag.
- δ_0 ~ N(0, 2) years is the continental slack.
- β_lag ~ N(0, (1.5/√24)²) years per basis function is the regional slack. The arrival map is a
  hand-built smoothed reconstruction, so its timing is coarse both overall and regionally.
- The severity field β_sev ~ N(0, (.5/√24)²) is on the logit scale.

**Recovery.** After local onset the amplitude decays toward (1 − ζ) of its peak with e-folding
time τ_rec.
- ζ = expit(ζ_raw), ζ_raw ~ N(−.5, 1), so the prior median recovered share is about 38%.
- log τ_rec ~ N(log 12, .5).
- LA1: ζ = .0146, τ_rec = 19.2 y.
- There is no delayed-recovery term (`recovery.delay_*` are inert).

## Initial population

`population_model._initpop_comment` · candidate adds `spinup` = {enabled: true, years: 40, habitat_years: 5}

**Seed.** The native-range seed is N^seed = I_0(x) · K⁰(x, 1902) ·
`core_fraction_of_local_capacity` (1.0).
- It uses the **pre-floor** capacity K⁰.
- I_0 is the dimensionless core/margin map: 1 in the core (350 cells), `margin_fraction_of_core`
  = 0.5 at the fringe (935 cells), coarsening fractions .125–.875 in 124 cells (54 km grid), and 0
  elsewhere.
- In the candidate input (`_nativeMX`) the seeded area includes a Mexican hull. This is the union
  of the convex hull of ≤ 1970 western BBS presences and the convex hull of pre-1940 Mexican
  specimens. 299 margin cells were added at .5.

**Spin-up (what the 1902 state actually is).**
1. The forward model runs 40 years from the seed, with all rates frozen at their 1902–06 means,
   no disease and no release.
2. The spun-up TOTAL N^sp becomes the 1902 state, re-split 50/50 into adults and juveniles. The
   spin-up's own age split is discarded; the consequence of that is not assessed [U].
3. The BBS expected count for 1902 uses N^sp as the previous year's state.

**Why a fraction of local capacity.** A seed stated as an absolute count is a multiple of
capacity at initialisation, the density brake crashes it, and step 0 scores badly. A fraction of
local K⁰ is right at both ends. 1.0 because the native range had been at equilibrium for
millennia.

**margin_fraction_of_core = 0.5 is ASSUMED**, not measured.
- Taking it from observed abundance would double-count the habitat gradient K⁰ already expresses.
- A low value would leave the Plains-facing margin near empty, which substitutes for a barrier at
  no cost to the fit.

**Departure from the native range.** It is governed by Veit departure × (1 − ν w_nat); see
Departure. At LA1, ν = .989, so the native core is nearly sedentary. The earlier "net exporter"
argument, which relied on the logistic departure target .8, is under Not active.

**Evidence that holds the Great Plains barrier.** It is now carried by:
- the range-absence term (t ≤ 1965, halo 400 km);
- the 1902–39 pseudo-zero BBS rows (*pending decision*; see Observation model);
- early CBC counts (1902–65);
- museum-specimen counts (1880–1940);
- and the habitat.

Two consequences remain. The initial condition depends on fitted parameters, so it moves during
optimisation. And pre-1966 states are model output, not measurements of the historical range.

## Observation model

`population_model.observation` = {form: breeding, observer_effect, conc_density_power} (candidate); route effects integrated by the fitting driver (`laplace_fit.py`)

**Mean of BBS row i** (cell x_i, year t_i, observer×route r(i), first-year flag FY_i):

    μ_i = max(s·[N_{t_i−1}(x_i) + ρ_fl J_{t_i}(x_i)], 1e-6) · exp(η FY_i + σ e_{r(i)})

- **Timing.** N_{t−1} is the previous year's end-of-year total (after breeding). It is labelled
  "breeders of year t" by convention [L], so it comes before that year's dispersal and overwinter
  survival.
- ρ_fl ~ Beta(1, 4) is the fledgling detectability weight. At LA1 it is 3.7e-4, so effectively
  zero.
- η ~ N(0, 1) is the observer's first-year effect.
- **Count likelihood.** Counts are NB2 with concentration φ_i = φ_0 · q_M^{tier_i} ·
  max(μ_i, 1e-3)^κ, where:
  - φ_0 ~ Exponential(1);
  - q_M ~ Beta(2, 2) is the multiplier for Mexico (tier 1, 236 rows);
  - κ ~ HalfNormal(.5).
- **Route effects.**
  - e_r ~ N(0, 1) iid over 19,127 observer×routes. They are integrated out per route by a Laplace
    approximation (30 clipped Newton steps from 0).
  - σ is FREE, with σ ~ HalfNormal(1) (`observer_effect.sigma_prior_scale` = 1). LA1: σ = 1.196.
  - `sigma_u` (.97) and `sigma_mode = fixed` do not enter the objective. The model's own
    ZeroSumNormal `observer_route_eps` site is held at 0 by the driver.
  - Without a zero-sum constraint, the common shift of the route effects is held only by their
    N(0, 1) prior.
- **No holdout.** All 191,895 BBS rows (123,191 surveyed + 68,704 pseudo-zeros) have weight 1. The held-out masks of the selection stage
  are not used in final inference.

**Pseudo-zero rows (decision pending).** 68,704 rows, 1902–39, have no route id. They enter the
NB2 term with route effect 0, ON TOP of the range-absence term. Tests found them inert (about 1–2
log-likelihood units). The open choice, *pending with the user*:
- keep them as they run now; or
- drop them (`laplace_fit.py --drop_pseudozeros`), leaving the range absences as the only
  pre-1966 BBS-grid absence evidence.

## NestWatch likelihood

`population_model.nest_likelihood` (candidate: enabled; 1,089 attempts in 330 cells; data span 1968–2020 within the 1902–2020 filter window)

The realised births per breeder F^eff at the nest cells give the expected fledglings per attempt:
μ^N_n = 2F^eff/n_br · exp(X_n β^N_m), with n_br = 2.5 broods (pinned).

**Covariates and survival.** X_n is a 0/1 design over {building, nestbox, other_artificial,
substrate_unknown, card}. Daily nest survival is S^d_n = expit(α_dsr + X_n β^N_s), over a 32-day
cycle.

**Likelihood terms.**
- Nest fate enters through exposure days and failure, for attempts with an exposure record.
- Fledged counts enter as NB2 on (fledged − 1) with size φ_n, for successful attempts with at
  least one fledgling.

**Priors.**
- β^N_m ~ N(0, .1);
- β^N_s ~ N(0, .5);
- α_dsr ~ N(3.5, 1);
- log φ_n ~ N(2.3, 1).

## Terms added by the fitting driver

The candidate objective (`research/exp/invasion_sandbox/laplace_fit.py`, main repo) adds these
data terms and nuisance priors to the model density. Every term has weight 1, and the specimen
rows carry a 0/1 collector-filter weight (571 of 5,522 rows weighted 0). Equations are in
`docs/model_spec/METHODS_DRAFT.md` §7 and `MODEL_SPEC.md` §5.

**CBC counts, 1902–65** (`--cbc_counts`, latent effort, BBL migrant source, linear drift).
- Data: 10,511 circle-years in 1,823 circles.
- The migrant shares, distances and bearing are hard-coded from an offline BBL fit
  (`p14_bbl_sw.py`). Only the eastern bearing concentration is computed from the pairs file at run
  time.
- Circle effects are Laplace-integrated.
- Priors: log scale ~ N(0, 5); effort power ~ N(.6, .3); log NB size ~ N(0, 1); decadal drift
  ~ N(0, 1).

**BBL adult dead-recovery m-array** (`--bbl`, with disease).
- Data: 8,778 cohorts and 543 recoveries.
- Reporting probability: 3 regions × 4 eras, with logit ~ N(logit .005, 1.5).

**Museum specimens as counts** (`--specimens --specimen_counts`).
- Data: 5,522 cell-decades, 1880–1940, NB2 likelihood. The decade window is the mean over the
  decade's model years, and the 1880–1901 row reads the 1902 state.
- Priors:
  - era collecting rates log s^S_era ~ N(0, 3);
  - effort exponent β_E ~ **Beta(2, 2)**, sampled on the logit scale;
  - log NB size ~ N(0, 2).
- **β_E target.** The MAP objective carries the Beta(2, 2) density without the Jacobian (by
  convention MAP omits Jacobians). HMC mode adds the logit Jacobian for this block, so the HMC
  target is Beta(2, 2) on β_E. This was fixed by the lead on 2026-11-08. At the LA1 measurement
  the HMC target lacked the Jacobian and was uniform on β_E.

**Range absences** (`--range_absence`, halo 400 km + native Mexico).
- Data: 74,306 cell-years with t ≤ 1965.
- Cost: −log NB2(0 | sN, k_P = .43724).

**Native-viability penalty** (`--native_viable`, scale .05). This is a one-sided penalty on
mean log potential growth λ^pot = S_a + S_j F in 1902, applied in the Mexican and US native
regions.

## Radial splits

`dispersal._radial_splits_comment`

`juvenile_radial_splits_km` accepts either an explicit list of km boundaries (must start at 0.0
and end ≥ 1e8) or the string "derive". "derive" computes equal-mass radial bands for whatever
`juvenile_mdd_km` is set: quantiles, by default terciles, overridable via
`juvenile_radial_split_quantiles`.

**The pinned list is not a set of quantiles.** The list [0, 155.36, 482.74, 1e9] is log-spaced
band edges from 50 to 1500 km, produced by a `get_log_spaced_splits(50,1500,3)` helper that no
longer exists. It does NOT move with `juvenile_mdd_km`:
- for the older area-density kernel, at mdd 150 the three cohorts hold 67/28/4% of the dispersal
  mass, and at 400 they hold 32/41/27%;
- for the candidate's J200 V&L-faithful kernel they hold about 63/26/11% (see Dispersal kernels).

It is retained for artifact compatibility. Ingest compares the resolved dispersal dict for exact
equality, and the candidate's inputs (all kernel variants) were built under these literals.

---

# Not active in the current model

The sections below describe mechanisms or parameterizations that the current candidate does
**not** run. Some are base-config defaults; others are earlier designs. They are kept so that
older runs and records can be read. Do not take any value here as a current setting.

## Rank-2 manifold prior (base default)

`manifold_prior.form = rank2` — base default; the candidate uses `capacity_demography`.

**Fields.** Correlated weights for FOUR demographic manifolds, ordered [survival_adult,
reproduction, capacity, survival_juv]. Each field has its own coefficient vector:
- H_s drives Sa;
- H_sj drives Sj and Q;
- H_r drives Fmax;
- H_k drives K.

**Parameterization (rank-2 angular; code names, not model symbols).**
- `r_j = sigmoid(r_raw_j)` is a communality in (0, 1), and `th_j` is an angle.
- `beta_j = w_scale_j * (r_j*cos(th_j)*f1 + r_j*sin(th_j)*f2 + sqrt(1 - r_j^2)*eps_j)`, with `f1`,
  `f2` and `eps` iid ACROSS FEATURES.
- Then `Var(beta_j) = w_scale_j^2` and `Corr(j, k) = r_j*r_k*cos(th_j - th_k)`.

**Why rank 2.** Rank 1 forces the tetrad identity, which caps ρ(Sa, Sj) at .576 when ρ(F, K) =
.85 and the cross-group correlation is .70.

**Calibration.** At communality_logit_scale .35 and angle_scale .15:
- Sa–Sj and F–K median .83, 5–95% [.75, .88];
- cross-group median .69, 5–95% [.49, .83].

LKJ is unusable here: concentration > 1 concentrates near the identity. A full covariance over
features would break the iid-across-features GP contract.

## Exchangeable form (sensitivity variant)

`manifold_prior.form = exchangeable` — used by the "four free fields" sensitivity variant, not by the candidate.

**Parameterization (code names).** `beta_j = w_scale_j*(sqrt(rho)*f + sqrt(1-rho)*eps_j)`, so the
correlation is `rho` for every pair, with `rho = sigmoid(Normal(logit(coupling_target),
coupling_logit_scale))`.

**Why it was introduced.** The run_18 HMC descent bought +7,400 nats by decoupling single fields
(reproduction communality .76 → .27). Meanwhile adult survival and capacity stayed coupled, and K
became an absence switch. With one ρ, decoupling one field decouples all of them.

**Sensitivity variant.** According to the records (SANDBOX_INVASION_SHAPE S:9778-9781,
"FOUR FREE FIELDS"), the four-free variant adds `coupling_priors` on all six pairs, with
`fixed_coupling` = null and `coupling_logit_scale` 1.0. The variant's overlay was not diffed [L]
(`docs/model_spec/SENSITIVITY_SET.md`).

## K trend

`population_model._k_trend_comment` — `k_trend.enabled = false` in the candidate (w_k_trend is a deterministic 0).

**Design.** The trend was a continental (spatially uniform) smooth time trend on K's pre-softplus
argument, time-centred so α_K keeps the level. It was meant as a SAFETY VALVE AND FAILURE
DIAGNOSTIC, not a modelled mechanism. It existed because the model then had no temporal degree of
freedom for capacity except the disease-on-K term.

**Prior.** budget = .05 put the implied K multiplier within 2.3% of 1.0 at the prior median and
7% at the 95th percentile, with 3 cosines over 124 years.

**Reading it.** A fitted trend that moved K by more than about 10–15% was to be read as failure,
not loosened.

## Invasion sites

`population_model._invasion_sites_comment` — base default `invasion_sites = 9`; the candidate's inputs carry 4 candidate cells and use only the first.

**Design.** The historical record documents one known sighting (Jones Beach, Long Island), not
that it was the only release. So `model_inputs.py` resolved the 9 NEAREST LAND CELLS to the
sighting coordinate and gave each its own per-year pulse: 9 × invasion_window_years (10) = 90
parameters. The fit could then allocate mass across sites and years.

**Release date.** The release was dated 1940 (`invasion_year`; see `docs/TEMPORAL.md`).

**Allee exemption.** A related key, `release_allee_exemption_years` (base 10), switched the Allee
factor off in the candidate release cells for that many years. The reason: cage birds released
together were aggregated far below grid scale. The candidate sets it to 0.

## Invasion budget

`population_model._invasion_budget_comment` — not active (the candidate's founder total is pinned).

**Design.** The 90 free (site, year) coefficients shared one prior BUDGET divided by
sqrt(n_coefficients). log_budget 3.0 gave a per-coefficient log sd of 3.0/sqrt(90) = 0.316.

**Median pulse.** median_fraction_of_global_q50 = 0.1 put the prior-median total pulse across the
9 sites at about .64 route counts per year (9 × .1 × .7083, the base q50).

## Disease on carrying capacity (base default)

`disease_prior.target = capacity` (base default; `severity_ceiling`, `_severity_comment`, `_ceiling_comment`, `_mu_comment` in the base config describe this form)

**Mechanism.** The epizootic reduced carrying capacity:
K = K_base · (1 − severity(x) · onset_gate(x,t) · (1 − recovery(t − arrival))).

**Severity.** severity(x) = severity_ceiling · expit(μ + …) · K_base^n/(K_base^n + K_half^n), a
Hill function of the potential density K_base, with:
- severity_ceiling .5;
- prior median severity .8 × .5 = 40% in the densest populations.

**Why K_base.** Severity keyed off K_base because the demographic fields are precomputed before
the forward simulation, "so N does not exist yet".

**Why the ceiling mattered.** The ceiling was load-bearing for identifiability. K had to stay
monotone in K_base: at .5 this held for every n tested, and at .7 it broke.

**Severity percentiles.** With K_half = 8 counts and n = 1.5 on the 27 km occupied-cell
distribution, severity ran 1% (q25) / 6% (q50) / 22% (q75) / 37% (q90) / 48% (q99).

**None of this runs in the candidate.** There:
- the Hill term acts on current density N′ inside the forward step;
- the ceiling is `mortality_ceiling` .8;
- n_h is bounded to (.5, 2.55).

## Logistic departure and the "net exporter" argument

`population_model.dispersal_target_capacity_fraction` (.8) and `dispersal_logit_*` — not active (Veit departure overrides them).

**Earlier mechanism.** Departure rose logistically with N/K. With the seed at K and a target of
.8, the emigration logit started at +.2 × slope, so the native range was a NET EXPORTER from year
one.

**The barrier argument.** This was deliberate. The historical fact to reproduce was that an
exporting population still did not cross the Great Plains for eighty years. The barrier therefore
had to be paid for by its own evidence (then the pre-invasion pseudo-zeros, buffer 700 km).

**What runs instead.** The candidate replaces this with Veit departure × (1 − ν w_nat); see
Departure.

## Unclumped Allee form (aggregation_k = None)

`allee_prior.aggregation_k` absent — base default; the candidate uses k_A = 3 per-capita.

**Form.** 𝒜 = 1 − exp(−γN) = 1 − 2^(−C/n50), which is exactly ½ at C = n50.

**Retention at the old prior.** At the old prior location n50_raw = −2.0 (median n50 = .127
counts), fecundity was retained:
- 42% at .1 counts;
- 75% at .25;
- 94% at .5;
- 99.6% at 1.0.

**Fold.** The Allee fold sat near 1/γ = n50/ln2.

**Prior.** The base config's prior was n50_raw ~ N(−2.0, 1.0). The candidate overlay sets
N(−3.4915, .25), which is inert while n50 is pinned.

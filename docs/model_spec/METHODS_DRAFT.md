# Methods (draft)

**Status.** This is a technical base for the manuscript Methods. It describes the HMC candidate as
executed. Every model statement is transcribed from `MODEL_SPEC.md` (LA1 evaluation), corrected
where the independent verification (`VERIFICATION.md`) found the executed code differs. Symbols
are those of `NOTATION.md` with the renames listed in §13; values come from `RESOLVED_CONFIG.json`.

- **Tags.** [L] marks a statement inferred from code rather than measured; [U] marks one left open.
- **Italic values** are LA1 point values, not priors.
- **Inputs and runs (2026-11-08).**
  - The current habitat-path input is the `_vlF` build: the V&L-faithful J200 kernel with the
    FIELD54 land-share floor 1e-4 restored (overlay `J200VWfl_ALL_local.json`, TACC `ALL_J200VWfl`).
  - The `_vlF` all-data refits have landed: L0F1 (J200) 201,573.1, L0F2 (J330) 201,637.5, L0F3
    (J110) 201,710.5. HMC slot A is initialised at L0F1.
  - The superseded `_vl` build (LA1; overlay `J200VW_ALL_local.json`) differs only at about 170
    coastal cell-band pairs (maximum absolute difference .31). At fixed parameters the change moves
    the objective by < 0.001 nats, so it is inert. The LA1 point values quoted below come from `_vl`.
- **Pending decisions.** Three questions are pending with the user (§12):
  1. whether the 1902–39 pseudo-zero BBS rows are dropped;
  2. whether to move from FIELD54 v1 states to `field54prod_v2_ema`;
  3. an open convention question on the census used for crowding and the Allee factor (§4).

---

## 1. Domain, time and units

**Grid and time.** The model runs on an equal-area grid (ESRI:102003) of 67 × 112 cells, each
54 × 54 km (2,916 km²). Of these, 4,318 are land cells, indexed by x. Years are indexed
t = 0, …, 123 for 1902–2025, with an annual step.

**Units.** The state variables are densities in internal units. The gauge s = 20 converts them to
the route-count scale C = sN: the expected Breeding Bird Survey (BBS) count of the median,
experienced observer×route (route effect 0, first-year flag 0). Absolute-scale priors are
declared in route counts (RC) and divided by s. The exception is the capacity floor K_min, which is
declared in internal units. No conversion to birds per km² enters the likelihood.

## 2. Habitat fields from DESK latent features

**DESK features.** Habitat enters only through Z_t(x) ∈ ℝ⁶⁴, the latent features of the DESK
community model, trained on the co-occurrence of other bird species (House Finch excluded).
The production cube (FIELD54, v1 states) is built in three steps:
1. a native 54 km DESK fit with exponentially smoothed output (EMA, half-life 10.9 yr);
2. a field correction fitted to all 1966–2025 summer BBS rows;
3. selection of the summer block (64 dimensions).

`DESK_PROVENANCE.md` gives the provenance. Z and Z^d are stored in float32.

**Kernel interpretation.** Z is an uncentred Nyström feature map of the Ružička kernel, so that
Z(x)·Z(x′) approximates community similarity. That kernel describes the feature map, not the
executed predictors.
- Under the coefficient prior of (2), and before the coupling factors of §8, the demographic
  predictors H_ℓ = w_ℓ(Z_t − z_c)·**f** (ℓ = a, f, j) are Gaussian processes with covariance
  w_ℓ²(Z_t(x) − z_c)·(Z_{t′}(x′) − z_c), i.e. centred at z_c.
- H_K mixes its level direction **β**_K with the change direction w_K**f** (3), so its covariance
  has r_K cross-terms.
- The coupling factors (29) make the effective coefficient prior non-Gaussian.

**Path features.** For each juvenile dispersal kernel k = 1, …, 12, Z^d_{t,k}(x) ∈ ℝ⁶⁴ is a
land-conditioned path summary of Z: an average over fractional displacements of kernel k with
water excluded and the land share renormalised, using 20 integration steps.

**Reference quantities.**
- Z̄(x) and Z̄^d_k(x) are the 1996–2025 means of Z_t and Z^d_{t,k}.
- The habitat centre z_c is the mean of Z_t(x) over the distinct cell-years with a positive BBS
  count.

**Coefficients.** There are four fields, ℓ ∈ {a, f, K, j}: adult survival, fecundity, capacity and
juvenile survival. With amplitudes w_ℓ and a capacity–demography correlation r_K,

$$
w_\ell=\operatorname{softplus}(\xi_\ell),\qquad r_K=\operatorname{expit}(\xi_r), \tag{1}
$$

$$
\boldsymbol\beta_a=w_a\mathbf f,\quad \boldsymbol\beta_f=w_f\mathbf f,\quad \boldsymbol\beta_j=w_j\mathbf f,\quad
\boldsymbol\beta_K=w_K\big(r_K\mathbf f+\sqrt{1-r_K^2}\,\mathbf g\big),\qquad \mathbf f,\mathbf g\sim\mathcal N(0,I_{64}). \tag{2}
$$

Adult survival, fecundity and juvenile survival therefore share one coefficient direction **f**.
Capacity mixes **f** with its own direction **g**.

**Split readout.** Each field's predictor separates a cell's long-run level (centred at z_c) from
its within-cell change (centred at the cell's own Z̄(x)). The change always reads the demographic
direction:

$$
H_\ell(x,t)=\big(\bar Z(x)-z_c\big)\cdot\boldsymbol\beta_\ell+\big(Z_t(x)-\bar Z(x)\big)\cdot w_\ell\mathbf f . \tag{3}
$$

For ℓ ∈ {a, f, j} this reduces to H_ℓ = w_ℓ h(x, t), with the shared demographic habitat index
h(x, t) = (Z_t(x) − z_c)·**f**. Space-for-time substitution therefore holds for S_a, F and S_j. It
does not hold for K: its spatial level uses **β**_K, while its temporal response uses **f**. The
journey index on path features is

$$
h^d_k(x,t)=\big(Z^d_{t,k}(x)-z_c\big)\cdot\mathbf f . \tag{4}
$$

## 3. Vital rates, capacity and density-regulation constants

**Rates.** With intercepts α_a, α_f, α_K and α_j (defined at z_c):

$$
S_a=\operatorname{expit}(\alpha_a+H_a),\qquad F=\operatorname{softplus}(\alpha_f+H_f),\qquad S_j=S_a\,\operatorname{expit}(\alpha_j+H_j), \tag{5}
$$

$$
K^0=\operatorname{softplus}(\alpha_K+H_K),\qquad K=K^0+K_{\min},\qquad K_{\min}=0.0231039\ \text{internal}, \tag{6}
$$

$$
Q_k(x,t)=\operatorname{expit}\!\big(\alpha_a+w_a h^d_k\big)\,\operatorname{expit}\!\big(\alpha_j+w_j h^d_k\big). \tag{7}
$$

- S_a is disease-free adult annual survival.
- F is the maximum fecundity: fledged juveniles per breeder at low density, before the Allee factor.
- S_j is the survival of juveniles that do not disperse.
- K⁰ is the pre-floor capacity and K the capacity.
- Q_k is the journey survival of juveniles dispersing via kernel k, evaluated at the arrival
  (destination) cell.

There is no time trend on K and no disease term on K. K_min is 0.462 RC, which is 20 ñ₅₀ in common
units (§5).

**Density-regulation constants.** These are computed from the disease-free rates. With
S_a^min = clip(1 − S_j F, 0.05, 1):

$$
c_s=\theta\max\!\Big(\frac{S_a}{S_a^{\min}}-1,0\Big),\qquad S_a^{K}=\frac{S_a}{1+c_s},\qquad
c=\max\!\Big(\frac{F S_j}{1-S_a^{K}+10^{-6}}-1,0\Big). \tag{8}
$$

- θ is the share of the density brake carried by adult survival.
- c_s is the adult-survival crowding constant, and c is the fecundity crowding constant.
- Where F S_j > 1 − S_a^K, the constants make S_a^K + S_j F/(1 + c) = 1 at N = K, ignoring the
  Allee and disease terms. In sink cells c is clipped to 0 and the sum stays below 1.

*At LA1, θ = 1.19 × 10⁻⁴, so the brake acts almost entirely on fecundity.*

## 4. Annual cycle

**State.** The state is A_t(x) adults and J_t(x) juveniles at the end of year t, after breeding,
with N_t = A_t + J_t. Year t starts from (A⁰, J⁰) = (A_{t−1}, J_{t−1}). The steps follow.

**Release.** At t_rel = 49 (1951), in the single cell x_rel = (27, 89), founders are added:
A⁰ += R/2 and J⁰ += R/2, with

$$
R=\frac{q_{50}}{s}\exp(\log R_{\rm tot})=\frac{0.6852}{20}=0.03426\ \text{internal}. \tag{9}
$$

- q₅₀ is the median modern per-cell route count over surveyed cells.
- log R_tot is pinned at 0.
- There is no Allee exemption for the release cell.

**Departure** (Veit & Lewis 1996; both ages). Let N⁰ = A⁰ + J⁰ and u⁰ = N⁰/max(K, 10⁻⁶). Then

$$
p=\operatorname{clip}\!\Big(u^0-\tfrac1{10}\operatorname{softplus}\big(10(u^0-1)\big),\,10^{-6},\,1\Big)\,\big(1-\nu\,w_{\rm nat}\big), \tag{10}
$$

where w_nat = I₀/max I₀ and ν reduces departure in the native range. *ν = 0.989.*

I₀(x) ∈ [0, 1] is the native seed shape:
- 1 in the native core (350 cells);
- 0.5 at the margin (935 cells);
- fractions from 0.125 to 0.875 in 124 cells, where coarsening to 54 km kept partial margins;
- 0 elsewhere.

**Movement.**

$$
A'=A^0(1-p)+\mathbb F\!\Big[\mathcal K_A\circledast\frac{A^0p}{\varepsilon_A+10^{-6}}\Big],\quad
J^s=J^0(1-p),\quad
J^m=\sum_{k=1}^{12}Q_k\,\mathbb F\!\Big[\mathcal K_k\circledast\frac{J^0p}{\varepsilon_k+10^{-6}}\Big]. \tag{11}
$$

- ⊛ is linear FFT convolution with zero padding, and ε_A, ε_k are the land-fraction edge
  corrections.
- 𝔽 is a smooth round-off floor applied elementwise. For a dummy argument ϖ,
  𝔽(ϖ) = ϖ₊³/max(ϖ₊² + τ_FFT², 10⁻¹⁵⁰), with τ_FFT = 64 ε_mach max|ϖ| (no gradient), where ε_mach
  is the machine epsilon of the working precision.
- **Adult kernel** 𝒦_A: isotropic, 2-D area density exp(−(d/b_A)^{0.859}) at distance d, with a
  configured mean dispersal distance of 100 km (b_A is the implied scale). Whether its
  1-D → 2-D translation matches the juvenile treatment is [U].
- **Juvenile kernels** 𝒦_k: 4 directions × 3 radial bands ([0, 155.4), [155.4, 482.7) and
  [482.7, ∞) km).
  - Each kernel restricts the V&L-faithful 2-D density ∝ exp(−(d/b_VL)^{a_VL})/d, with
    a_VL = 0.468 and b_VL = 24.937 km (1-D mean 200 km), using raised-cosine wedges and logistic
    band edges.
  - Kernels are cell-integrated.
- Adult dispersers have no journey mortality. Juvenile arrivers survive the journey with Q_k in
  place of S_j.

**Epizootic mortality** (mycoplasmal conjunctivitis). Let N′ = A′ + J^s + J^m. The added
mortality fraction is

$$
m=\Delta(x,t)\,\frac{(N'+10^{-12})^{n_h}}{(N'+10^{-12})^{n_h}+k_h^{n_h}}, \tag{12}
$$

$$
\Delta(x,t)=\mathbb 1[t\ge 91]\,m_{\max}\,\operatorname{expit}\!\big(\mu_{\rm sev}+b_{\rm late}\,o_{\rm dec}(x)+b_{\rm nat}I_0(x)+B(x)\cdot\boldsymbol\beta_{\rm sev}\big)\,G\big(t-o^*(x)\big)\,\Big[1-\zeta\Big(1-\exp\!\big(-\max(t-o^*,0)/\tau_{\rm rec}\big)\Big)\Big]. \tag{13}
$$

- m_max = 0.8. The indicator is zero before 1993 (t < 91).
- o(x) is the exogenous arrival map (timestep units). o*(x) = o(x) + δ₀ + B(x)·**β**_lag adds a
  continental slack δ₀ and a regional slack.
- o_dec = (o + 1902 − 1993)/10 is decades since the epidemic's start.
- B(x) is a 24-function land-centred spatial basis, with coefficient vectors **β**_sev (severity)
  and **β**_lag (lag).
- G is a 3-year smoothstep onset gate: G(t − o*) = smoothstep(clip((t − o*)/3, 0, 1)), where
  smoothstep(ϖ) = ϖ³(10 − 15ϖ + 6ϖ²).
- ζ is the eventual recovered share, and τ_rec the e-folding time of recovery.
- k_h (internal units) and n_h are the Hill half-density and steepness.

*At LA1: k_h = 0.0136 (0.272 RC), n_h = 2.18, ζ = 0.0146, τ_rec = 19.2 yr, max Δ = 0.431.*

**Survival and reproduction.** Arriving juveniles are reduced by the added mortality before the
census used for crowding and mate-finding is formed. That census is N″ = A′ + J^s + (1 − m)J^m,
with crowding ratio u = N″/max(K, 10⁻¹²):

$$
F^{\rm eff}=\frac{F}{1+c\,u^2}\,\mathcal A(N''),\qquad \tilde S_a=\frac{S_a(1-m)}{1+c_s u^2}, \tag{14}
$$

$$
A_t=A'\tilde S_a+J^sS_j(1-m)+J^m(1-m),\qquad J_t=A_t\,F^{\rm eff}, \tag{15}
$$

followed by masking to land and truncation at 0. All winter survivors breed as yearlings.
F^eff(x, t) is the realised number of births per breeder.

- The Hill term (12) uses N′, which includes the arrivers before disease.
- N″ = N′ before 1993, since m = 0 then.

> **Open convention question (pending, user).** In the census N″, adults A′ and stayers J^s enter
> *before* the year's added mortality, while arrivers enter *after* it, (1 − m)J^m
> (`age_forward.py:751` vs `:1170`). The asymmetry may be unintended. It affects crowding and the
> Allee factor only from 1993 onward. The alternatives are:
> - census everyone before mortality (N′);
> - census everyone after it (A′ + J^s + J^m, each × (1 − m), or survivors).
>
> The model currently runs N″.

## 5. Mate-finding Allee effect

The mate-finding factor is the clumped, per-capita form. It is evaluated on the census N″ of §4,
before survival and after the added mortality on arrivers (N″ = N′ before 1993):

$$
\mathcal A(N)=1-\Big(1+\frac{\gamma N}{k_A}\Big)^{-(k_A+1)},\qquad k_A=3,\qquad \gamma=\frac{s\ln 2}{n_{50}+10^{-6}} . \tag{16}
$$

- n₅₀ = softplus(n50_raw) is stated in route counts and **pinned** at
  n50_raw = −3.756179, i.e. n₅₀ = 0.0231039 RC. This is 0.2 × the 2026-11-06 pin.
- In internal units, ñ₅₀ = n₅₀/s = 0.0011552, and γ = 600.0 per internal unit.
- Because γN = ln 2 · C/n₅₀, the factor depends on the route-count scale alone.
- 𝒜 = ½ at C½ = 0.819 n₅₀ (0.01892 RC), **not** at n₅₀. The unclumped form 1 − 2^(−C/n₅₀), which
  is ½ at n₅₀, is not used.

**Choice of n₅₀.** All-data J200 objectives at floor .0231 are:

| setting | objective | runs |
|---|---|---|
| ×0.5 | 201,596.8 | LC31r |
| ×0.33 | 201,558.5 | LA7; `_vlF` LFA7 201,558.5 |
| ×0.2 | 201,573.2 | LA1p; `_vlF` L0F1 201,573.1 |

The candidate stays at ×0.2, chosen by the user for HMC navigability. ×0.33 is the
likelihood-preferred sensitivity neighbour (§11).

## 6. Initial state and spin-up

**Seed.** The native-range seed is N^seed(x) = I₀(x) K⁰(x, 0). It uses pre-floor capacity, with
the core at the full local capacity.

**Native range in the inputs.** I₀ includes a native-Mexico hull. This is the union of the convex
hull of western BBS presences up to 1970 and that of pre-1940 Mexican specimens. It covers 1,409
seeded cells: 350 core, 935 at 0.5 and 124 at other fractions.

**Spin-up.**
1. From the seed, the forward model (§4) runs 40 years with all rates frozen at their 1902–06
   means, with Δ = 0 and no release.
2. The spun-up total N^sp starts 1902, split equally: A_{−1} = J_{−1} = N^sp/2.
3. The spin-up's own age split is discarded; the consequence of that is not assessed [U].

## 7. Observation models and likelihood

**NB2 distribution.** All count terms use, for a count y with mean μ and size φ,

$$
\log\mathrm{NB2}(y\mid\mu,\phi)=\ln\Gamma(y+\phi)-\ln\Gamma(\phi)-\ln\Gamma(y+1)+\phi\ln\frac{\phi}{\phi+\mu}+y\ln\frac{\mu}{\phi+\mu}. \tag{17}
$$

**7.1 BBS counts.** There are 191,895 rows i with counts y_i:
- 123,191 surveyed rows (US/Canada 1966–2025; Mexico 2008–18, 236 rows);
- 68,704 pseudo-zero rows for 1902–39.

Each row has cell x_i, year t_i, observer×route r(i) and an observer first-year flag FY_i. The
expected count is

$$
\mu_i=\max\!\big(s\,[N_{t_i-1}(x_i)+\rho_{\rm fl}J_{t_i}(x_i)],\,10^{-6}\big)\exp\!\big(\eta\,\mathrm{FY}_i+\sigma e_{r(i)}\big),
\qquad \phi_i=\phi_0\,q_M^{\,\mathrm{tier}_i}\max(\mu_i,10^{-3})^{\kappa}. \tag{18}
$$

- N_{t−1} is the previous year's end-of-year total, with N_{−1} ≡ N^sp. Reading it as the breeding
  population of year t is a convention [L].
- ρ_fl is the fledgling detectability weight. *ρ_fl = 3.7 × 10⁻⁴.*
- η is the first-year observer effect.
- e_r ~ 𝒩(0, 1) are standardised observer×route effects (19,127 routes), with standard deviation
  σ.
- φ₀ is the base concentration, q_M the multiplier for Mexican rows (tier 1, 236 rows), and κ the
  density power.
- The 68,704 pseudo-zero rows (no route) have no route term.

**Route effects integrated by an embedded Laplace approximation.** For route r, let
𝓛_r(e) = Σ_{i∈r} log NB2(y_i | μ_i(e), φ_i(μ_i(e))). Note that φ_i depends on μ_i through κ. Let
e*_r be its mode: 30 Newton steps from 0, each clipped to ±1 and unrolled so the gradient is exact.
Let 𝓘_r be the observed information with respect to log μ, taking the total derivative:

$$
-\log p(\mathbf y_r)\approx-\mathcal L_r(e^*_r)+\tfrac12 e^{*2}_r+\tfrac12\log\big(1+\sigma^2\mathcal I_r\big),\qquad
\mathcal I_r=\max\!\Big(-\sum_{i\in r}\frac{d^2\log\mathrm{NB2}\big(y_i\mid\mu_i,\phi_i(\mu_i)\big)}{d(\log\mu_i)^2}\Big|_{e=e^*_r},0\Big), \tag{19}
$$

dropping the constant −½ log 2π. The BBS term is

$$
L_{\rm BBS}=\sum_r\big[-\log p(\mathbf y_r)\big]_{\rm Laplace}-\sum_{i:\,r(i)=\varnothing}\log\mathrm{NB2}(y_i\mid\mu_i\,|_{e=0},\phi_i). \tag{20}
$$

*L_BBS = 165,500.34; σ = 1.196.*

**7.2 NestWatch.** There are 1,089 nest attempts n in 330 cells. The data span 1968–2020, within
the 1902–2020 filter window. X_n is a 0/1 design over five substrate/record covariates. The terms
are:

$$
\mu^N_n=\frac{2F^{\rm eff}(x_n,t_n)}{n_{\rm br}}\exp\!\big(X_n\boldsymbol\beta^N_m\big),\quad S^d_n=\operatorname{expit}\!\big(\alpha_{\rm dsr}+X_n\boldsymbol\beta^N_s\big),\quad
\mu^{(1)}_n=\tfrac1{10}\operatorname{softplus}\!\Big(10\big(\mu^N_n/(S^d_n)^{32}-1\big)\Big),
$$
$$
\mathcal L_n=w^{\rm exp}_n\big[E^{\rm pre}_n\log S^d_n+\mathbb 1^{\rm fail}_n\log\big(1-(S^d_n)^{d^{\rm last}_n}\big)\big]+w^{\rm cnt}_n\log\mathrm{NB2}\big(\text{fledged}_n-1\mid\max(\mu^{(1)}_n,10^{-12}),\phi_n\big). \tag{21}
$$

- n_br = 2.5 broods (pinned), and the nesting cycle is 32 days.
- S^d_n is the daily nest survival, and (S^d_n)^{32} the probability of surviving the cycle.
- E^pre_n are the exposure days before the last interval, and d^last_n is the length of the last
  interval in days.
- 𝟙^fail_n = 1 if the attempt failed.
- w^exp_n = 1 if the attempt has an exposure record, and w^cnt_n = 1 if its outcome is a success
  with at least one fledgling (`data_loading.py:404-405`).
- φ_n is the NB size of the fledged count.

**7.3 Christmas Bird Count (CBC) counts, 1902–65.** There are 10,511 circle-years ϱ in 1,823
circles ι(ϱ), with counts y_ϱ. t_ϱ is the model year whose end-of-year state the count reads, and
yr_ϱ its calendar year. The circle's source abundance mixes local birds with arriving winter
migrants:

$$
N^{C}_\varrho=\sum_x s\,N_{t_\varrho}(x)\,W_{\iota(\varrho),x},\qquad W_{\iota,x}=(1-m^{\rm mig}_x)W^{L}_{\iota,x}+m^{\rm mig}_x\sum_{x'}P_{x,x'}W^{L}_{\iota,x'}, \tag{22}
$$

$$
\log\mu^C_\varrho=s_C+\gamma_C\tfrac{\mathrm{yr}_\varrho-1940}{10}+p_C\psi_\varrho+\tfrac12p_C^2V^E_\varrho+\log\max(N^C_\varrho,10^{-12})+v_{\iota(\varrho)},\qquad
\phi^C_\varrho=\Big[(1+\phi_C^{-1})\exp\!\big(p_C^2V^E_\varrho\big)-1\Big]^{-1}. \tag{23}
$$

- W^L are local circle-to-cell weights.
- m^mig_x is the migrant share by origin: 0.57 east of 95°W, 0.06 on the Pacific coast (west of
  117°W) and 0.14 elsewhere in the west.
- P_{x,x′} is the migrant destination density: LogNormal(log 265 km, 0.83) in distance with a
  von Mises bearing (218°, κ_bear) for eastern origins, and isotropic LogNormal(log 337 km, 0.83)
  otherwise.
  - The shares, distances and bearing are hard-coded from an offline fit to Bird Banding Laboratory
    (BBL) summer–winter pairs (`p14_bbl_sw.py`).
  - Only κ_bear is computed from the pairs file at run time. *κ_bear = 6.24.*
- ψ_ϱ is the log effort. Where effort is missing, ψ_ϱ = ψ̄(t), a hat-basis trend in recorded
  effort, with V^E_ϱ = τ_E² = 0.836²; otherwise V^E_ϱ = 0.
- s_C, p_C, φ_C and γ_C are the log scale, effort power, NB size and decadal drift.
- The circle effects v_ι ~ 𝒩(0, 1) are Laplace-integrated as in (19), with 40 clipped Newton
  steps.

*L_CBC = 9,212.40.*

**7.4 BBL adult dead recoveries.** There are 8,778 banding cohorts b, with 543 recoveries in total.
Cohort b was banded in cell x_b, model year t^band_b, region reg_b, with n^B_b birds released;
y^B_{bt} birds were recovered dead in year t. Let 𝒯_b be the recovery years of cohort b's
m-array row. Whether the banding year itself is the first interval is [U].

Survival in the recovery model applies the same disease and crowding terms to the end-of-year
state:

$$
S^B(x,t)=\operatorname{clip}\!\Big(\frac{S_a\big(1-\Delta\,\mathrm{hill}(N_t)\big)}{1+c_s(N_t/K)^2},10^{-6},1-10^{-6}\Big), \tag{24}
$$

$$
P^B_{bt}=\Big[\prod_{t'\in\mathcal T_b,\,t'<t}S^B(x_b,t')\Big]\big(1-S^B(x_b,t)\big)\,\pi^B_{\mathrm{reg}_b,\,\mathrm{era}(t)},\qquad
\mathcal L^B=\sum_{b}\sum_{t\in\mathcal T_b}y^B_{bt}\log P^B_{bt}+\sum_b\Big(n^B_b-\sum_{t}y^B_{bt}\Big)\log\Big(1-\sum_{t}P^B_{bt}\Big). \tag{25}
$$

- hill(·) is the Hill factor of (12). Here it is evaluated on N_t rather than N′.
- π^B is the reporting probability, for 3 regions × 4 eras (<1980, 1980–94, 1995–2005, ≥2006).
  It is sampled on the logit scale.

*L_BBL = 4,666.49.*

**7.5 Museum specimens, 1880–1940.** There are 5,522 cell-decade rows ς, 582 of them positive, with
counts y_ς. Preserved Aves specimens give the collecting effort E_ς. The terms are:

$$
\bar N_\varsigma=s\sum_tT_{\varsigma t}N_t(x_\varsigma),\qquad
\mu^S_\varsigma=\max\!\big(s^S_{\mathrm{era}(\varsigma)}E_\varsigma^{\beta_E}\,\mathcal D(\bar N_\varsigma),10^{-12}\big),\qquad
\mathcal D(N)=1-(1+N/k_P)^{-k_P}, \tag{26}
$$

with y_ς ~ NB2(μ^S_ς, φ_S).
- T is the row's decade window: each row of T sums to 1, i.e. the mean over the decade's model
  years. The 1880–1901 row reads the 1902 state.
- s^S_era is the collecting rate for two eras (pre-1910 and 1910–40), and β_E is the effort
  exponent.
- Rows carry a 0/1 collector-filter weight: 571 of the 5,522 rows have weight 0. The filter rule
  is [U].

*L_spec = 2,415.28.*

**7.6 Range absences.** There are 74,306 cell-years (t, x) in the set 𝒳_abs, with t ≤ 1965. They are
defined from a 400 km halo with a native-Mexico hull [L: the builder was not re-read]. Each
contributes −log NB2(0 | sN, k_P):

$$
L_{\rm abs}=\sum_{(t,x)\in\mathcal X_{\rm abs}}k_P\log\!\big(1+sN_t(x)/k_P\big),\qquad k_P=0.43724. \tag{27}
$$

**7.7 Native viability.** With potential growth λ^pot(x) = S_a(x, 0) + S_j(x, 0) F(x, 0), regions
𝓡 ∈ {Mexico (88 cells), US native (257 cells)} and scale 0.05:

$$
L_{\rm nv}=\sum_{\mathcal R}\tfrac12\Big(\min\big(\overline{\log\max(\lambda^{\rm pot},10^{-6})}_{\mathcal R},0\big)/0.05\Big)^2 . \tag{28}
$$

## 8. Priors and prior factors

**Table 1. Priors on sampled quantities** (prior scale 1).

| quantity | prior | use |
|---|---|---|
| ξ_ℓ (ℓ = a, f, K, j) | 𝒩([−.0134, .0025, −.0034, .4778], [1.0263, 1.0263, 1.0227, 1.0263]) | w_ℓ = softplus(ξ_ℓ) |
| ξ_r | 𝒩(logit .85, .15) | r_K = expit(ξ_r) |
| **f**, **g** | 𝒩(0, I₆₄) | directions |
| α_a; α_j; α_f; α_K | 𝒩(.5, .5); 𝒩(.405, .5); 𝒩(4.993, .75); 𝒩(−1.98192, 1) | α_K loc = softplus⁻¹(2.582/s) |
| μ_sev; b_late | 𝒩(1.386, .5); 𝒩(−.15, .15) | |
| b_nat | −softplus(𝒩(0, .7)) | ≤ 0 |
| **β**_sev (24); **β**_lag (24) | 𝒩(0, (.5/√24)²); 𝒩(0, (1.5/√24)²) yr | |
| δ₀ | 𝒩(0, 2) yr | |
| log k_hᶜ | 𝒩(log 8, .7) (RC) | k_h = k_hᶜ/s |
| n_h | .5 + 2.05·expit(𝒩(−.04879, .8)) | n_h ∈ (.5, 2.55) |
| ζ; log τ_rec | expit(𝒩(−.5, 1)); 𝒩(log 12, .5) | |
| θ; ν; ρ_fl | Beta(1, 1); Beta(1.5, 1.5); Beta(1, 4) | |
| η; φ₀; q_M; κ | 𝒩(0, 1); Exp(1); Beta(2, 2); HalfNormal(.5) | |
| **β**^N_m (5); **β**^N_s (5); α_dsr; log φ_n | 𝒩(0, .1); 𝒩(0, .5); 𝒩(3.5, 1); 𝒩(2.3, 1) | |
| σ | HalfNormal(1) | route-effect SD |
| logit π^B (12) | 𝒩(logit .005, 1.5) | |
| s_C; p_C; log φ_C; γ_C | 𝒩(0, 5); 𝒩(.6, .3); 𝒩(0, 1); 𝒩(0, 1) | |
| log s^S_era (2); β_E; log φ_S | 𝒩(0, 3); Beta(2, 2)ᵃ; 𝒩(0, 2) | |
| e_r; v_ι | 𝒩(0, 1) | integrated (Laplace) |

ᵃ β_E is sampled on the logit scale.
- **MAP objective.** It adds −log[β_E(1 − β_E)], the Beta(2, 2) density in β_E. By convention the
  MAP objective omits Jacobians.
- **HMC (current code).** HMC mode adds the logit Jacobian for this block, so the HMC target is
  Beta(2, 2) on β_E.
- **History.** At the LA1 measurement the HMC mode lacked that Jacobian, which made the target
  uniform on β_E. The lead fixed this in `laplace_fit.py` (HMC mode only) on 2026-11-08.

**Fixed values.**
- n50_raw = −3.756179;
- log R_tot = 0;
- the departure logits (inert);
- m_max = 0.8; n_br = 2.5; K_min; s = 20; k_P; k_A = 3.

**Prior factors on derived quantities** (inside the model density, no Jacobian). There are two
one-sided coupling factors on realised adult-survival–capacity alignment:

$$
\log\Pi_{\rm coup}=\sum_{\chi\in\{\chi_{\rm coef},\chi_{\rm map}\}}-\tfrac12\min\!\Big(\frac{\operatorname{atanh}\chi-\operatorname{atanh}0.9}{0.1},0\Big)^2. \tag{29}
$$

- χ_coef = **β**_a·**β**_K / (‖**β**_a‖²‖**β**_K‖² + 10⁻¹²)^{1/2} is the coefficient cosine.
- χ_map is the cosine of the land-cell maps (Z̄ − ⟨Z̄⟩)**β**_a and (Z̄ − ⟨Z̄⟩)**β**_K, where ⟨Z̄⟩
  is the land mean. Because the maps are land-mean-centred, χ_map is a Pearson correlation.
- χ is clipped to ±0.999999 before atanh.
- *χ_coef = .598, giving −30.57; χ_map = .939, giving 0.*

A weak local age-structure regulariser adds 100 · mean_{x,t} log Beta(clip(ω_K, 10⁻⁵, 1 − 10⁻⁵) |
1.01, 1.01). Here ω_K = F_K/(1 + F_K) is the juvenile fraction at K, with F_K = F 𝒜(K)/(1 + c).
The NestWatch log-likelihood Σ_n 𝓛_n is also part of the model density.

## 9. Inference

**Objective.** Let Θ be the model parameters. log 𝒫_model(Θ) is the model log density, which
includes:
- all prior sites, the factors of §8 and the NestWatch term;
- the constant log-probabilities of held or conditioned sites: the route-effect site
  `observer_route_eps` held at 0 (≈ −17,575.6), the conditioned n50_raw (−0.093) and the held
  departure logits. These are needed to reproduce 𝒰.

The BBS term is replaced by its route-marginal (20). Let Π_CBC, Π_BBL and Π_spec be the negative
log priors of the nuisance blocks in Table 1. The objective is

$$
\mathcal U=-\big[\log\mathcal P_{\rm model}(\Theta)-\log\mathcal P_{\rm BBS}(\mathbf y\mid e{=}0)\big]+L_{\rm BBS}+\tfrac12\sigma^2+L_{\rm CBC}+\Pi_{\rm CBC}+L_{\rm BBL}+\Pi_{\rm BBL}+L_{\rm spec}+\Pi_{\rm spec}+L_{\rm abs}+L_{\rm nv}. \tag{30}
$$

**Coordinates.** Sampling is on 233 unconstrained coordinates υ:
- 212 model coordinates. Normal-family sites are whitened, Beta sites logit-transformed, and
  Exponential and HalfNormal sites log-transformed.
- Then log σ, the 12 BBL reporting logits, 4 CBC parameters and 4 specimen parameters.

The HMC potential is

$$
U_{\rm HMC}(\upsilon)=\mathcal U(\upsilon)-\log\Big|\det\frac{\partial\Theta}{\partial\upsilon_{1:212}}\Big|-\log\sigma-\log\big[\beta_E(1-\beta_E)\big]. \tag{31}
$$

The last term is the specimen-block Jacobian added by the 2026-11-08 fix (Table 1, note a).

All 19,127 route effects and 1,823 circle effects are integrated out by (19). Neither appears in
υ.

*At LA1:*
- *𝒰 = 201,574.84. The local evaluation matched the TACC value to 2 × 10⁻⁹ nats absolute.*
- *U_HMC = 201,703.58, measured before the β_E Jacobian was added.*
- *That term is −log[.949 × .051] ≈ +3.0 at LA1 [arithmetic].*

**Point estimate, then sampling.**
1. A point estimate is found by L-BFGS on 𝒰.
2. The posterior is then sampled with the No-U-Turn sampler: maximum tree depth 8, whitened
   diagonal mass matrix, 300 warm-up and 1,000 sampling iterations, run in segments, 46 h wall
   limit. These settings come from the TACC slot flags; `run_hmc` itself was not executed in the
   coverage run.
3. The computation runs in float64. Z and Z^d are stored in float32, and the FFT kernels are
   stored as complex64, so each convolution is effectively single precision.

**The starting point is not a stationary point.** *At LA1, max |∇U_HMC| = 908, on* **f**. The
records' Hessian pre-check, on the masked LC19 and LC11 L-BFGS points, found 3–5
negative-curvature directions.

## 10. Model selection and held-out design

**Selection stage.** Candidate settings were compared on **masked twins**, which withhold part of
the BBS record through `observation_mask.holdout_years` and `holdout_blocks` (spatial blocks of
4 × 4 cells, 216 km). The withheld set is the years 2021–25 plus 31 pinned 216 km blocks (records
S:2418).
- Each twin was fitted by L-BFGS on the unmasked rows.
- The held-out score is the route-marginal NLL of the withheld rows. Its computation script is
  outside the executed path [U].
- The candidate's twin, LC19, scored a held-out NLL of 48,024.3. Restart spread in held-out
  scores is about 6–23 nats.

**Diagnostics alongside held-out fit.** The records also report:
- the trough contrast D, a contrast in potential growth λ^pot between the GP fit and an ETF
  reference (definition not restated in the specification [U]);
- the ETF λ rise.

**Final inference.** Final inference uses **all** data (191,895 BBS rows, weight 1); held-out
scores are quoted from the selection stage. Masked and all-data objectives are not comparable.
The all-data runs include 26,728 more surveyed rows and 35,050 more rows in total, because the
twins also withhold 8,322 pseudo-zero rows.

## 11. Sensitivity set

Variants differ from the candidate only in the keys named (`SENSITIVITY_SET.md`). All-data
objectives are given where available; `_vlF` refits are prefixed L0F/LF.

| axis | settings | all-data runs and objectives |
|---|---|---|
| juvenile kernel (decisive) | J330 (b_VL 41.056 km), **J200**, J110 (13.715 km); J60A30 (adult mdd 30 km, no V&L-faithful rebuild) as a lower bound | `_vlF` at the candidate setting: L0F1 J200 201,573.1; L0F2 J330 201,637.5; L0F3 J110 201,710.5 (`_vl`: LA1p, LA2, LA3). Masked LC20, LC21p. No ×0.2 run for J60A30 |
| Allee n₅₀ (floor .0231, J200) | ×0.5, **×0.2**, ×0.33 (likelihood-preferred neighbour), pin −2.1 | ×0.5 201,596.8 (LC31r); ×0.33 201,558.5 (LA7; LFA7 201,558.5); ×0.2 201,573.2 (LA1p; L0F1 201,573.1). Also ×0.5 LC32 (J330) 201,630.5 and LC33 (J110) 201,933.0 |
| K floor (internal) at ×0.2 | .0462 (40 ñ₅₀), **.0231 (20 ñ₅₀)**, .0116 (10 ñ₅₀) | quarter floor (LA4–LA6 series): J200 201,571.9; J330 201,622.5; J110 201,675.5. Masked: LC26–28 (.0462), LB1–3 (.0116) |
| habitat structure | **K-only** (`capacity_demography`) vs four free fields (`exchangeable` + coupling priors on all six pairs) | four-free only at the pin and ×0.33 (masked LC7, LC14); none at ×0.2 |
| coupling tightness | sd .1 vs .05 | masked LC22r–24, no consistent gain |

The candidate stays at ×0.2 and floor .0231, chosen by the user for HMC navigability. ×0.33 is the
likelihood-preferred sensitivity neighbour (15 nats better at J200).

## 12. Pending decisions (to be resolved before the final run is reported)

1. **1902–39 pseudo-zero BBS rows.** 68,704 rows with no route id are counted in (20) in addition
   to the range absences (27). Tests found them inert (about 1–2 log-likelihood units).
   - *Option A:* keep them, as now.
   - *Option B:* drop them (`laplace_fit.py --drop_pseudozeros`), leaving (27) as the only pre-1966
     BBS-grid absence evidence.
2. **DESK states.**
   - *Option A:* keep FIELD54 v1 states, as now.
   - *Option B:* move to `field54prod_v2_ema`, which fixes coastal climate and changes 219 cells.
     This alters Z and Z^d and requires refitting.
3. **Census convention for crowding and the Allee factor** (§4 box). Arrivers enter after the
   added mortality, while adults and stayers enter before it. This is possibly unintended, and it
   matters only from 1993 onward.
   - *Option A:* keep N″, as now.
   - *Option B:* use one convention for all classes.

## 13. Symbols added or renamed relative to `NOTATION.md`

`NOTATION.md` §2–3 remains the reference table.

**Convention.** No *bare* symbol (a letter without subscript or superscript) carries two meanings
in this document. Some base letters carry several *decorated* symbols, each with one meaning:
- m, m_max, m^mig: added mortality, its ceiling, the migrant share;
- s, s_C, s^S_era: gauge, CBC log scale, specimen rate;
- γ, γ_C: Allee rate, CBC drift;
- κ, κ_bear: NB density power, bearing concentration;
- n_50, n_h, n_br, n^B_b: Allee constant, Hill steepness, broods, releases;
- p, p_C: departure probability, effort power;
- the families φ (NB sizes), μ (means), β (coefficient vectors and β_E), α (intercepts), τ (τ_rec,
  τ_E, τ_FFT) and y (observed counts).

Generic dummies (ϖ) are bound locally in their definitions.

| here | NOTATION / MODEL_SPEC | why |
|---|---|---|
| expit(·) | σ(·) (logistic) | σ is the route-effect SD |
| r_K | r (K–demography correlation) | r indexes routes |
| **f**, **g** (bold) | f, g | f is also the fecundity field label; g(N) was the detection function |
| d; a_VL, b_VL; b_A | r (kernel distance); a, b (V&L shape, scale) | r is the route index; a is a field label, b the cohort index |
| u⁰ = N⁰/K | x (departure ratio) | x is the land cell |
| N″ = A′ + J^s + (1−m)J^m | (new) | census for crowding and Allee (§4) |
| υ (233 coordinates) | x | x is the land cell |
| Θ (model parameters) | θ | θ is the survival-brake share |
| σe_r (route offset) | u | u is the crowding ratio N″/K |
| ε_A, ε_k (edge corrections) | e_A, e_k | e_r is the route effect |
| τ_FFT, ϖ (FFT floor tolerance; generic dummy) | τ, u | bare τ is not used; u is the crowding ratio |
| smoothstep(ϖ) in G | G(a) = v³(…), v = clip(a/3) | v_ι is the circle effect; a is a field label |
| **β**_sev, **β**_lag (disease basis coefficients) | ω_sev, ω_lag | ω_K is the juvenile fraction at K |
| n_br (broods) | B | B(x) is the disease basis |
| μ^N_n, S^d_n, α_dsr, **β**^N_m, **β**^N_s, φ_n (nest mean, daily survival, DSR intercept, covariate effects, NB size) | m_n, s_n, a_s, δ_m, δ_s, e^{λ_c} | m mortality; s gauge; δ₀ onset slack; λ^pot; a field label |
| (S^d_n)^{32} inline | π_n | π^B is the reporting probability |
| q_M (Mexico concentration multiplier) | q | q₅₀ is the founder reference |
| logit π^B (reporting logits) | λ^B | λ^pot is potential growth |
| 𝓛_r, 𝓛_n, 𝓛^B (log-likelihoods); L_• (NLL totals) | ℓ_r, ℓ_n, ℓ^B | ℓ is the field index |
| 𝓘_r, 𝓘_ι (Laplace information) | H_r, H_c | H_ℓ are the habitat predictors |
| ψ_ϱ, ψ̄(t), V^E_ϱ (CBC log effort, effort trend, latent-effort variance) | ℓ_ϱ, m(t), v_ϱ | ℓ field index; m mortality; v_ι circle effect |
| ι(ϱ), v_ι, t_ϱ (CBC circle, circle effect, model year) | c(ϱ), v_c | c is the fecundity crowding constant |
| N^C_ϱ, μ^C_ϱ, φ^C_ϱ, yr_ϱ (CBC source abundance, mean, size, calendar year) | A_ϱ, μ_ϱ, φ_ϱ, y_ϱ (year) | A_t are adults; y_ϱ is the count |
| ς, μ^S_ς, s^S_era, 𝒟(N) (specimen row, mean, era rate, detection) | ϱ, λ_ϱ, s_e, g(N) | ϱ is the CBC row; e is the route effect; λ^pot |
| b, 𝒯_b, n^B_b, y^B_{bt}, t^band_b, reg_b, P^B_{bt} (BBL cohort, recovery years, releases, recoveries, banding year, region, cell probability) | i, k, l, n_i, m_ik, b_i, g_i, P_ik | i is the BBS row; k the kernel; g a direction; m mortality |
| E^pre_n, 𝟙^fail_n (nest exposure days, failure indicator) | e^pre_n | e is the route effect |
| 𝓡 (native-viability regions) | R | R is the founder number |
| L_abs, L_nv | C_abs, C_nv | C = sN is the route-count scale |
| 𝒳_abs (absence set), 𝒫 (densities), Π_coup | — | new |
| C½ = 0.819 n₅₀ | (NOTATION V8) | half-point of 𝒜 |

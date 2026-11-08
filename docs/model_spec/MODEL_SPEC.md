# LA1 model specification (as executed)

This file specifies the target that HMC samples for candidate **LA1**. The candidate is J200, Allee n50 ×0.2 of the 11-06 pin, K floor 0.0231 absolute, K-only `capacity_demography` with one-sided Sa–K coupling, and all data.

- **How it was derived.** Everything here was derived from the executed path: the coverage, the numpyro trace at the LA1 point, and the resolved config. See EXECUTED_CODE.md and RESOLVED_CONFIG.json.
- **Code locations.** `src/` = worktree `.worktrees/route-variance` @ `0f73b31`. `LF` = main-repo `research/exp/invasion_sandbox/laplace_fit.py` @ `dee075d`.
- **Notation.** Symbols are defined once, in NOTATION.md. Read the two files together.
- **Tags.** **[E]** measured or quoted, **[L]** inferred from code, **[U]** unknown. Unless tagged otherwise, each equation transcribes the cited executed lines [E].

Numbers in *italics* are values at the LA1 point (trace, `best_params.npz`). They are not priors.

---

## 0. Domain, units, scales

| item | value | evidence |
|---|---|---|
| grid | 67 × 112 cells of 54 × 54 km (2,916 km²); 4,318 land cells x | `metadata.pkl` Ny/Nx/N_land; `Sandbox.cell_km2` |
| years | t = 0…123 ↔ 1902…2025 (annual step); spin-up 40 extra years before 1902 | `data['years']`; `spinup.years` |
| state units | **internal density** N (model units, "relative units") | `simulated_density` |
| gauge | route count = s·N, s = `pop_scalar` = 20 route counts per internal unit | `population_scale_route_counts_per_relative_unit` = 20 |
| birds/km² | no conversion enters the likelihood. The three conversions in the code/records disagree (NOTATION §3) | — |

---

## 1. Habitat fields from the DESK latent Z

**Inputs [E].**
- $Z_t(x)\in\mathbb R^{64}$ (`Z_gathered`, 124×4318×64, float32).
- $Z^{d}_{t,k}(x)\in\mathbb R^{64}$, the land-conditioned path summary for juvenile kernel k = 1…12 (`Z_disp_gathered`).
- $\bar Z(x)$ = the mean of $Z_t(x)$ over 1996–2025 (`Z_ref`, `readout.ref_window`); $\bar Z^{d}_k(x)$ likewise (`data_loading.py:475-485`).
- $z_c$ = the mean of $Z_t(x)$ over the distinct (t, x) with a positive BBS count (`habitat_center_occupied`, `data_loading.py:21-41`).
- Provenance of Z is in DESK_PROVENANCE.md.

**Coefficients (`manifold_prior.form = capacity_demography`)** (`age_priors.py:772-809`, `habitat_geometry.py`). Field order ℓ ∈ (a, f, K, j) = (adult survival, fecundity, capacity, juvenile survival):

$$
w_\ell=\operatorname{softplus}(\xi_\ell),\qquad r=\sigma(\xi_r),\qquad
\beta_a=w_a f,\;\beta_f=w_f f,\;\beta_j=w_j f,\;\beta_K=w_K\big(r f+\sqrt{1-r^2}\,g\big),
$$

where $f,g\in\mathbb R^{64}$, $f,g\sim\mathcal N(0,I)$. The slopes $\gamma_\ell$ are fixed at 1 (`age_priors.py:1267-1284`).
*Values: w = (2.742, 5.399, 6.155, 0.222); r = 0.787.*

**Split readout with a shared temporal direction (`readout.split`, `within_shared = "demographic"`)** (`age_priors.py:950-962`, `age_fields.py:221-263`). The within-cell weights are $\beta^{w}_\ell=w_\ell f$ for every field. The linear predictors are

$$
H_\ell(x,t)=(\bar Z(x)-z_c)\cdot\beta_\ell+(Z_t(x)-\bar Z(x))\cdot w_\ell f .
$$

This reduces to $H_\ell=w_\ell\,h(x,t)$ for ℓ ∈ {a, f, j}, with the **shared demographic habitat index** $h(x,t)=(Z_t(x)-z_c)\cdot f$. Capacity is

$$
H_K(x,t)=w_K\big[(\bar Z(x)-z_c)\cdot(rf+\sqrt{1-r^2}g)+(Z_t(x)-\bar Z(x))\cdot f\big].
$$

**K's *spatial level* uses its own direction. K's *temporal change* uses the demographic direction f** [E, algebra of executed code].

**Path (journey) predictors** take the same form with $Z^{d}_{t,k}$: $h^{d}_k(x,t)=(Z^{d}_{t,k}(x)-z_c)\cdot f$ (`age_fields.py:293-294, 402-403`).

**Vital-rate fields** (`age_fields.py:321-378, 389-404`):

$$
S_a=\sigma(\alpha_a+H_a),\quad F=\operatorname{softplus}(\alpha_f+H_f),\quad S_j=S_a\,\sigma(\alpha_j+H_j),
$$
$$
K^{0}=\operatorname{softplus}(\alpha_K+H_K),\qquad K=K^{0}+K_{\min},\quad K_{\min}=0.0231039\ \text{(internal)},
$$
$$
Q_k(x,t)=\sigma\big(\alpha_a+w_a h^{d}_k\big)\,\sigma\big(\alpha_j+w_j h^{d}_k\big).
$$

- There is no K time trend: `k_trend.enabled` = false, so $w^{\rm trend}=0$.
- There is no disease term on K: `disease_prior.target` = survival.
- $K_{\min}$ = `capacity_floor_internal`. It is added after everything else and is not applied to the native seed.
- $Q_k$ is evaluated at the arrival cell x of kernel k [L: Q is multiplied with each kernel's arrivals on the destination grid, `age_forward.py:266`].

*LA1 ranges over (x, t):*

| field | min | median | max |
|---|---|---|---|
| S_a | 0.0003 | 0.376 | 0.825 |
| S_j | 0.00005 | 0.119 | 0.291 |
| F | 5e-6 | 2.97 | 6.96 |
| K | 0.0231 | 0.0471 | 4.68 |
| K⁰ | 6e-11 | 0.024 | 4.66 |

---

## 2. Density-regulation constants (computed from disease-free rates)

From `equilibrium_age_quantities`, yearling form with `adult_survival_dd` (`age_priors.py:699-711, 1797-1803`). Let $S_a^{\min}=\operatorname{clip}(1-S_jF,0.05,1)$.

$$
c_s=\theta\max\!\Big(\tfrac{S_a}{S_a^{\min}}-1,0\Big),\qquad S_a^{K}=\frac{S_a}{1+c_s},\qquad
c=\max\!\Big(\frac{F S_j}{1-S_a^{K}+10^{-6}}-1,0\Big).
$$

*θ = 1.19e-4, so the survival brake is negligible at LA1.* The constant c makes $S_a^K+S_jF/(1+c)=1$ at N = K, ignoring the Allee and disease terms.

The quantities $F_K=\frac{F}{1+c}A(K)$ and $\omega_K=F_K/(1+F_K)$ feed only the age-structure prior (§7).

---

## 3. Annual cycle (forward simulation)

The cycle is in `age_forward.forward_sim_age_structured`, with `veit_departure` = True, `yearling_breeding` = True, a crowding power of 2, and `allee_relative` = False (forward_calls.json).

**State.** $A_t(x)$ adults and $J_t(x)$ juveniles at the end of year t, after breeding (`Na_grid`, `Nj_grid`); $N_t=A_t+J_t$ (`simulated_density`).

**Initial state (§3.6).** $A^{\rm init}=J^{\rm init}=\tfrac12 N^{\rm sp}$, where $N^{\rm sp}$ is the spun-up *total* (`age_forward.py:544-545`). The spin-up's own age split is discarded.

**One year t, from the start-of-year pools $(A^0,J^0)=(A_{t-1},J_{t-1})$:**

**3.1 Release** (`age_forward.py:613-617`). At $t=t_{\rm rel}=49$ (**1951**) and cell $x_{\rm rel}=(27,89)$:

$$
A^0\mathrel{+}=\tfrac12 R,\qquad J^0\mathrel{+}=\tfrac12 R,\qquad R=\frac{q_{50}}{s}e^{\log R_{\rm tot}}=\frac{0.6852}{20}e^{0}=0.03426 .
$$

$\log R_{\rm tot}$ (`log_inv_total`) is pinned at 0. There is a single release year (`n_years` 1) and no Allee exemption.

**3.2 Departure (Veit & Lewis 1996, both ages)** (`age_forward.py:320-345`). With $x=N^0/\max(K,10^{-6})$ and $N^0=A^0+J^0$:

$$
p=\operatorname{clip}\!\Big(x-\tfrac1{10}\operatorname{softplus}(10(x-1)),\,10^{-6},\,1\Big)\cdot\big(1-\nu\,w_{\rm nat}(x)\big),
$$

where $w_{\rm nat}=I_0/\max I_0$ and $I_0$ is the native seed shape (`initpop_latent` of the inputs; values 0…1 in steps, 350 cells = 1).
*ν = 0.9886: native-core cells almost never emit.*
`dispersal_logit_intercept/slope` and `dispersal_target_capacity_fraction` are computed and then overwritten. They have no effect.

**3.3 Movement** (`age_forward.py:347-376, 203-269, 109-129`):

$$
A'=A^0(1-p)+\mathbb F\!\Big[\mathcal K_A\circledast\frac{A^0p}{e_A+10^{-6}}\Big],\qquad
J^{s}=J^0(1-p),\qquad
J^{m}=\sum_{k=1}^{12}Q_k\,\mathbb F\!\Big[\mathcal K_k\circledast\frac{J^0p}{e_k+10^{-6}}\Big].
$$

- $\mathcal K_A$ is the adult kernel: isotropic, mdd 100 km, shape 0.859.
- $\mathcal K_k$ are the juvenile kernels: 4 directions × 3 radial bands [0, 155.4), [155.4, 482.7), [482.7, ∞) km. The J200 form is the V&L-faithful 2-D density ∝ $\exp(-(r/b)^a)/r$ with b = 24.937 km and a = 0.468 (`KERNEL_DISCRETISATION.json`).
- $e_A$, $e_k$ are the land-fraction edge corrections.
- ⊛ is the linear FFT convolution with zero padding (`numerics.fft_pad` 0).
- $\mathbb F$ is the smooth round-off floor $\mathbb F[u]=u_+^3/\max(u_+^2+\tau^2,10^{-150})$, where $\tau=64\,\epsilon_{\rm mach}\max|u|$ (stop-gradient). The FFT kernels are stored complex64 even in float64 runs [E: dtype in data_keys; `HOUFIN_COMPLEX128` unset].
- **Adult dispersers suffer no journey mortality. Juvenile arrivers survive with $Q_k$ in place of $S_j$.**

**3.4 Epizootic added mortality** (`disease_prior.target = survival`) (`age_forward.py:744-751`; `age_fields.py:100-135`; `disease_onset.py`). Let $N'=A'+J^s+J^m$ be the post-dispersal total. Then

$$
m=\Delta(x,t)\,\frac{(N'+10^{-12})^{n_h}}{(N'+10^{-12})^{n_h}+k_h^{n_h}},
$$
$$
\Delta(x,t)=\mathbb 1[t\ge 91]\;m_{\max}\,\sigma\!\big(\mu_{\rm sev}+b_{\rm late}\,o_{\rm dec}(x)+b_{\rm nat}I_0(x)+B(x)\!\cdot\!\omega_{\rm sev}\big)\;G\!\big(t-o^*(x)\big)\;\big[1-\zeta(1-e^{-\max(t-o^*,0)/\tau_{\rm rec}})\big].
$$

The terms are:
- $m_{\max}=0.8$ (`mortality_ceiling`).
- $o^*(x)=o(x)+\delta_0+B(x)\cdot\omega_{\rm lag}$, with o the arrival map in timestep units.
- $o_{\rm dec}=(o+1902-1993)/10$ (`disease_onset_decades`).
- $B$ is the 24-function land-centred basis. The same array serves severity and lag.
- $G(a)=x^3(10-15x+6x^2)$ with $x=\operatorname{clip}(a/3,0,1)$ (`compact_start`, 3 years).

*At LA1: k_h = 0.0136 internal (0.272 route counts), n_h = 2.18, ζ = 0.0146, τ_rec = 19.2 y, max Δ = 0.431.*

**3.5 Survival and reproduction** (`age_forward.py:1170-1216`). *(Corrected 2026-11-08 after independent verification, VERIFICATION.md.)* Arrivers are reduced by the added mortality before the census is formed (`age_forward.py:751`). Reproduction then sums $N''=A'+J^{s}+(1-m)J^{m}$ (`age_forward.py:1170`) for both crowding and the Allee factor. With $u=N''/\max(K,10^{-12})$:

$$
F^{\rm eff}=\frac{F}{1+c\,u^2}\;\mathcal A(N''),\qquad \tilde S_a=\frac{S_a(1-m)}{1+c_s\,u^2},
$$
$$
A_t=A'\tilde S_a+J^{s}S_j(1-m)+J^{m}(1-m),\qquad J_t=A_t\,F^{\rm eff},
$$

then $A_t,J_t\leftarrow\max(\cdot\,\text{land mask},0)$. All winter survivors breed, because yearlings breed.

The Hill term of §3.4 uses $N'=A'+J^s+J^m$. Before 1993 m = 0, so $N''=N'$.

**Open convention question (pending, user).** In $N''$, adults and stayers enter the census *before* the year's added mortality, while arrivers enter *after* it. This is possibly unintended, and it differs from $N'$ only from 1993 onward.

**Mate-finding (Allee) factor** (`age_forward.py:1105-1117`; `allee_prior` absolute, `aggregation_k` = 3, `per_capita` = true):

$$
\mathcal A(N)=1-\Big(1+\frac{\gamma N}{k_A}\Big)^{-(k_A+1)},\qquad k_A=3,\qquad
\gamma=\frac{s\ln 2}{n_{50}+10^{-6}}=600.0\ \text{per internal unit},
$$

with $n_{50}=\operatorname{softplus}(-3.756179)=0.0231039$ **route counts** (pinned), i.e. $\tilde n_{50}=n_{50}/s=0.00115520$ internal.
- **$\mathcal A=\tfrac12$ at C = 0.01892 route counts = 0.819 n₅₀, not at n₅₀** [E, numerical].
- $\mathcal A$ is evaluated on $N''=A'+J^s+(1-m)J^m$ (arrivers after disease, the others before), before survival. It equals N′ before 1993.
- In route-count units, $K_{\min}$ = 0.462 = 20 n₅₀.

The realised births per breeder $F^{\rm eff}(x,t)$ at the 330 nest cells are returned for the nest likelihood (`nest_F_real`).

**3.6 Spin-up and native seed** (`age_priors.py:1779-1786, 1970-1986`).
- The seed is $N^{\rm seed}=I_0(x)\,K^{0}(x,0)$, using **K⁰ without the floor**, × `core_fraction_of_local_capacity` 1.0.
- The forward model is then run for 40 years. The rates $(S_a,S_j,F,K,c,Q,c_s)$ are frozen at their 1902–06 means, Δ = 0, and there is no release (`inv_t` = 1040).
- $N^{\rm sp}=N_{40}^{\rm spin}$ becomes the 1902 start state (`initpop_spun`).

---

## 4. BBS observation model and route-effect marginalisation

**Mean** (`observation.form = breeding`; `age_priors.py:2061-2110`; LF:679). Row i has cell $x_i$, year $t_i$, observer×route $r(i)$ and first-year flag $\mathrm{FY}_i$:

$$
\mu_i=\max\!\big(s\,[N_{t_i-1}(x_i)+\rho_{\rm fl}J_{t_i}(x_i)],\,10^{-6}\big)\,\exp\!\big(\eta\,\mathrm{FY}_i+\sigma e_{r(i)}\big).
$$

- $N_{-1}\equiv N^{\rm sp}$.
- The route term is absent (u = 0) for the 68,704 pseudo-zero rows (`obs_obsroute_idx` = −1; years 1902–39; 1,808 cells × 38 years; y = 0).
- *ρ_fl = 3.7e-4, so the expected count is the previous year's end-of-year total.*
- [L] Timing: $N_{t-1}$ is the population after breeding in year t−1, i.e. *before* the dispersal and overwinter survival of year t. The code's description of it as "the breeding population of year t" is a modelling convention, not the post-winter breeders $A_t$.

**Likelihood** (`age_priors.py:2113-2163`; LF:618-622). The count is NB2:

$$
\log p(y_i)=\log\Gamma(y_i+\phi_i)-\log\Gamma(\phi_i)-\log\Gamma(y_i+1)+\phi_i\log\frac{\phi_i}{\phi_i+\mu_i}+y_i\log\frac{\mu_i}{\phi_i+\mu_i},
$$
$$
\phi_i=\phi_0\,q^{\,\mathrm{tier}_i}\,\max(\mu_i,10^{-3})^{\kappa}.
$$

- Tier 1 is Mexico, 236 rows.
- All 191,895 rows have weight 1. The holdouts were removed.

**Route effects.** These are $e_r\sim\mathcal N(0,1)$ iid, r = 1…19,127, integrated out per route by a Laplace approximation (LF:638-659). With $\ell_r(e)=\sum_{i\in r}\log p(y_i\mid e)$:

$$
-\log p(\mathbf y_r\mid\theta,\sigma)\approx-\ell_r(e^*_r)+\tfrac12 e^{*2}_r+\tfrac12\log\!\big(1+\sigma^2H_r\big),\qquad H_r=\max\!\big(-\textstyle\sum_{i\in r}\partial^2_u\log p(y_i)\big|_{u=\sigma e^*_r},0\big).
$$

- $e^*_r$ comes from 30 Newton steps from 0, each clipped to ±1. The steps are unrolled, so the gradient flows through them.
- The constant −½ log 2π per route is dropped.
- *σ = 1.196; max |e*| = 4.30.*
- The model's own `observer_route_eps` site is ZeroSumNormal, but LF fixes it at 0 (a constant 17,575.6 in lp). The fixed `sigma_u` = 0.97 does not enter.

**BBS total.**

$$
L_{\rm BBS}=\sum_{r}\big[-\log p(\mathbf y_r)\big]_{\rm Laplace}-\sum_{i:\,r(i)=-1}\log p(y_i\mid u=0)
$$

*= 165,500.34.* The pseudo-zero rows still count, in addition to the range-absence term (§5.4). The LF comment at LF:715-716 says they are removed, but the code is `pass`.

---

## 5. Other likelihood terms (added in LF; all weight 1)

**5.1 NestWatch** (in the model; `age_priors.py:2015-2046, 649-666`; 1,089 attempts in 330 cells, 1902–2020, precision ≤ 1).

$$
m_n=\frac{2F^{\rm eff}(x_n,t_n)}{B}e^{X_n\delta_m},\quad \log s_n=\log\sigma(a_s+X_n\delta_s),\quad \pi_n=s_n^{32},\quad \mu^{(1)}_n=\tfrac1{10}\operatorname{softplus}\!\big(10(m_n/\pi_n-1)\big),
$$
$$
\ell_n=w^{\rm exp}_n\big[e^{\rm pre}_n\log s_n+\mathbb 1_{\rm fail}\log(1-s_n^{d^{\rm last}_n})\big]+w^{\rm cnt}_n\log\mathrm{NB2}\big(\text{fledged}_n-1\mid\max(\mu^{(1)}_n,10^{-12}),e^{\lambda_c}\big).
$$

B = 2.5 broods (pinned). $X_n$ is a 0/1 design over {building, nestbox, other_artificial, substrate_unknown, card}. *−Σℓ = 1,881.22.*

**5.2 CBC counts, 1902–65** (`--cbc_counts`, latent mode, BBL source, linear drift; LF:343-424; `cbc_obs.py`). There are 10,511 circle-years in 1,823 circles, 4,772 with latent effort.

$$
A_\varrho=\sum_x s\,N_{t_\varrho}(x)\,W_{c(\varrho),x},\qquad W_{c,x}=(1-m^{\rm mig}_x)\,W^{L}_{c,x}+m^{\rm mig}_x\sum_{x'}P_{x,x'}W^{L}_{c,x'},
$$
$$
\log\mu_\varrho=s_C+\gamma_C\,\tfrac{y_\varrho-1940}{10}+p_C\ell_\varrho+\tfrac12p_C^2v_\varrho+\log\max(A_\varrho,10^{-12})+v_{c(\varrho)},\qquad
\phi_\varrho=\Big[(1+\phi_C^{-1})e^{p_C^2v_\varrho}-1\Big]^{-1}.
$$

- $W^{L}$: local circle-to-cell weights (`cbc_w_local`).
- $m^{\rm mig}$: migrant share by origin. East (lon > −95) 0.57; west 0.14 (Plains included); Pacific (lon < −117) 0.06.
- $P$: the migrant destination density over land cells. Distance is LogNormal(log 265 km, 0.83) for East origins, with a von Mises bearing at 218°, κ = 6.24 (fitted from BBL pairs). Other origins are isotropic LogNormal(log 337 km, 0.83).
- $\ell_\varrho$: log effort. Rows with missing effort use $\ell=m(t)$, a hat-basis fit at knots 1920…1966 to recorded efforts, with $v_\varrho=\tau_E^2=0.836^2$; otherwise $v=0$.
- $v_c\sim\mathcal N(0,1)$ is Laplace-integrated as in §4: 40 clipped Newton steps, $+\tfrac12 v_c^2+\tfrac12\log(1+H_c)$.
- $N_t$ is the end-of-year total.

*NLL 9,212.40.* Parameters: $(s_C,p_C,\log\phi_C,\gamma_C)$ *= (1.401, 0.554, −0.518, 0.242)*.

**5.3 BBL adult dead-recovery m-array** (`--bbl`, until 2025, `--bbl_disease`; LF:489-532). There are 8,778 cohorts and 543 recoveries.

$$
S^{B}(x,t)=\operatorname{clip}\!\Big(\frac{S_a(1-\Delta\,\mathrm{hill}(N_t))}{1+c_s(N_t/K)^2},10^{-6},1-10^{-6}\Big),
$$
$$
P_{ik}=\Big[\prod_{l<k}S^{B}_{il}\Big](1-S^{B}_{ik})\,\pi^{B}_{g_i,e(b_i+k)},\qquad
\ell^{B}=\sum_{ik}m_{ik}\log P_{ik}+\sum_i\Big(n_i-\sum_k m_{ik}\Big)\log\Big(1-\sum_kP_{ik}\Big).
$$

- $\mathrm{hill}$ is the same Hill term as §3.4, evaluated on the end-of-year $N_t$ rather than the post-dispersal N′ [E: LF:516-519 vs `age_forward.py:746`].
- $\pi^B=\sigma(\lambda^B)$ for 3 regions × 4 eras (<1980, 1980–94, 1995–2005, ≥2006).

*NLL 4,666.49.*

**5.4 Range absences** (`--range_absence` halo400 nativeMX; LF:551-555, 711-714). There are 74,306 (t, x) pairs, t ≤ 1965, all with w = 1.

$$
C_{\rm abs}=\sum_{(t,x)}w\,k_P\log\!\big(1+sN_t(x)/k_P\big),\qquad k_P=0.43724.
$$

This is $-\log$ NB2(0 | mean sN, size $k_P$). *= 23.15.*

**5.5 Museum specimens as counts** (`--specimens` collectors cache, `--specimen_counts`; `specimen_lik.specimen_nll_counts`). There are 5,522 cell-decade rows, 582 of them positive, 1880–1940.

$$
\bar N_\varrho=s\sum_tT_{\varrho t}N_t(x_\varrho),\quad
\lambda_\varrho=\max\!\big(e^{\log s_{e(\varrho)}+\beta_E\log E_\varrho}\,g(\bar N_\varrho),10^{-12}\big),\quad g(N)=1-(1+N/k_P)^{-k_P},
$$

with $y_\varrho\sim\mathrm{NB2}(\lambda_\varrho,\phi_S)$, weight $w_\varrho\in\{0,1\}$ (collector filter [U: filter rule not re-read]), and $T$ the row's decade window (`tmask`).
*NLL 2,415.28; ($\log s_{\rm pre1910},\log s_{1910-40},\operatorname{logit}\beta_E,\log\phi_S$) = (−3.358, −3.296, 2.914, −1.000).*

**5.6 Native-viability penalty** (`--native_viable`, scale 0.05; LF:582-598). The regions are mex (88 cells) and us (257 cells). With $\lambda^{\rm pot}(x)=S_a(x,0)+S_j(x,0)F(x,0)$:

$$
C_{\rm nv}=\sum_{R}\tfrac12\Big(\min\big(\overline{\log\max(\lambda^{\rm pot},10^{-6})}_R,0\big)/0.05\Big)^2 .
$$

*= 2.34.*

---

## 6. Priors (all free coordinates; prior_scale = 1)

Location and scale are on the sampled site. *Values at LA1* are given in italics.

| symbol | site | prior | transform → used as | LA1 |
|---|---|---|---|---|
| ξ (4) | `w_scale_raw` | N([−.0134, .0025, −.0034, .4778], [1.0263, 1.0263, 1.0227, 1.0263]) | w = softplus(ξ) | *ξ = (2.675, 5.395, 6.153, −1.394)* |
| ξ_r | `capacity_correlation_logit` | N(logit .85 = 1.7346, .15) | r = σ(ξ_r) | *1.308 → .787* |
| f (64) | `manifold_factor` | N(0, 1) iid | shared direction | |
| g (64) | `capacity_factor` | N(0, 1) iid | K-only direction | |
| α_a | `alpha_a` | N(.5, .5) | logit S_a at h = 0 | *.181* |
| α_j | `alpha_j` | N(.405, .5) | logit S_j/S_a | *−.719* |
| α_f | `alpha_f` | N(4.993, .75) | softplus⁻¹ F | *4.267* |
| α_K | `alpha_k` | N(−1.98192, 1); loc = softplus⁻¹(2.582/20) | softplus⁻¹ K⁰ | *−1.585 (k_level 3.73 route counts)* |
| μ_sev | `disease_mu_sev` | N(1.386, .5) | | *−1.050* |
| b_late | `disease_b_late` | N(−.15, .15) | | *−.401* |
| b_nat | `disease_b_native_raw` | N(0, .7) | b_nat = −softplus(raw) | *−.570* |
| ω_sev (24) | `disease_w_sev` | N(0, (.5/√24)²) | | |
| δ₀ | `disease_lag0` | N(0, 2) | years | *−2.93* |
| ω_lag (24) | `disease_w_lag` | N(0, (1.5/√24)²) | years | |
| log k_hᶜ | `disease_log_k_half_counts` | N(log 8, .7) | k_h = e^{·}/s (internal) | *−1.301* |
| n_raw | `disease_hill_n_raw` | N(−.04879, .8) | n_h = .5 + 2.05 σ(n_raw); the bound .85(4/.8 − 2) = 2.55 | *2.18* |
| ζ_raw | `disease_rec_raw` | N(−.5, 1) | ζ = σ | *.0146* |
| log τ_rec | `disease_log_tau_rec` | N(log 12, .5) | | *19.2 y* |
| θ | `dd_survival_share` | Beta(1, 1) | | *1.19e-4* |
| ν | `native_departure_r` | Beta(1.5, 1.5) | | *.9886* |
| ρ_fl | `obs_fledgling_weight` | Beta(1, 4) | | *3.7e-4* |
| η | `observer_first_year_eta` | N(0, 1) | | *−.0106* |
| φ₀ | `concentration` | Exponential(1) | | *.532* |
| q | `quality_conc_mult` | Beta(2, 2) | | *.486* |
| κ | `obs_conc_kappa` | HalfNormal(.5) | | *.720* |
| δ_m (5) | `nest_mean_delta` | N(0, .1) | | |
| δ_s (5) | `nest_surv_delta` | N(0, .5) | | |
| a_s | `nest_dsr_logit` | N(3.5, 1) | | *4.48* |
| λ_c | `nest_log_conc` | N(2.3, 1) | | *4.86* |
| log σ | (LF tail) | σ ~ HalfNormal(1) (`sigma_prior_scale`) | σ = e^{·} | *.179* |
| λᴮ (12) | (LF tail) | N(logit .005, 1.5) | π^B = σ | |
| s_C, p_C, log φ_C, γ_C | (LF tail) | N(0, 5), N(.6, .3), N(0, 1), N(0, 1) | | |
| log s_e (2), logit β_E, log φ_S | (LF tail) | N(0, 3) ×2; **density ∝ β_E(1 − β_E) on the logit coordinate**, so the HMC target WAS **Uniform(0, 1) on β_E** when LA1 was measured. *2026-11-08 fix (laplace_fit.py, `--hmc` only): the logit Jacobian β_E(1 − β_E) is now added, so the HMC target is Beta(2, 2) on β_E; the MAP objective is unchanged (≈ +3.0 nats on U_HMC at the LA1 point).*; N(0, 2) | | *β_E = .949* |
| e_r (19,127), v_c (1,823) | integrated | N(0, 1) iid | Laplace | |

On the β_E row: LF:762 says "Beta(2, 2) + logit Jacobian", but only one factor β(1 − β) is present [E].

**Pinned or fixed, not sampled** [E: trace, LF:269].
- `n50_raw` = −3.756179 is conditioned, so its N(−3.4915, .25) log prob is a constant.
- `log_inv_total` = 0 is deterministic.
- `dispersal_logit_intercept` = 2.0 and `dispersal_logit_slope` = 4.0 are excluded from the coordinates and have no effect.
- `observer_route_eps` = 0.
- `disease_tau` = 1.0 is a constant.
- γ slopes = 1; m_max = 0.8; B = 2.5; $K_{\min}$; s; $k_P$; $k_A$ = 3.

---

## 7. Factors on derived quantities (inside the model log density)

| factor | form | LA1 |
|---|---|---|
| `coupling_coef_prior` | $-\tfrac12\min\big((\operatorname{atanh}\chi_{\rm coef}-\operatorname{atanh}.9)/.1,0\big)^2$, with $\chi_{\rm coef}=\beta_a\!\cdot\!\beta_K/\sqrt{\lVert\beta_a\rVert^2\lVert\beta_K\rVert^2+10^{-12}}$ | *χ = .598 → −30.57* |
| `coupling_map_prior` | the same, with $\chi_{\rm map}$ = the uncentred cosine of the land-cell vectors $(\bar Z-\langle\bar Z\rangle)\beta_a$ and $(\bar Z-\langle\bar Z\rangle)\beta_K$ | *χ = .939 → 0* |
| `local_age_structure_regularizer` | $100\cdot\operatorname{mean}_{x,t}\log\mathrm{Beta}(\operatorname{clip}(\omega_K,10^{-5},1-10^{-5})\mid1.01,1.01)$ | *+.076* |
| `nest_loglik` | Σ_n ℓ_n (§5.1) | *−1,881.22* |

These sit on derived quantities and carry no Jacobian. Under HMC the effective prior is their product with the coefficient priors (`age_priors.py:1662`).

---

## 8. The objective and the HMC target

LF:661-764 defines the objective and LF:40-42 the HMC potential. Let θ be the model coordinates, so $\log p_{\rm model}$ is numpyro's `log_density`: all prior sites and factors, including the BBS `obs` site at e = 0. Then

$$
\mathcal U(x)=-\big[\log p_{\rm model}(\theta)-\log p_{\rm obs}(\mathbf y\mid e{=}0)\big]+L_{\rm BBS}+\tfrac12\sigma^2+L_{\rm CBC}+\Pi_{\rm CBC}+L_{\rm BBL}+\Pi_{\rm BBL}+L_{\rm spec}+\Pi_{\rm spec}+C_{\rm abs}+C_{\rm nv},
$$
$$
U_{\rm HMC}(x)=\mathcal U(x)-\log\Big|\frac{\partial\theta}{\partial x_{1:212}}\Big|-\log\sigma .
$$

**Coordinates x (233)** (`local_map_coordinates.py`). The order is as follows.
- 212 model coordinates, sorted by site name. Normal-family sites are whitened, $x=(\theta-\mu)/\mathrm{sd}$ (constant Jacobian). Beta sites map through the logit; Exponential and HalfNormal sites through the log.
- Then log σ, 12 BBL reporting logits, 4 CBC parameters, and 4 specimen parameters. These are already unconstrained and carry no Jacobian.

**Measured values at LA1** [E]: $\mathcal U$ = 201,574.842 (TACC 201,574.842); $U_{\rm HMC}$ = 201,703.579; log|J| = −128.916.

**Not a stationary point** [E]: max|∇U_HMC| = 908 (`manifold_factor`), ‖∇‖ = 2,477. The records' Hessian pre-check reports 3–5 negative-curvature directions at L-BFGS stopping points (SANDBOX_INVASION_SHAPE 11-08, "HESSIAN PRE-CHECK").

---

## 9. Known open points for the spec

| item | status |
|---|---|
| BBS timing: the expected count uses $N_{t-1}$ (after breeding, before winter) as "breeders of year t" | [L] convention; documented differently in code comments |
| Spin-up age split discarded (re-split 50/50 at 1902) | [E] `age_forward.py:544`; consequence not assessed [U] |
| Specimen collector filter rule; `tmask` normalisation (decade mean?) | [U] cache builder not re-read |
| BBL m-array input identity with the TACC `inits/` copy | [L] the objective matches TACC to 2e-9 |
| Whether `Q_k` is indexed by the destination (assumed) or the origin cell | [L] |
| Adult kernel form (2-D area density vs the V&L-faithful juvenile form) | [U] (DESK_PROVENANCE open question) |

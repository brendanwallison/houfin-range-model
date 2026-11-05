# Dispersal expressiveness plan: path-dependent mortality, carefully (2026-10-30)

Status: design only. No model code edited, no fits run. Written for GRID54_LADDER step 20 (amendment: the loose
jdec / jcost / jboth arms are the "what do the data want when free" reference, not candidates).

Code cited: live fit code `~/houfin/work/houfin/research/GRID54_LADDER_20261029/code` at HEAD 7ec22e2 (WSL).
Records cited: age-model records in that tree (`research/age_model/experiments/E0xx_*.md`, `ledger.csv`, `DIGEST.md`,
`CONSIDERATIONS.md` = C1-C12), and this repo's `research/experiments/GRID54_*.md`, `SDM_BASELINE_B*.md`,
`research/CONSIDERATIONS.md` (G/M/P items). Marks: **established** (read in code or measured), **likely** (reasoned,
not tested), **unknown**.

---------------------------------------------------------------------------------------------------------------------

## 0. Summary (read this first)

Five facts about how dispersal works now matter more than any new parameter:

1. **Q is not an extra journey cost; it REPLACES juvenile survival for every juvenile that departs.** Arrivals are
   added to next year's adults with no S_j factor (`age_forward.py:889` `surviving_arrivers = N_j_arrivers`); only
   stayers get S_j (`:888`). In the coupled form Q = S_j evaluated on path habitat (`age_fields.py:357-369`), so a
   disperser's year survival is "S_j of the land around and behind where it lands". (established)
2. **Under the step 3 departure rule almost every juvenile in an established cell departs.** Step 3 uses
   `dispersal_departure.form = veit` (resolved config, runs/step3_obsroute/out/resolved_config.json): p = min(N/K, 1),
   smoothed, both ages (`age_forward.py:222-228`); p = .93 at N = K, .50 at N = K/2. So in occupied cores juvenile
   survival is effectively Q; S_j acts mainly on the stayers of sparse cells (front, sinks). (established from code)
   Consequence (likely): decoupling Q from S_j is not only "journey vs home" -- it is "survival in dense cells vs
   survival in sparse cells", i.e. a new free DENSITY-DEPENDENT juvenile survival that can trade with Allee strength,
   Beverton-Holt crowding and capacity. This is the main identifiability hazard of the journey block and the reason
   "extremely correlated a priori" is the right instinct.
3. **Q is indexed by DESTINATION and cohort (direction x band), averaged over all origins of that wedge weighted by
   kernel mass -- not by where birds actually came from.** Z_disp[p, k] is the kernel-weighted, land-normalised mean
   of Z at points p - s*d along straight segments from p back toward each origin, s = 1/20 ... 1 (`build_path_features.py:
   92-137`). A front cell's outer-band "to_EAST" Q reflects habitat 480 km to the continent's edge to its west whether or
   not any bird came from there. (established)
4. **Path habitat is an arithmetic mean of H along the path, then one sigmoid.** Because H = Z.beta is linear and the
   path operator is linear, H_disp = mean of H over path points exactly; Q = sigmoid(mean H), not the product of
   per-step survival. A short bad stretch (barrier) is diluted (C10). Water is renormalised out at every step, so
   crossing lakes / gulfs costs nothing (`build_path_features.py:116-132`; config `_path_comment`). (established)
5. **The inner band is mostly the destination's own neighbourhood.** At 54 km cells the 0-155 km band has mean distance
   80 km (computed from the V&L kernel: band means 80 / 286 / 867 km, integrated band masses .37 / .42 / .21,
   GRID54_INPUTS_2026-10-29.md sec. 5), and the path samples s*d run from ~4 km to the origin, so Q_inner is close to
   S_j of the destination's 1-2 cell neighbourhood -- nearly collinear with local S_j. Only the outer bands carry real
   "path" information. The soft band boundaries are ~2 cells wide (108 km; `build_kernels.py:170-188`), so the inner and
   middle bands overlap by about +-60 km around 155 km. (established for geometry; collinearity likely)

Identifiability history in one line: every dispersal knob the records opened was absorbed by a compensating change
elsewhere -- departure parameters became an "exodus + regathering" pair worth >1,000 nats (E008/E011/E013), the learned
departure layer clamped ~85% of juveniles at home (E043/E050 user finding), the learned kernel tail flipped sign with
the base (E023 fatter, E041/E043 thinner, step 8a fatter x6/x21), and every brake on the front was bought off by
founders or fecundity (steps 7, 8b, 8b2). Kernel distance itself (250 / 330 / 450 km) did not change the refined basin
(E050c/d).

Recommended first arms (sec. 4), each behind a synthetic null + recovery gate and never combined with band_weights:
- **J1 distance hazard**, ONE parameter: Q_k = Q_k(coupled) * exp(-h * dbar_k / 100 km), dbar = 80 / 286 / 867 km,
  h ~ HalfNormal(0.05) per 100 km (inner-band survival >= .92, outer >= .43 at the 95% prior bound). h = 0 is step 3.
- **J2 water-crossing hazard**, ONE parameter on a mask-only path feature (no Z): Q_k *= exp(-hw * W_k), W_k = mean water
  share along cohort k's paths; hw ~ HalfNormal(1). Needs one small precomputed array (mask + stored kernels only).
- **J3 tight correlated decoupling** only if J1/J2 move the front: one-sided level a_off = -softplus-type cost with
  HalfNormal(0.2) on the logit, slope ratio pinned at 0 first (g ~ N(0, .1) only as a later arm).
- Structural, no new parameters: **4 directions x 4 bands** (split the outer band at ~1,000 km) before any 8-direction
  build; it needs no N_DIRECTIONS change.
Stopping rule (sec. 4.4): stop when an arm fails its synthetic gate, or moves lambda change by < .1 and the frozen-habitat
front by < 5 yr, or buys its effect with fecundity / founders outside the step 3 seed band.

---------------------------------------------------------------------------------------------------------------------

## 1. Implementation map (live code, HEAD 7ec22e2)

### 1.1 Kernels (`src/model/build_kernels.py`, `src/model/dispersal_spec.py`)

| what | where | detail |
|---|---|---|
| kernel family | `build_kernels.py:66-102` `radial_kernel_grid` | 2-D radial generalized Gaussian exp(-(r/scale)^shape), normalised to mass 1 on the toroidal FFT frame (Ly, Lx) = (2Ny-1, 2Nx-1) (`:296-297`) |
| scale from MDD | `dispersal_spec.py:22-29` `get_gamma_scale` | scale = mdd * Gamma(2/a)/Gamma(3/a): juvenile (330 km, a .468) scale 11.5 km; adult (100 km, a .859) 36.0 km |
| values | config `dispersal` (`config/age_model_config.json`), docs/PRIORS.md "Dispersal kernels" | Veit & Lewis 1996 modified Weibull fits to Massachusetts band recoveries INCLUDING non-dispersers (27% of juveniles / 22% of adults recaptured at the banding site, E051) |
| 54 km discretisation | `build_kernels.py:85-102`, `dispersal_spec.py:110-137` | cell_integrated: PDF averaged over source-uniform x destination-uniform displacement (tent per axis), S = 31 subsamples (grid54 overlay `dispersal.kernel_discretization`). Self-cell retention at 54 km: adult .123, juvenile .0475 (point would be .237 / .170; GRID54_INPUTS sec. 5) |
| directions | `build_kernels.py:105-138` `angular_weights_toroidal` | FOUR raised-cosine tapers of half-width pi/2 about N/S/E/W, a partition of unity; the origin's mass is split .25 each (`:135`). "to_EAST" = mass at +x (column) offsets, "to_NORTH" = -y (row) offsets |
| radial bands | `build_kernels.py:158-216` | soft sigmoid boundaries, slope 4/smoothness, smoothness = 2 x cell (108 km at 54 km) (`:170-188`); NOT renormalised per wedge (`:209-211`); kernel index = direction * n_bands + band, direction order N, S, E, W (`:191-214`) |
| splits | config `juvenile_radial_splits_km` = [0, 155.36, 482.74, 1e9] (pinned literals, `dispersal_spec.py:61-104`) | log-spaced legacy edges, not quantiles; band masses .37 / .42 / .21 (integrated), band mean distances 80 / 286 / 867 km (computed here from the radial CDF, gammainc). 5.2% of juvenile kernel mass lies beyond 1,000 km, 0.6% beyond 2,000 km |
| juvenile stack | `build_kernels.py:227-266` `make_juvenile_kernel_stack` | single source for forward kernels AND Z_disp; mass-conservation assert `:263-265` |
| forward struct | `build_kernels.py:269-352` `build_simulation_struct` | adult kernel + edge correction (`:305-309`); 12 juvenile FFT kernels + per-kernel edge corrections computed on the per-wedge NORMALISED shape (`:322-340`) |
| edge correction | `build_kernels.py:140-154` | fraction of a kernel launched from each origin that lands on land (cross-correlation); the forward model DIVIDES dispersers at the origin by it, so no mass is lost to water / off-grid |
| spec guard | `dispersal_spec.py:140-205` | spec recorded in path_feature_meta and checked by exact equality at ingest and fit time; `kernel_discretization` deliberately outside the dict |

### 1.2 Path features (`src/model/build_path_features.py`, driver `src/processing/generate_all_path_features.py`)

| what | where | detail |
|---|---|---|
| operator | `build_path_features.py:92-137` `integrate_paths` | for s in linspace(1/steps, 1, steps) (`:108`; steps = 20 in config): shrink each kernel by s (`resize_kernel_stack`, `:19-58`: fftshift, bilinear resize, pad, ifftshift, restore mass), convolve land-masked Z (`:125`) and the land mask (`:116`), divide (`:128-132`), average over s (`:136`) |
| what Z_disp[p, k] is | same | convolution conv(Z, K_s)(p) = sum_d K_s(d) Z(p - d): the kernel-mass-weighted, land-conditioned mean of Z at points p - s*d, i.e. along straight segments from the DESTINATION p back to each origin of wedge k (for to_EAST: points west of p). Uniform weight over s; s = 0 (the destination exactly) is excluded, s = 1 (the origin) included |
| land handling | `:103-105`, `:116-118` | water set to 0 in the numerator and removed from the denominator at every step: no water-crossing cost, only "habitat on the land part of the path" |
| low land share | none in the ladder code (`has_land_support > 1e-10`, `:117`) | where a (cell, band) sends nearly all mass to water / off-grid, the stored value is FFT roundoff / ~0: 181 of 51,816 cell-kernel pairs (168 cells) at 54 km (GRID54_INPUTS_FDINTERP_2026-10-29.md:67-74). A floor (`land_share_floor`, steps below the floor use the cell's own Z) exists only in THIS repo's uncommitted working tree (`src/model/build_path_features.py`, `generate_all_path_features.py --land-share-floor`; FIELD54_FIELD_2026-10-30.md:112-122). The s64 Z_disp used by steps 3-20 was built WITHOUT it |
| driver mismatch | ladder `generate_all_path_features.py:200-203` | calls `make_juvenile_kernel_stack` WITHOUT `discretization` and writes no `kernel_discretization` key. The s64 54 km Z_disp was built from the `code54` snapshot, which passes it (GRID54_INPUTS_2026-10-29.md:21, :153). Rebuilding Z_disp from the ladder tree would silently use POINT kernels and then fail the ingest guard (`model_inputs.py:289-291`). Any new path build (4 bands, 8 directions, water feature) must use code54 or this repo's working tree, or take the kernels from metadata as `st_correction_basis.py:46-53` does |
| ingest | `src/data/combine/model_inputs.py:257-291, 424-515` | Z_disp gathered to (time, N_land, K, M) memmap; labels must equal the forward kernels' (`:582`) |
| size | GRID54_INPUTS sec. 7 | Z + Z_disp 1.78 GB at 54 km (4,318 land cells, frame 133 x 223), K = 12 |

### 1.3 How Q is made (`src/model/age_fields.py`)

- Path habitat: `H_sj_disp = Z_disp_t . beta_sj - off_sj` (`:268-269`), the same weights as local juvenile survival
  (comment `:257-267`). Under step 2's single field (`manifold_prior.fixed_coupling 1.0`) beta_sj = w_j * f, so it is
  the one habitat field on path points.
- E067 correction: its path image `h_path` enters Q (`:277-286`, `:353-356`), built through the same operator.
- Coupled form (journey off): `Q = sigmoid(alpha_j + gamma_j * H_sj_disp + q_sj)` (`:357-358`).
- Journey block (7ec22e2): `Q = sigmoid(alpha_j + a_off + gamma_j * exp(g) * H_sj_disp + q_sj) * exp(-c_band[k % n_bands])`
  (`:359-364`); priors in `age_priors.py:351-357` (switch parse) and `:1314-1328` (a_off ~ N(0, offset_scale = 1),
  g ~ N(0, log_ratio_scale = .5), c ~ HalfNormal(cost_scale = 1) per band; deterministic `journey_band_survival`).
- juvenile_ratio branch (step 3 uses `juvenile_survival.form = ratio`, ratio logit prior N(.405, .5)):
  `Q = sigmoid(alpha_a + gamma_a * Hs_disp + q_s) * Q` (`:365-369`) -- the adult-survival factor on path habitat times
  the ratio factor. **The journey offset and slope ratio act only on the ratio factor**, so the path-habitat slope of Q
  can never fall below the adult factor's (established; matters for g's interpretation).
- Q is computed from the SAME year's Z_disp (`:228`), checkpointed scan over years (`:372-380`).

### 1.4 Forward dispersal (`src/model/age_forward.py`)

| step | where | detail |
|---|---|---|
| departure | `:182-230` `dispersal_step_age_structured` | logistic `departure_logit` (`:173-179`: intercept + slope (N/K - target_fraction 0.8) + optional crowding arm); adult offset `:221`. **Step 3 uses veit** (`:222-228`): p = min(N/K, 1) smoothed (k = 10), both ages, no parameters -- the logit intercept/slope/target fraction are then unused. Optional native-range scale (E077) `:229-230` |
| per-year noise | `:419` `dispersal_logit_intercept + dispersal_random[t]` | off in step 3 (`dispersal_random_enabled false`); irrelevant under veit anyway |
| adults | `:232-236` | dispersers / edge correction -> one isotropic FFT convolution; **no journey survival**; arrivals join adults and survive with S_a at the destination (`:887`) |
| juveniles | `:238-247` -> `juvenile_dispersal_vectorized` `:142-171` | per kernel: dispersers / edge_correction_k (origin boost, `:159`) -> convolution (`:160`) -> `* Q` (`:168`, Q grid (K, Ny, Nx) scattered at the DESTINATION, `:378-393`) -> optional band weights (`:169-170`) -> sum over kernels |
| band weights | `:124-139` `juvenile_band_weights` | w_b = exp(theta_b)/sum_b M_b exp(theta_b) from kernel masses (zero-frequency FFT term), so mass-preserving BEFORE Q; theta = 0 is the current kernel. `N_DIRECTIONS = 4` (`:121`) |
| settlement | `:437-477` | off in step 3 (`settlement.form none`) |
| disease | `:496-505` | added mortality also multiplies arrivals (`:505`) |
| survival | `reproduction_age_structured` `:854-891` | stayers x S_j (`:888`); **arrivals x 1** (`:889`): Q is the arrivals' whole survival term |

What each Q means spatially (established): Q[k, p] = survival of juveniles ARRIVING at cell p via cohort k (direction
of travel k_dir, distance band k_band), computed from habitat on land points between p and the wedge's origins.
Approximations: (i) one value for all origins of a wedge (origin-averaged, kernel-mass-weighted, not
disperser-weighted); (ii) mean of H then sigmoid (no product, no minimum); (iii) water and off-grid excluded rather
than costed; (iv) same-year habitat; (v) adults pay nothing; (vi) the origin boost (edge correction) re-routes mass that
would land in water onto land destinations at no cost.

### 1.5 Diagnostics already reading Q
`src/vis/barrier_crossing.py` (directed low-density crossing operator; its docstring calls Q "transit mortality" --
correct as a sub-stochastic operator, consistent with Q replacing S_j); `scripts/viz/map_diagnostics.py:1600-1910,
2545` (journey asymmetry W vs E, G-horizon W->E / E->W). `reconstruct_map` uses `Predictive(build_model_2d)` with the
fitted latents (`map_diagnostics.py:230-263`), so research/exp/grid54_front_speed.py (frozen fronts) picks up journey
parameters automatically when present (established by reading; verify on the first journey run -- memory "test agreed
design end to end").

---------------------------------------------------------------------------------------------------------------------

## 2. What is learnable now vs pinned, and what happened when each was opened

Step 3 (the ladder base) resolved config: veit departure, band_weights false, journey off, dispersal_random off,
settlement none, juvenile_survival ratio, single field, relative Allee n50 pinned, founders free.

| site | step 3 status | history (record IDs) |
|---|---|---|
| `dispersal_logit_intercept`, `_slope` | SAMPLED (`age_priors.py:1054-1055`, N(2,1), N(4,1)) but UNUSED under veit -> prior-only latents in the AutoDelta guide (remove or pin before any curvature check, else they show as pure-prior directions) | E004 pre-reg: fitted intercept -.58 vs prior 2 (suppressed dispersal sharpens the invasion curve). E008: under K-settlement the curve became a strong NEGATIVE density dependence (-4.52/-8: 87% leave at N=0, 1% at N/K .8) worth ~1,260 nats, and catastrophic without settlement (E011 +11,288). E012: adult offset -88 nats, slope unchanged; E037 natal60: realistic natal departure (60% at target) "free" with adult offset -7.4. E043: the learned layer kept ~85% of juveniles home on top of literature kernels that already encode staying ("massive clamp", user); E050: literature dispersal fixed, not learned. E051: veit eq. 18 adopted (best literature MAP, converged refine) |
| `dispersal_logit_adult_offset` | off (`:1058-1064`) | E012 / E037 / E041 / E043 above (-2.3 to -7.4) |
| `dispersal_crowding_beta` | off (`:1104-1107`) | E013: not demanded with settlement (beta .19), demanded without (6.98; 60% of settlement's gain) |
| `dispersal_random` | off (`:1113-1117`) | run_18 probe: 124 directions at prior curvature, moved ~3e-5 (comment `:1110-1112`; PRIORS.md) |
| kernel MDD / shape | pinned in inputs (C6: learnable distance conflicts with precomputed Z_disp; "distance is a discrete sweep") | E050a/c/d (330 / 250 / 450 km): every MAP realistic, every refine contorts the same way; "kernel reach does not change the refined basin". E051: adult shape bug (.468 -> .859) fixed |
| radial splits | pinned literals | PRIORS.md "Radial splits": not quantiles of the 330 km kernel |
| `dispersal_band_logw` (band_weights) | off | E022/E023 (run_32 overlay `map_run32_kernel_tail`): dm_band outer log-w +.89 / +1.74 (fatter), -184 nats. E041 c40_bio_natal: tail shrinks -.81 / -1.58. E043 c40_kern (departure 98%): outer tail x0.07 ("330 km too long once everyone disperses"). GRID54 step 8a: +1.81 / +3.06 (x6, x21 fatter), lambda change -.21, held-out flat, frozen front unchanged; step 9 (fdi Z): +2.14 / +3.46. **Sign of the tail estimate depends on the departure layer and base**: weakly identified, trades with departure and fecundity (8a: F 11.0 -> 7.2). Era-varying weights: step14_dispera (S-disp screen), result not yet in the records |
| `native_departure_r` | off | E077: 77n NULL pass (r .06 refined), 77r RECOVERY pass (.56 vs .6), real r .65 refined -- the data want a ~65% more sedentary native core; first refine without plains-K collapse |
| settlement | off | E008 (-1,729, largest gain then), E014 (fitness IFD flattens vital rates), E030/E034 (run_36 overlay: dispersers-only; ct_ds +7,629 breaks the exodus), E074 (best-looking MAPs, every refine still contorts, held-out slightly worse) |
| departure by age / crowding | off | run_26 = E012 (age offset), run_27 = E013 (crowding arm) |
| journey a_off, g, c_band | off (new, 7ec22e2) | step 20 loose arms jdec / jcost / jboth running on TACC (reference only); jboth8a withdrawn |

Identifiability lessons from this history (each established in its record):
- L1 Dispersal knobs get used as something else: departure became an exodus-and-regather brake (E008/E012), a home clamp
  (E043), band weights traded with fecundity (8a).
- L2 Sign instability: the tail went fatter (E023, 8a, step 9) and thinner (E041, E043) depending on the base. A
  parameter whose sign flips with the base is not being measured.
- L3 Free levers buy off any brake: founders (step 7 allee005: +4.5 sd), fecundity (allee010/020, 8b2 F 14.5), the release
  exemption. Pin or watch them in every brake-like arm (8b's pin silently failed: log_inv_total outside sample_priors;
  guard added 6e96506).
- L4 Synthetic gates catch some of this and can fail: E077n/r passed; E064n went non-finite; E070r recovered an
  injected pulse at only 29% (strong shrinkage).
- L5 MAP vs refine: the contortions appear in refinement (E050, E051), so every gate is read at the refined point.

---------------------------------------------------------------------------------------------------------------------

## 3. Design options

Common principle: every new piece is zero at its default (step 3 exactly), one parameter where possible, sign-
constrained to "costs only", with a prior that keeps journey survival tied to S_j. All are read against step 3
(seeds 41/42/43: lambda change +.743 / +.742 / +.766) with the step 20 loose arms as the data's direction.

### 3.0 Literature anchors (what is established vs assumed)

- Kernel: Veit & Lewis 1996, Am Nat 148:255-274, https://www.journals.uchicago.edu/doi/10.1086/285924 -- juvenile mean
  329 km, a .468; adult 98 km, a .859; band recoveries incl. non-dispersers; weak Allee + long-distance dispersal explain
  the slow-then-fast eastern spread (established for their data; E051 notes).
- Eastern partial migration: Able & Belthoff 1998, Proc R Soc B 265:2063-2071, https://pmc.ncbi.nlm.nih.gov/articles/PMC1689500 --
  by the early 1960s 36% of eastern birds moved > 80 km from the banding site, later 28-54%; the southern Californian
  source population ~80% sedentary (established). Belthoff & Gauthreaux 1991, Condor 93:374-382,
  https://sora.unm.edu/sites/default/files/journals/condor/v093n02/p0374-p0382.pdf (partial and differential migration
  in the East). **Implication (unknown, cheap to check):** if V&L's recoveries were not restricted to the breeding season,
  their juvenile kernel mixes seasonal migration with natal dispersal, and its fat tail would overstate range-expansion
  speed -- one candidate for the too-fast front. Check V&L's methods (the user has the PDF; E051) before any kernel arm.
  Chu & Claramunt 2023, Ecol Evol 13:e9789, https://pmc.ncbi.nlm.nih.gov/articles/PMC9909001/ restrict recoveries to
  the breeding season and range precisely "to avoid the incorporation of migratory movements" (method established; no
  House Finch value found).
- Western sedentariness: Able & Belthoff 1998 (above); E077's real fit wanted the native core ~65% less dispersive.
- Dispersal costs exist in the transfer phase but are rarely measured as mortality per distance: Bonte et al. 2012,
  Biol Rev 87:290-312, https://www.mv.helsinki.fi/home/msaastam/pdfs/Bonte%20et%20al.%202012_BiolRev.pdf (review;
  established that costs exist; magnitudes taxon-specific). Post-fledging survival of passerines .23-.87 over the first
  3 weeks, habitat effects in 63% of studies: Cox et al. 2014, J Wildl Manage 78:183-193,
  https://research.fs.usda.gov/treesearch/45642 (established). **No House Finch estimate of disperser vs philopatric
  survival or of mortality per km was found: the priors below are ASSUMED scales, chosen to keep effects modest.**

### 3.1 (a) Correlated decoupling of Q from S_j

What "extremely correlated" means here. With the slope ratio pinned (g = 0), Q and S_j read the same field with the same
slope; an offset is a pure level shift, so their SPATIAL pattern correlation stays ~1 whatever a_off is. Pattern
decoupling only comes from g (or a separate field). So the risk in (a) is mostly about LEVEL, and because of fact 2 (veit:
cores disperse ~93%), a level offset is a dense-vs-sparse survival contrast.

Options, most conservative first:
- (a1) one-sided level cost: Q = Q_coupled * exp(-d), d ~ HalfNormal(0.2) (equivalently a uniform band cost). 95% prior
  bound d < .39 -> disperser survival >= .68 of the coupled value. Direction is biological (transfer costs, Bonte 2012);
  scale assumed. Default d = 0 is step 3.
- (a2) two-sided logit offset with a tight prior: a_off ~ N(0, 0.2) (the jdec form with offset_scale .2, log_ratio pinned
  0). Near the operating point (ratio logit ~.4) +-.4 logit moves the ratio factor by about +-10% relative. Two-sided lets
  journeys be SAFER than staying, which has no biological support; prefer (a1).
- (a3) slope ratio: g ~ N(0, 0.1) (exp(g) within .82-1.22 at 95%). Only after (a1) shows a signal. Note (sec. 1.3) g acts
  on the ratio factor only.
- (a4) correlation prior on the pairs (alpha_j, gamma_j) vs (alpha_jq, gamma_jq): equivalent to (a2)+(a3) with
  correlated deviations; adds nothing over tight independent deviations at MAP. Not recommended.
Identifiability risk: d trades with fecundity and K in dense cells (exporters lose recruits -> F up), with the relative
Allee (front cells are sparse, mostly S_j -> unaffected, so d slows the front only through lower export), and with
founders. Pathology to watch: d used to kill the WEST's export (frozen native-seed front: West crosses the Plains 75 yr
early, front_speed.json) rather than to slow the eastern front -- read W->E flux and the native-seed frozen front.

### 3.2 (b) Path-dependent mortality forms

- (b1) **distance hazard (recommended first)**: c_k = h * dbar_band(k) / 100 km, dbar = 80 / 286 / 867 km; h ~ HalfNormal(.05)
  per 100 km. Prior median h .034 -> survivals .97 / .91 / .75; 95% bound .098 -> .92 / .76 / .43. Monotone in distance by
  construction, one parameter (jcost has three free HalfNormal(1) costs, allowing outer survival e^-2 = .14 at 1 sd).
  Identifiability risk: collinear with a uniform cost (b1 vs a1 differ only in how cost grows with band) and with
  band_weights (a mass-NON-preserving tail cut vs a mass-preserving reallocation; 8a moved the tail the OTHER way) --
  never fit b1 with band_weights or with a1 in the first round. Prediction is genuinely uncertain: 8a says the data
  want MORE long jumps on this base.
  Implementation (no new data): a one-parameter variant of the existing band_cost (`age_priors.py:1323-1328`), with
  dbar computed from the stored kernels at load time.
- (b2) **water-crossing hazard**: W_k(p) = mean over s of (1 - land share of the s-scaled wedge) -- the quantity the
  operator currently normalises away (`convolve_mask_step`, `:78-90`). Mask-only, so no Z, no learned-beta dependence,
  one array (K, N_land) ~ 200 KB, static in time. Q_k *= exp(-hw * W_k), hw ~ HalfNormal(1) (hw = 1: a path that is half
  water keeps e^-.5 = .61). Biology: open-water crossing is a plausible barrier for a small finch (assumed). Build with
  the input's stored kernels (as `st_correction_basis.py:46-53`), not the ladder builder (point kernels). Also fixes,
  as a side effect, nothing about the 181 low-land-share pairs -- apply the floor first (sec. 1.2).
- (b3) **path bottleneck / soft-minimum of habitat**: needs a path feature that is NONLINEAR in the learned field, so it
  cannot be precomputed on Z. Options: (i) in-model path integration of a scalar field u = exp(-kappa H) or log S_j(H) with
  the 12 kernels x 20 scales: ~240 extra FFTs per year vs ~13 in the forward step now -> roughly an order of magnitude more
  forward cost; with 5 scales ~4x; (ii) a precomputed bottleneck of a FIXED scalar proxy (e.g. the step 3 refined H frozen,
  or an external barrier covariate), no gradient through H -- cheap, but a frozen fitted field is circular and a
  covariate needs its own justification (M25: a family's name is not its content). Defer: build only if (b1)/(b2)
  show that path cost moves the front and the residual pattern points at specific barriers (Great Plains corridor).
- (b4) terrain (Rockies) crossing: as (b2) on an elevation-roughness mask; data availability on the 54 km grid not
  checked (unknown). Low priority: the front problem is in the East (GRID54 front-speed diagnostic), where terrain barriers
  are weak.
Pathologies common to (b): cost of long jumps lowers the colonisation of distant sink-like cells, so the fit may raise
fecundity or the habitat amplitude to recover them; it may also deepen the early-East sink less (good) or relocate the
gating into Allee-dead cells (watch Allee-dead share; C3: Allee-dead vs lambda trade is non-identifiable).

### 3.3 (c) Re-opening kernel parameters and departure

- Juvenile MDD, shape fixed. True in-model MDD needs Z_disp recomputed in-model (C6) -- reject. Cheap consistent proxy:
  a ONE-parameter band-mass tilt implied by MDD m: w_b(m) = mass_b(m)/mass_b(330) on the fixed splits (masses from the
  radial CDF, as computed here: 250 km -> .48/.39/.14, 450 km -> .28/.40/.32), applied through `juvenile_band_weights`;
  Q stays matched. Prior log(m/330) ~ N(0, .2) (95% 223-490 km, spanning E050's sweep). Literature: V&L 329 km is the only
  House Finch estimate; the migration-contamination point (3.0) argues for allowing shorter (assumed). Risk: E050 found
  the refined basin insensitive to 250-450 km, and band weights' sign flipped across bases (L2) -- expect weak
  identification. Lower priority than (b1).
- Shape: keep pinned (V&L).
- Departure: keep veit (parameterless, E051 best literature fit). Re-opening the logistic layer reproduces L1 (exodus /
  clamp). The one departure extension with passing synthetic gates is E077 native_departure (r ~ Beta(1.5, 1.5));
  porting it to 54 km is a legitimate, literature-motivated arm (western sedentariness, Able & Belthoff) and it acts on
  the West's export directly -- a cleaner lever for "West too exporting" than journey mortality.

### 3.4 (d) Finer paths

- Code that assumes 4 directions: `build_kernels.py:105-138` (four tapers, width pi), `:191` direction order;
  `age_forward.py:121` `N_DIRECTIONS = 4` (`:135-139`); `age_priors.py:1317` (`_Kk // 4`), `:1357`, `:1367`, `:1371`;
  `generate_all_path_features.py:87` and `scripts/validation/test_path_features_single_year.py:80` (`K // 4`);
  `scripts/viz/map_diagnostics.py:543`, `src/vis/visualize_age_model.py:452`, `src/vis/barrier_crossing.py` (labels /
  12-cohort docstring). 8 directions needs raised-cosine tapers of half-width pi/4 spaced pi/4 (still a partition of
  unity) and the index convention kept direction-major.
- Bands are already generic (`n_bands = K // 4`; tests exercise K = 16 per `build_kernels.py:167`): **4 x 4 needs NO
  direction code change**, only new splits + a Z_disp rebuild. Splitting the outer band (483 km to infinity, mean 867 km)
  at ~1,000 km isolates the 5% of mass that jumps > 1,000 km/yr -- the part that blurs barriers most.
- Memory / compute at 54 km: Z_disp scales with K: 12 -> ~1.64 GB, 16 -> ~2.2 GB, 32 (8 x 4) -> ~4.4 GB (step 20
  pre-reg); forward FFT convolutions per year 12 -> 16 -> 32 (frame 133 x 223, cheap relative to the scan's other work,
  likely < 2x step time at 32; not measured). Path build is CPU/GPU minutes-hours (fdinterp build: 8.9 min CPU for K = 12).
- Information without parameters: under the coupled Q, more kernels add directional / distance resolution to Q and to
  the band-weight / hazard handles at zero parameter cost. But inner-band directions at 54 km are barely resolved (r ~
  1.5 cells), so 8 directions mainly sharpen the outer bands; whether that changes the fit is unknown. Order: 4 x 4
  first; 8 x 4 only if 4 x 4 or (b1)/(b2) shows the outer band matters.
- Prerequisite for any rebuild: the code54 builder (or this repo's working tree) with cell_integrated kernels and the
  land-share floor; the ladder's own driver would produce point kernels (sec. 1.2).

---------------------------------------------------------------------------------------------------------------------

## 4. Guards and sequenced plan

### 4.1 Guards for every arm (pre-registered pattern of E077 / misspec suite A1-A3)

- G-null (synthetic): truth = step 3 refined point, structure under test ON at its zero (h = 0 / d = 0 / hw = 0), counts
  from `research/age_model/exp/simulate_observations.py --run-dir <step3 refined> --set <site>=<tiny>` on the real design
  (the simulator refuses missing latents; note HalfNormal latents cannot be set to exactly 0 in log space -- use 1e-4),
  fit the arm cold MAP -> refine. PASS: recovered parameter inside its prior's lower 50% (e.g. h < .034, outer-band
  survival > .9), lambda change within the step 3 seed band (+-.02), fecundity within +-5%, S_a / S_j / founders within
  seed spread, frozen-habitat front within +-3 yr of the truth refit.
- G-recovery: truth = step 3 + a known cost (h = .05 -> survivals .96/.87/.65; or hw = 1), same pipeline. PASS: recovered
  within +-50% of the truth AND no leakage: fecundity, habitat amplitudes (w_scale), Allee, founders within the null
  refit's band.
- G-cross (leakage the other way): truth = step 3 with fecundity intercept lowered (misspec A3 style, front ~10 yr
  slower), fit WITH the journey term: PASS if the journey parameter stays at its null value (the term must not absorb a
  demographic change).
- Curvature at the refined MAP: `src/model/age_hmc_probe.py` (dense Hessian, eigen-spectrum) restricted to the dispersal
  parameter + {fecundity intercept and amplitude, alpha_j, alpha_a, ratio, founders, n50 if free}; report the Schur-
  complement (marginal, not conditional) correlations; |corr| > .9 with fecundity or founders = not identified. Drop the
  unused `dispersal_logit_*` latents first. Cheaper and more robust for a one-parameter arm: a 3-point profile (parameter
  pinned at 0, mid, high; warm-started refines) -- as E001/E002 did -- reading how F / founders move along it.
- Real-data readouts: lambda change (all-land, central-east, east of 80W; research/exp/grid54_band_lambda.py), paired
  S-real overall / east (research/exp/rescore_run.sh, B0 scorer; T marginal is a coarse tie-check only), fronts
  (research/exp/grid54_front_speed.py: fitted, release-only, frozen release-only, frozen native-seed), journey_band_survival
  and the share of juvenile survival carried by Q in cores vs fronts, fecundity / S_a / S_j / founders, west sink share,
  W->E and E->W G-horizon (map_diagnostics), Allee-dead share. Takeover-style rule: an arm whose effect is bought by F or
  founders moving outside the step 3 seed band is read as compensation, whatever its score.
- Interpretation always cites research/CONSIDERATIONS.md G3 (habitat should carry habitat; a front fitted as "habitat
  improved" is the D3 failure), G1 (neither more nor less movement is a goal), P1/P3/P4, and age-model C3, C6, C10.

### 4.2 Sequence

0. **Read the loose reference** (jdec / jcost / jboth, TACC) when it lands. Pre-registered reading: the SIGN and band
   pattern of c, the sign of a_off, and what F / founders / S_j did. If the free fit puts c ~ 0 on the outer band and a_off
   ~ 0, the conservative arms are unlikely to matter (go to step 4); if it wants large costs with F rising, that is L3
   (buy-off) and J1's guards become decisive. Not a candidate either way.
1. **Prerequisites (no fits):** (i) read V&L's methods on recovery season (migration contamination of the kernel, 3.0);
   (ii) decide the Z_disp builder (code54 / working tree with cell_integrated + land-share floor); (iii) implement J1 and
   J2 as default-off switches with exact-zero tests and a check that the readouts (front_speed via Predictive) see them.
2. **J1 distance hazard** (h ~ HalfNormal(.05)/100 km): G-null + G-recovery locally (GPU) in parallel with nothing else
   heavy; then the real arm on step 3. Decides: does a monotone distance cost slow the frozen-habitat release front and cut
   lambda change without buy-off?
3. **J2 water hazard** (hw ~ HalfNormal(1)): same gates; can run alongside J1's real arm (CPU path feature build is small).
   Decides: are unpriced water crossings (Great Lakes, Gulf, Atlantic shelf) part of the over-fast spread?
4. **Structural 4 x 4** (outer band split ~1,000 km, coupled Q, no new parameters; then J1 on it if J1 passed). Decides
   whether outer-band resolution changes Q and the front. 8 x 4 only if 4 x 4 moves the front by >= 5 yr.
5. **J3 decoupling** (a1 one-sided level, HalfNormal(.2), g pinned) only if J1 or J2 passed and moved the front; its null
   must show d stays ~0 when the truth is coupled (the dense-vs-sparse confound of fact 2 makes this the likeliest leak).
   (a3) slope ratio N(0, .1) after that, if at all.
6. In parallel as a separate lever (not path mortality): **native_departure** ported to 54 km (E077 form and gates), since
   it targets the West's export directly.

### 4.3 What each result decides
- J1/J2 pass gates AND real arm cuts lambda change by >= .1 with frozen front slower by >= 5 yr and S-real east within the
  seed floor -> candidate rung; next rung per sequence.
- Gates pass, real effect < .1 / < 5 yr -> path cost of that form is not the lever at 54 km; record and move on.
- Gate fails (leak into F / habitat / founders, or null not recovered) -> the form is not identified on this design; do
  not run or read the real arm.
- Real arm moves F / founders outside the seed band -> compensation (L3), not a candidate.

### 4.4 Stopping rule
Stop adding dispersal freedom when two consecutive conservative arms (J1, J2, 4 x 4) either fail their synthetic gates
or pass but move lambda change by < .1 and the frozen-habitat front by < 5 yr. At that point the record should say:
"path-dependent mortality, in identifiable forms, does not carry the eastern gating at 54 km", and the effort returns to
the front-speed defect through other routes (kernel provenance / migration contamination, native departure, establishment
terms of the misspec suite). Never open more than one new dispersal parameter per arm, and never combine a cost with
band_weights.

---------------------------------------------------------------------------------------------------------------------

## 5. Mechanism ledger
- Q replaces S_j for arrivals; veit departure makes Q the core juvenile survival: established (code).
- Q destination-indexed, origin-averaged, mean-then-sigmoid, water-free: established (code).
- Decoupling Q from S_j acts as density-dependent juvenile survival: likely (follows from the two facts above; G-null
  for J3 tests it).
- V&L juvenile tail inflated by eastern partial migration in band recoveries: unknown (check V&L methods).
- Distance / water path costs slow the front without buy-off: unknown (J1 / J2).
- Outer-band resolution matters: unknown (4 x 4).

## User decisions and additions (2026-10-30, evening)
- Approved to try: (1) stochastic establishment at the front (Poisson colonists, occupancy closure E063 revisited at
  54 km, b anchored); (2) conspecific attraction in settlement (weight (eps + N/K)^a, a >= 0 tight, a = 0 = current);
  (5) port native/partial-migration departure (E077) to 54 km; (6) terrain-crossing hazard (built like J2 W, data only).
- Dropped: path cost on a human-associated covariate (user: hacky).
- Kernel provenance (read from veit1996.pdf pp. 264-265): BBL records of finches banded in Massachusetts 1960-90; 45
  hatching-year birds later recaptured (33 used in Fig. 5B) and 27 adults (21 used); distances banding -> recapture with
  NO stated restriction on recapture season, in a partly migratory population; Weibull fit a_J .468, b_J 41.056 km
  (mean 329 km); 27% of juveniles / 22% of adults recaptured where banded. The fat juvenile tail rests on ~33 birds
  (max ~700 km). Belthoff & Gauthreaux (1991), 333 recaptures across eastern North America, report a pooled mean of
  109 km. E050c/d tested juvenile MDD 250 / 330 / 450 only. Candidate data-anchored sensitivity: juvenile kernel scaled
  toward the B&G distribution (MDD ~110-200 km), shape fixed.
- Under consideration (user question): limited path-dependent decision-making = DIRECTIONAL CHOICE at the origin --
  departing juveniles split across the 4 direction kernels of each band by softmax(kappa * H_ahead), mass-preserving;
  H_ahead for direction d at origin p is available from existing Z_disp (the opposite-direction kernel at p integrates
  the habitat that lies in direction d). kappa >= 0 tight; kappa = 0 = current. Never fitted together with free
  journey mortality first; guards incl. cross-leak both ways (inject mortality -> kappa ~ 0; inject kappa -> mortality ~ 0).

## Correction (2026-10-30, user): stochastic establishment is NOT new and reintroduces the absolute scale
History: the ABSOLUTE (Poisson-encounter) Allee was the long-standing form -- Allee-dead carving (14-23% of suitable
plains/west, E070/E071), the establishment switch the refine exploited (E069, probe 0321jbz), every clean-regime stall
(71a, 73h0, 75d, 76b; E079). RELATIVE Allee (n50/K; E071c, ladder step 1) is the recent fix (Allee-dead 0 by
construction). Stochastic establishment = E063 occupancy closure, tested at 27 km on the absolute base: no better
conditioning at the invasion threshold (fine sweep) and it did not prevent the contortion (63a-c: F 10-15, Allee-dead,
western sink returned). Idea (1) would bring back an absolute scale (b x arrivals). Revised order: conspecific
attraction (scale-free on N/K; moves birds) FIRST; (1) only as a retest on the new base with relative Allee kept,
founders pinned, b pinned to a literature birds-per-count conversion (never learned), and Allee-dead share + refine
stall diagnostics (median |grad|, iterations) as explicit gates.

## The soft (aggregated-encounter) Allee is already in every ladder arm (E079; read 2026-10-30)
E079: the absolute Poisson-encounter Allee's stalls trace to the establishment separatrix; the soft form averages
encounter over gamma-distributed (clumped) within-cell densities, 1 - (1 + gamma N / k)^(-k) (k -> inf Poisson, k = 1
hyperbolic), same low-density slope (still strong), broader transition; k = 3 (79b) converged and keeps mate
limitation; smoother but not stall-proof (80f). Ladder step 1 = relative Allee WITH aggregation_k 3.
Consequences: (1) stochastic establishment would double-count low-density stochasticity the k = 3 form already
represents and would re-sharpen the threshold E079 softened -> DROPPED unless the user asks for it. (2) Likely reading of
the step 7 profile: the broad transition makes a larger n50 brake sparse interior cells too, not only the vanguard --
consistent with held-out falling monotonically without much front slowing (untested). (3) Conspecific attraction and
k = 3 both encode clumping (settlement vs mating); k stays pinned; each attraction arm reports the implied mate-finding
success at the front so the two assumptions are read together.

## User decision (2026-10-31): realism of path survival is an inherent improvement
"Anything that improves the realism of path survival" is wanted even if neutral on the pathologies. Next build:
COMPOUNDING path survival (segmented path features; a per-distance crossing survival compounded along segments on top
of the existing annual coupled term; limit = current model exactly). Design note first (units: annual juvenile survival
must not be compounded per 100 km), then build, guards (recovery + model-comparison), step 30.

## Design note: compounding path survival (2026-10-31, user-approved priority; before building)
User: "anything that improves the realism of path survival is an inherent improvement, even if neutral on the
pathologies". Problem (C10, fact 4 above): Q = sigmoid(alpha_j + gamma_j * MEAN path habitat) (x the adult-factor sigmoid
under juvenile_survival.form ratio), so a long stretch of poor habitat is averaged with the good habitat at both ends and
a crossing cannot be costly in itself.

### A. Segmented path features (established design; build checks below)
integrate_paths averages 20 shrink steps s = 1/20 ... 1 (s small = next to the destination, s = 1 = the origin). Keep the
SAME operator (same stored kernels, shrink, FFT, land normalisation at every step; the s64 input has no share floor, so
none is added) and average the steps in n_seg = 4 consecutive blocks of 5: segment 1 = s in .05-.25 (nearest the
destination), ..., segment 4 = s in .80-1.00 (the origin end). Z_disp_seg (time, N_land, K, 4, M); the mean of the 4
segments equals the stored Z_disp up to float32 FFT round-off (checked for EVERY year at build time; the build fails if
the median relative error exceeds 1e-5). Built from the input's own stored cell_integrated kernels and its stored Z (as
J2's W), written as a NEW file next to the input (z_disp_seg4_gathered.dat + json), never touching Z_disp.
Memory at 54 km (4,318 cells, K 12, M 64, 124 yr, float32): 6.58 GB, on top of Z_disp 1.64 + Z .14 = 8.4 GB of inputs
(the stored Z_disp is kept so the limit is the current model bit for bit). 3090 (24 GB, fraction .72 = 17.7 GB): fits
one fit (~3.2 GB working set at 54 km). A100 40 GB: NOT 3 per GPU at fraction .3 (12 GB < 8.4 + ~3.5); use
GRID54_MEM_FRACTION .45 with 2 fits per GPU. Gather: per-year jnp.take inside the checkpointed year scan (as Z_disp);
float32 on device under HOUFIN_Z_FLOAT32 (added to LARGE_INPUT_KEYS); host staging kept float32. Extra compute per year:
two (N, K, 4, M) . (M) dots -- negligible next to the FFTs.

### B. Units and form
Q is the ANNUAL survival of a juvenile that disperses (it replaces S_j for arrivals), ~.1-.6 on path habitat at the
step-3 point. Compounding an annual survival per 100 km is absurd (.55^8.6 ~ .006 on the outer band), and so is any
cost measured against a FIXED good-habitat reference: at the step-3 refined point S_j (= the coupled survival on local
habitat) is below its occupied-centre value (sigmoid(alpha_a) sigmoid(alpha_j) = .513) in 79-96% of land cells
(1950-2020) and is ~.001-.005 in much of the early East (the "Z darkens the East" defect, SDM_BASELINE_B1), so a
fixed-reference form puts a uniform, huge cost on every eastern arrival -- a level effect fecundity absorbs (J1's
failure mode), not a crossing effect. Numbers with a fixed reference, a 1,000 km crossing of the central Plains
(36-44N, 104-96W): factor .38 / .14 / .02 at lambda .025 / .05 / .1 in 1950 and .70 / .49 / .24 in 2000 (step 3); .63 /
.40 / .16 and .81 / .66 / .44 at the native-Z step 3 point. Rejected.
Proposed (BOTTLENECK form): keep the coupled annual term on the mean path habitat and compound a crossing factor for
the parts of the path that are WORSE than the path's own mean:
    log Q_k(p) = log Qc(H_bar_k(p)) + lambda * sum_seg (L_k / (n_seg l)) * min(0, log Qc(H_seg) - log Qc(H_bar_k)),
Qc = the CURRENT coupled survival function (ratio form: sigmoid(alpha_a + gamma_a H_s) sigmoid(alpha_j + gamma_j H_sj),
same weights, intercepts, per-cell drifts), H_bar = the stored path mean (exactly today's argument), H_seg = the segment
habitat, L_k = kernel k's mean distance (cell-centre, from the stored kernels: 86 / 277 / 856 km by band), l = 100 km
PINNED. Per 100 km of path, a segment whose survival is x-fold below the path average multiplies Q by x^-lambda.
Properties: (i) lambda = 0 -> exactly today's Q (factor exp(0) = 1); (ii) a homogeneous path costs nothing extra --
the level stays with the annual term, so this is a SHAPE (barrier) effect, not a level that fecundity can trade
one-for-one; (iii) tied to the juvenile-survival field (no free habitat weights); (iv) a long poor stretch between good
ends -- the Plains/Rockies between western and eastern habitat -- is what it costs; (v) costs only, never a bonus
(min(0, .)). Learned: ONE scalar lambda ~ HalfNormal(.25) per 100 km (pinnable). Pinned: l = 100 km, n_seg = 4, the
reference = the path's own mean. Scale (assumed; no literature magnitude for finch crossing mortality, Bonte 2012
establishes only that transfer costs exist): illustrative W->E transects at 38-42N with 4 segments of the recon S_j
fields (cells, not wedges, so the real H_seg is smoother and the factors milder):
| path (dest <- origin) | fit / year | segment S_j dest -> origin | lambda .1 / .25 / .5 |
|---|---|---|---|
| Plains, 95W <- 105W (~850 km) | step 3, 1950 | .042 .020 .008 .002 | .64 / .33 / .11 |
| same | step 3, 1970 | .162 .136 .103 .060 | .87 / .71 / .51 |
| same | step 3, 2000 | .202 .198 .206 .303 | .94 / .85 / .72 |
| Rockies + Plains, 100W <- 112W (~1,000 km) | step 3, 1970 | .059 .034 .094 .132 | .79 / .56 / .31 |
| same | native-Z step 3, 1970 | .200 .071 .100 .176 | .82 / .60 / .36 |
| Ohio valley, 82W <- 90W (~700 km) | step 3, 2000 | .572 .567 .647 .551 | .98 / .96 / .92 |
At the prior median (.17) a Rockies + Plains crossing keeps ~.7 of its annual survival; at the 95% bound (.49) ~.3;
homogeneous eastern paths ~1. Caveat (likely): the early-East S_j is so low (~.01 or below) that relative contrasts there
are large even where all habitat is poor; the min(0, .) keeps only the worse-than-average segments.

### C. Identifiability and guards (pre-registered with the build as GRID54 step 30)
J1 / J2's lesson: a planted cost is absorbed by fecundity and founders. The bottleneck form is a shape effect, which
should be harder to absorb, but that is a hypothesis (likely, untested). Guards, synthetic truth = step 3 refined point:
NULL lambda = 0 (pass lambda <= .05); RECOVERY lambda = .25 (pass [.125, .375] with fecundity / founders / w_scale in the
null's band, reporting where any shortfall went); MODEL-COMPARISON: the lambda = .25 data fitted by the CURRENT model
(step 3 config) -- if its refined log density and held-out score are within the null-vs-truth noise of the compounding
fit (|delta| < 10 nats, S-real within the seed floor), the data cannot see compounding and the real arm is uninformative
about it; CROSS-LEAK vs fecundity (truth F x .90, lambda = 0; pass lambda <= .05). Composes with the yearling-breeding
life cycle (step 29): it changes only Q, so the same switch applies on that base (a later arm), not tested here.

### D. Amendment (coordinator, before building): ENDPOINT reference
The path-mean reference under-penalises broad barriers: when 3 of 4 segments are poor, H_bar is itself poor, so the
poor segments sit near it and a long uniformly poor crossing costs nothing extra. Built form (default, `reference:
"endpoint"`):
    log Q_k = log Qc(H_bar_k) + lambda * sum_{seg in 2,3} (L_k / (4 l)) * min(0, log Qc(H_seg) - ref_k),
    ref_k = max(log Qc(H_seg1), log Qc(H_seg4))   (the destination-end and origin-end segments).
Why the interior segments only: the end segments are where the bird is reared and settles; their quality is already
in the annual term (and in S_j / the destination's vital rates), and costing the worse end against the better one would
penalise every move from good to poor habitat -- a destination-quality (level) effect, not a crossing. The interior
half of the path is the crossing. Properties kept: lambda = 0 exact; costs only; one learned scalar lambda ~
HalfNormal(.25) per 100 km; a path whose interior is no worse than its better end costs nothing; a crossing much worse
than where the bird came from now scales with its length. The path-mean form stays available as `reference: "mean"`
(all four segments against log Qc(H_bar)).
Both forms on the same cell transects (factors at lambda .1 / .25 / .5):
| path | fit, year | path-mean | endpoint |
|---|---|---|---|
| Plains 95W <- 105W | step 3, 1950 | .64 / .33 / .11 | .59 / .27 / .07 |
| same | step 3, 1970 | .87 / .71 / .51 | .88 / .72 / .52 |
| same | step 3, 2000 | .94 / .85 / .72 | .84 / .65 / .42 |
| same | native-Z, 1970 | .91 / .79 / .62 | .78 / .53 / .28 |
| same | native-Z, 2000 | .92 / .82 / .67 | .79 / .56 / .31 |
| Rockies + Plains 100W <- 112W | step 3, 1950 | .71 / .43 / .19 | .53 / .20 / .04 |
| same | step 3, 1970 | .79 / .56 / .31 | .65 / .34 / .11 |
| same | step 3, 2000 | .79 / .55 / .30 | .57 / .25 / .06 |
| same | native-Z, 1970 | .82 / .60 / .36 | .64 / .33 / .11 |
| Ohio valley 82W <- 90W | step 3, 1970 | .92 / .81 / .65 | 1.00 / .99 / .99 |
| same | step 3, 2000 | .98 / .96 / .92 | 1.00 / 1.00 / .99 |
| same | native-Z, 1950 | .82 / .61 / .37 | .82 / .62 / .38 |
Reading: the endpoint form makes the Rockies + Plains crossing markedly costlier (at the prior median .17 it keeps ~.5
of its annual survival vs ~.7 under the mean form) and leaves good eastern paths free; the path-mean form charged
heterogeneous but good eastern paths (Ohio 1970 .81 at lambda .25), which the endpoint form does not. The 1950 Ohio
transect (all poor, S_j ~ .001-.003, origin end the best) is costed in both -- the early-East darkening (B1) again.

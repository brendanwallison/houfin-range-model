# SDM and hybrid-SDM baselines for the House Finch age model -- PLAN (2026-10-30, not yet run)

Status: plan only. Nothing here has been fitted. Mechanism labels: ESTABLISHED (measured in our records or stated by
the source), LIKELY (my reading, untested), UNKNOWN.

Prompt: the user shared Zurell et al. 2016 (Global Change Biology, doi 10.1111/gcb.13251). Its four lessons, as the
user summarised them: dynamic models beat static SDMs under change; simple MigClim (SDM + colonisation) often did
as well as complex hybrids; how suitability is linked to demography matters a lot; calibrating demography needs
abundance time series. Added scope (user, 2026-10-30): source/sink in the validation suite must cover more than one
definition (e.g. Runge et al. 2006, C_r), and the plan must say which baselines can produce which source/sink measures.

Records this plan builds on (read before writing, per houfin-read-records-first):
- Age-model ladder and its readouts: research/experiments/GRID54_LADDER_2026-10-29.md (steps 0-4, front-speed
  diagnostic, step 10 gates T1/T2), GRID54_LADDER_2026-10-30_MISSPEC_SUITE.md, DESK_AND_DOWNSTREAM_SYNTHESIS_2026-10-26.md
  sec. 9 (fundamental lambda, 2021-25 holdout).
- Snapshot age-model records (WSL) ~/houfin/work/houfin/research/GRID54_LADDER_20261029/code/research/age_model/:
  experiments E063 (occupancy closure, realised lambda), E070 (holdout design, gates and ranking), E072 (route-level
  zero inflation, P(0)), E076 (likelihood-only comparison, residual structure by region/era/density); exp/
  simulate_observations.py (synthetic counts on the real design), exp/plot_decomp.py (log-density by years since first
  positive), exp/baseline_checks.py (H8/H9 calibration checks).
- Mask code: src/model/data_loading.py `holdout_observation_mask` (holdout_years; holdout_blocks incl. pinned_blocks
  with block_km, so masks nest across 27/54 km grids).
- Front products: scripts/build_colonization_front.py (first-detection raster, eastern expansion only, native hull and
  pseudo-zero block masked); research/exp/grid54_front_speed.py.
- DESK-side baselines (habitat representation, NOT the population model): DIGEST.md sec. "Does DESK add anything over its
  own inputs" (E039, E039b-i), CONSIDERATIONS M16 (covariate GP broken, A13; spacetime GP is degenerate damped
  persistence), E016/E028 (regional trend and damped persistence placebos), M6 (regional trend can be habitat).

---------------------------------------------------------------------------------------------------------------------

## STATUS (2026-10-30, later): step 0 (D1 design table, D2 scorer) and B0 DONE -- pre-registration and results in
research/experiments/SDM_BASELINE_B0_2026-10-30.md. Scorer parity gate passed (< 1e-13). Headlines: T conditional
beats observer-route persistence (+.063, CI .03-.10); T MARGINAL TIES cell persistence (-.008, CI -.04 to +.03; rule
R1 triggered: T marginal cannot rank ladder dynamics); S-real beats kernel interpolation (+.132, CI .04-.21), mostly
in the east (+.25). Regional-trend extrapolation is worse than flat persistence at 5 years.
Ladder rescored on S-real with research/exp/rescore_run.sh (GRID54_SREAL_RESCORE_2026-10-30.md). B1 DONE
(SDM_BASELINE_B1_2026-10-30.md): age model beats every Z-only SDM on S-real (+.165 vs B1-dyn, east 1966-93 +.39), but
a Z-only correlative fit's early-East expected counts are ~12x below modern (ratio .080): the equilibrium confound
(not-yet-arrived attributed to Z), not evidence about location quality (user correction, 2026-10-30).
Z attribution (GRID54_Z_EAST_ATTRIBUTION_2026-10-30.md): Z's eastern change is shared by non-invading sedentary common
species (out-of-band corr .89/.93); the finch fit's own dims (Z_15, Z_6, Z_18, Z_21) only partly (central-east .66,
east of 80W .25 with CI spanning 0). B2 (SDM_BASELINE_B2_2026-10-30.md): static-suitability colonisation reaches the
East 33 yr early with the wrong cohort shape; B2-dyn fits the front (-5) but its suitability already contains the
arrival confound (partly circular). Dispersal-only is ruled out in this class; habitat-gate vs Allee unresolved.
2026-10-31 (SDM_BASELINE_KISSMIG_B2ALLEE_2026-10-30.md, SDM_BASELINE_B6_RANGR_2026-10-31.md): project-local R library
(research/r/lib: kissmig 2.0.1, rangr 1.0.10, terra 1.9.50). kissmig reproduces B2's cohort-shape finding (PARTIAL
agreement). B2-Allee on static suitability: Allee used but not sufficient (-23.5, late cohorts still ~39 yr early).
B6 rangr static-habitat truth: a Z-only SDM invents a ~4x early deficit (.26) where arrival fell in the survey era --
the equilibrium confound reproduced outside our structure; S-real ranks the static model correctly, T marginal does
not. Age-model fit to that truth (GPU) is the decisive next test; B4 not started.

## 1. What already exists (no invention)

### 1a. Baselines we already have
| Baseline | Level | Where | What it showed |
|---|---|---|---|
| 19 WorldClim bioclim; bioclim + squared + 12 land-cover classes ("strongest SDM-style set", 50 vars) through the age model's own LINEAR readout | habitat field (static, per-species readout on BBS) | E039c/d/f, DIGEST | DESK 24 dims beat bioclim by +.06/+.16 level skill (5-20% / >=20% prevalence); beat the strongest SDM-style set by ~.03/.08 after matched smoothing (E039f). ESTABLISHED. |
| Top-k PCs of DESK's own covariates; bird-supervised linear projection | habitat field | E039b, E039g | DESK's edge is mostly WHICH directions it keeps; a supervised linear map nearly matches it spatially (.00-.03). ESTABLISHED. |
| No-change, spacetime GP (neighbours' observed abundance), damped persistence, regional-trend placebo, covariate GP | community change | E016, E028, E039h-i, M16 | Regional-trend placebo matches DESK in time; covariate GP is BROKEN (A13) and must not be quoted; spacetime_sum is degenerate damped persistence. ESTABLISHED. |
| Age-model arms against each other on the same mask | population model | E070, E072, E076, GRID54 ladder | Arms are only ever ranked against other age-model arms. |

What does NOT exist (grep of DIGEST, registry, experiments, CONSIDERATIONS for SDM, MigClim, occupancy model,
maxent, biomod, AUC, TSS, Brier: no hits on the population side): no House Finch SDM, no dispersal-limited SDM, no
dynamic occupancy model, no simpler process model, and no naive forecast (persistence / climatology) scored on the
age model's held-out sets. The age model has never been compared with anything outside its own family. That is the
gap this plan fills. Note the DESK-side SDM comparisons above answer a different question (is Z a good habitat
summary for many species?) than the one here (is the House Finch population model better than a correlative or
dispersal-limited range model of the House Finch?).

### 1b. Metrics we already use for the age model (the plan scores every baseline on these, unchanged)
- M-A Held-out mean NB log predictive density per BBS observation (higher is better), two sets:
  TEMPORAL = all observations in 2021-25 (5-year forecast); SPATIAL = pinned 216 km blocks (54 km: 4x4 blocks; 27 km:
  8x8; same block indices, `holdout_blocks.pinned_blocks`), all years. ~18% of observations withheld (GRID54 step 0).
  Since step 3 there are three versions: plug-in, MARGINAL (observer x route effect integrated, Gauss-Hermite 24; the
  comparable score) and CONDITIONAL (integrated only for observer-routes unseen in the fitted set; the forecaster's
  score). All spatial held-out routes are unseen, so S conditional = S marginal. Current reference (54 km step 3
  refined): T marginal -1.844, T conditional -1.731, S marginal -1.219; 27 km s64 refined T -1.804 / S -1.210 (pre-
  observer-effect, plug-in). Differences under ~.01 nats/obs read as no difference (one fit per step).
- M-B Fitted-set and held-out log1p RMSE and correlation (E070, E076; ladder tables "log1p corr").
- M-C Region x era breakdown of per-observation log density: west / plains / (midwest) / east; 1966-79, 80-94,
  95-09, 2010-25; central-east band 95-80W, 25-49N; east of 80W (E076, ladder).
- M-D Residual log(obs/pred) by observed-density quintile within region, cell-decade means, cells with >= 5 obs (E076
  probe 2; read against the correctly specified synthetic CONTROL 68b, spread .51 in the west).
- M-E Zero fraction: predicted vs observed P(y = 0) by region (E072: plains .85 vs .93).
- M-F Front timing (grid54_front_speed.py; GRID54 step 10 gate T1): 782 eastern cells (east of 95W, 25-49N) surveyed
  >= 3 years before their first BBS positive; model first-occupied year (density >= 1% and >= 10% of the 2010-25 median
  occupied density) minus first BBS positive; median overall and by observed-arrival cohort 1966-74 / 75-84 / 85-94 /
  95-09; share of eastern cells never reaching 10%. Observed first detections also as a raster from
  build_colonization_front.py (the west has no front: masked).
- M-G Log-density cost by years since first positive and years since disease arrival (plot_decomp.py --meta; E001/E002
  located the Allee cost at +-5 y of eastern arrival).
- M-H Source/sink readouts (age model only today): all-land / central-east / east-of-80W mean FUNDAMENTAL lambda
  (post-establishment, density-independent local eigenvalue; excludes dispersal, disease, density limitation and Allee)
  for 1902-15 / 1940-55 / 2010-25; suitable / source / sink shares by era; native-west fundamental-sink share; share of
  modern BBS positives in fundamental-sink cells; Allee-dead share of suitable habitat; occupied-conditional REALISED
  lambda (E063). Plausibility gates from E070: native west lambda_f >= 1 throughout, plains trough.
Stratification: the age model is single-species, so the residency / prevalence-tier rules (houfin-sedentary-primary,
houfin-report-prevalence-tiers) do not apply to these scores (GRID54 step 3 states this). The strata are held-out set x
region x era x arrival cohort. If a baseline is later run on proxy species (sec. 6), sedentary-common is the primary
stratum.

---------------------------------------------------------------------------------------------------------------------

## 2. Package and literature scan (2016-2026)

Status checked 2026-10-30 on CRAN / project pages; anything not checked is marked IMPRESSION.

| Tool | What it is | Status | Fits counts / time series? | Fit for our case |
|---|---|---|---|---|
| MigClim (Engler & Guisan 2009) | SDM suitability maps + cellular-automaton colonisation, LDD, maturity age | ARCHIVED on CRAN 2020-02-25 ("no maintainer"); mirror github.com/cran/MigClim; C code. ESTABLISHED (cran.r-project.org/package=MigClim) | No: simulates presence from given suitability; dispersal parameters set by hand or by calibration loops | Re-implement the rule in NumPy (tens of lines); do not depend on the package |
| kissmig (Nobis & Normand 2014) | Simple stochastic 3x3-neighbourhood migration on suitability rasters; first-occurrence output | On CRAN, v2.0-1 (2025-10-23 per CRAN mirror pages). ESTABLISHED | No (presence only); "first occurrence" map is exactly our M-F product | Cheap cross-check of our own implementation; R on Windows |
| demoniche (Nenzen et al. 2012) | SDM + stage-structured matrix model + dispersal | Not on CRAN (404); historically R-Forge (IMPRESSION) | Takes demography as given, does not estimate it | Not worth reviving; the age model is already a fitted superset |
| RangeShiftR / RangeShifter 2.0 (Malchow et al. 2021) | Individual-based demography + dispersal (incl. SMS movement) on dynamic landscapes | GitHub only (RangeShifter/RangeShiftR-pkg, v1.1.x; C++ build); old repo archived. ESTABLISHED | Simulator; calibration (ABC/pattern-oriented) is external. Individual-based: 10^6-10^7 birds x 124 years is slow (LIKELY) | Virtual-ecologist truth generator only; not a baseline |
| rangr (Markowska et al. 2024/25, MEE doi 10.1111/2041-210X.14475) | Grid simulator: Ricker/Gompertz/Beverton-Holt growth, carrying capacity from a suitability raster, Allee, dispersal kernels, virtual-ecologist sampling | On CRAN (v1.0.9, 2026-01-23; 1.0.10 on some mirrors); rOpenSci. ESTABLISHED | Simulator; no fitting | Best off-the-shelf generator of ALTERNATIVE-STRUCTURE synthetic truths (sec. 4 B6) |
| poems (Fordham et al. 2021) | Lattice population models + pattern-oriented ABC validation | CRAN / r-universe v1.4.x. ESTABLISHED | Calibrates to patterns by simulation (slow per run; LIKELY hours-days at our size) | Idea worth borrowing (pattern-oriented validation = our M-F/M-H); package not needed |
| steps (Visintin et al. 2020) | Spatially explicit stage-based population simulation | ARCHIVED on CRAN 2026-01-15 ("issues not addressed"). ESTABLISHED | Simulator | Skip |
| Pagel & Schurr 2012 DRM (GEB 21:293) | Hierarchical Bayesian: local Ricker growth r(env), K(env), dispersal, fitted to abundance/occupancy | No maintained package found (search 2026-10-30); original supplementary code (IMPRESSION) | Yes, built for this | Our age model is a richer DRM; a STRIPPED DRM is the right "simpler process model" baseline (B4) |
| drmr (Godoy, Fredston, ..., Pinsky; EcoEvoRxiv preprint v2, 2026-04-15, not reviewed) | Bayesian age-structured DRMs, env effects on recruitment and survival, forecasting; Stan backend (IMPRESSION -- not confirmed) | Preprint + github.com/pinskylab/drmr_paper; no CRAN release found. ESTABLISHED that it exists | Yes (designed for spatio-temporal survey data) | Closest external analogue; built for marine patch series (tens-hundreds of patches; IMPRESSION). Worth a read for structure and metrics; unlikely to scale to 4,000 cells x 124 y with dispersal without work |
| Fredston et al. 2025 (preprint, summer flounder) | DRM out-of-sample range-edge/centroid forecasts beat a statistical SDM and persistence over a decade | Preprint | -- | Supplies a precedent metric: range centroid / edge position error (sec. 5, new metric N3) |
| Hmsc | Joint SDM (multi-species, latent factors, spatial) | CRAN v3.3-7 (2025-08-28). ESTABLISHED | Counts yes (Poisson/lognormal-Poisson), static or with temporal random levels; no dispersal | Not needed single-species; DESK already plays its role on the habitat side |
| spOccupancy (stPGOcc, multi-season spatial occupancy) / spAbundance | Bayesian occupancy / abundance with NNGP spatial and AR(1) temporal random effects | CRAN (spOccupancy 0.8.x). ESTABLISHED | Detection/non-detection with replicates; spAbundance for counts. Temporal trends, NOT colonisation from neighbours (no spread mechanism) (LIKELY from the model forms) | Optional statistical comparator; sdmTMB covers the same ground with NB counts directly |
| unmarked colext / pcountOpen | Dynamic occupancy (colonisation/extinction) and open N-mixture | CRAN v1.5.2 (2026-08-20). ESTABLISHED | Needs within-season replicates (BBS has 50 stops; route totals only in our pipeline); colonisation is covariate-driven, not neighbour-driven | Poor match without stop-level data |
| flocker (Socolar & Mills 2023) | brms/Stan occupancy incl. multi-season colex (HMM) | CRAN 1.0-0; r-universe 1.0-2. ESTABLISHED | Same replicate need; Stan at 4,000 sites x 60 years is slow (LIKELY) | Skip |
| Bled, Royle & Cam 2011 (Ecol. Appl. 21:290) | Dynamic occupancy of an invader (Eurasian Collared-Dove) on BBS with distance-dependent colonisation | Paper (BUGS-era code) | Built on BBS | The right TEMPLATE for B3 (a dispersal-aware occupancy baseline on BBS); implement ourselves |
| Hooten & Wikle 2008; Hooten et al. 2007 (Biometrics 63) | Reaction-diffusion / matrix-model invasion inside hierarchical Bayes, Collared-Dove BBS | Papers | Yes | Historical precedent: same class as our model; no baseline value beyond B4 |
| sdmTMB (Anderson et al. 2025, JSS 115(2)) | Spatial / spatiotemporal GLMMs (SPDE, TMB): NB2, delta-NB, AR1 spatiotemporal fields, time-varying coefficients | CRAN v1.1.0 (1.0.0 Jan 2026). ESTABLISHED | Yes: route counts, covariates, spatiotemporal field, forecasting | Strongest statistical forecaster we can run cheaply (B5) |
| biomod2 | Ensemble SDM platform | CRAN 4.3-4-x (2025-12 to 2026). ESTABLISHED | Presence/absence (or abundance in recent versions -- IMPRESSION, not checked) | Not needed: one well-specified NB GLM/GAM is a stronger, likelihood-comparable SDM than an ensemble of presence models scored by AUC |
| ENMeval | Maxent tuning / spatial-block evaluation | CRAN v2.0.6 (2026-09-11). ESTABLISHED | Presence-only | Not applicable (we have counts and absences) |
| dynamicSDM (2023, MEE) | Tools for spatio-temporally matched covariates in SDMs | CRAN. ESTABLISHED (exists) | -- | We already have time-matched Z; skip |
| Briscoe et al. 2021 (GCB doi 10.1111/gcb.15723) | Dynamic occupancy vs SDMs, 69 Swiss birds | Paper | -- | Lesson: flexible static SDMs won on discrimination; DOMs helped on trends with enough data; nobody predicted WHERE change happens. Sets expectations for B1 vs B3 |

Reading of the scan (LIKELY): there is no maintained package that fits a dispersal-aware demographic range model to
~150k route counts on 4,000-17,000 cells over 124 years; the field's packages are either correlative fitters (sdmTMB,
spOccupancy, Hmsc, biomod2), presence simulators (MigClim, kissmig) or demographic simulators (rangr, RangeShiftR,
poems). So the baselines below are mostly short re-implementations on our own data tables and masks, with packages used
as cross-checks or truth generators. That is also what makes the comparison fair: one design table, one scorer.

Sources: cran.r-project.org/package=MigClim; github.com/cran/MigClim; cran.r-project.org/package=kissmig;
github.com/RangeShifter/RangeShiftR-pkg; cran.r-project.org/package=rangr; docs.ropensci.org/rangr;
cran.r-project.org/package=poems; cran.r-project.org/package=steps; ecoevorxiv.org/repository/view/12564 (drmr);
cran.r-project.org/package=Hmsc; cran.r-project.org/package=spOccupancy; cran.r-project.org/package=unmarked;
cran.r-project.org/package=flocker; cran.r-project.org/package=sdmTMB; cran.r-project.org/src/contrib/Archive/biomod2;
cran.r-project.org/package=ENMeval; pubs.usgs.gov/publication/70034098 (Bled et al. 2011);
pubs.usgs.gov/publication/70029701 (Hooten et al. 2007); doi 10.1111/gcb.15723 (Briscoe et al. 2021);
usgs.gov/publications/role-local-populations-within-a-landscape-context-defining-and-classifying-sources-and (Runge et
al. 2006, Am Nat 167:925, doi 10.1086/503531); pubs.usgs.gov/publication/70206024 (Diffendorfer et al. 2019 C-metric).

---------------------------------------------------------------------------------------------------------------------

## 3. Shared infrastructure (build once; every baseline uses it)

D1. BASELINE DESIGN TABLE (one Parquet/NPZ per grid, 54 km first, 27 km second), exported in WSL by a small script
    that calls `src.model.data_loading.load_data` with the SAME resolved config as the GRID54 step-3 run and writes, per
    observation: count, year, cell row/col, lon/lat, route id, observer-route id, quality tier, obs_weight,
    is_pseudo_zero, obs_holdout (temporal / spatial flags separately, from `holdout_observation_mask`), region and
    arrival-cohort labels; per cell-year: the Z dimensions the age model reads (record which: block-averaged kernel mean
    embedding at 54 km), Z_disp if used, land mask. Hash it (CODE/DATA manifest) so every baseline cites the same table.
    No refit of anything; minutes on CPU.
D2. ONE SCORER (Python): given per-observation predictive mean mu and NB concentration (or a full predictive pmf for
    ZINB/occupancy baselines), and optionally an observer-route effect variance, returns M-A (plug-in, marginal via the
    same Gauss-Hermite 24 integration, conditional), M-B, M-C, M-D, M-E, and Brier/AUC for P(y > 0) (new, N1). It must
    reproduce the age model's step-3 numbers from its saved per-observation mu to < .001 nats/obs before it scores any
    baseline (unit test: houfin-test-agreed-design-end-to-end -- both the fit side and the scoring side exercised).
D3. ONE FRONT READER: M-F from any model's yearly expected-density (or occupancy) cube on the grid, reusing
    grid54_front_speed.py's cell set and thresholds; for presence-only baselines, "first occupied" = first year the
    cell is colonised / P(occ) >= .5 (stated per baseline; thresholds are not interchangeable -- see risks).
D4. OBSERVATION-MODEL PARITY: every count baseline uses NB2 with an observer x route random effect (sigma_u estimated
    on the training rows, integrated for unseen observer-routes exactly as the age model) and the same pseudo-zero and
    obs_weight handling. Pseudo-zeros (1902-39) are TRAINING data only where the baseline has a time axis that reaches
    them; they are never scored (they are not observations).

Where it runs (feasibility, local only): WSL has Python 3.12 + JAX, 32 cores, 176 GB RAM; R 4.5.1 exists on Windows
(C:\Program Files\R\R-4.5.1) but not in WSL. R packages (sdmTMB, kissmig, rangr) go into a project-local user library
(e.g. renv or R_LIBS_USER under research/), never system-wide. CONSTRAINT FROM OUR RECORDS: GRID54 "Run environment"
notes CPU-heavy jobs coinciding with GPU-fit segfaults; baselines therefore run single-threaded-capped (OMP/MKL threads
<= 8) through pueue at low priority, or when no local GPU fit is running. No TACC.

---------------------------------------------------------------------------------------------------------------------

## 4. The baselines, in order

Order rule: cheapest first, each one answers a question the next one depends on. B0-B2 are the core (days, CPU only);
B3-B5 follow if B0-B2 leave a question open; B6 runs alongside from B2 on.

### B0. Naive forecasts (persistence / climatology / neighbour interpolation) -- MINIMAL FIRST STEP
(a) Question: is the age model's held-out skill more than "routes keep their recent counts" (temporal) and "a block
    looks like the nearest observed routes" (spatial)? Without this floor no held-out number in the ladder has a scale
    (the DESK suite learned this the hard way: validation_report.py header, "a dir-cos of .48 ... lost to plain IDW").
(b) Inputs: D1 table only. Temporal: per observer-route, mean log1p count over the last k training years (k = 3, 5,
    10; k chosen on 2016-20 as an inner validation), NB concentration fitted on training residuals; fallback to route,
    then cell, then 300 km regional mean for routes new in 2021-25. Spatial: inverse-distance / kernel average of
    training-route means by year (bandwidth chosen by leave-one-block-out on TRAINING blocks), NB concentration fitted
    the same way. Also a "regional trend" variant (each 300 km region's 2011-20 log-linear trend extrapolated) because
    E016/E039i found regional trend extrapolation to be the strongest placebo on the DESK side.
(c) Metrics: M-A (plug-in and marginal), M-B, M-C, M-E. No front, no source/sink (B0 has no dynamics).
(d) Runtime: seconds to minutes, NumPy. One day including the scorer (D2) and its parity test.
(e) Risks: persistence uses observer-route memory, so compare it with the age model's CONDITIONAL temporal score, and
    its route-free version with the MARGINAL score; mixing them is an apples/oranges error (GRID54 step 3 lesson). The
    spatial IDW has no habitat: if it ties the age model on S, the 216 km blocks are interpolable from neighbours and S
    is a weak test of habitat (that is a finding about the test, not the model).
Decision informed: whether T and S as currently defined can rank arms at all. If persistence beats every ladder arm on
T conditional, the 5-year forecast is dominated by route memory and lambda-related changes (amplitudes, noise forms)
must be judged on S, M-F and M-H rather than T. Implication for production (reported, not decided): the score used for
selection in GRID54 steps 4-10 may need re-weighting.

### B1. Correlative SDM on the same habitat field (static and time-varying)
(a) Question (core): is the age model's skill just correlative niche? Specifically (i) does a count SDM on the same Z
    predict held-out routes as well as the process model; (ii) where does it fail -- expected (LIKELY): the eastern
    front and the post-1994 disease decline, which an equilibrium SDM cannot represent; (iii) does an SDM, trained with
    the pre-invasion eastern absences, learn "the east is poor habitat" -- the same confound the age model shows as the
    early-east fundamental sink (GRID54 step 1-3: east of 80W lambda .08-.36 before 1955; front-speed diagnostic). If
    the SDM reproduces that "early east bad" map, the inflation is LIKELY the generic equilibrium-assumption failure
    (absence from dispersal limitation read as habitat), not something peculiar to the age model's structure (G3/D3).
(b) Inputs: D1 table; NB2 GLM/GAM with log link on the age model's Z dims (same dims, same year-matched values),
    quadratic terms (hump-shaped niches; E039c found they matter), observer x route random effect; years 1966-2020
    training rows outside pinned blocks. Arms:
    B1-static: Z averaged over 1966-2025 per cell (classic SDM on climate-normal-like habitat).
    B1-dyn: year-matched Z (time-varying SDM; habitat change enters through Z, no dynamics).
    B1-native: trained on the native west only (classic invasive-species SDM, Zurell/Elith practice), projected east.
    B1-full+yr: B1-dyn plus a smooth year x region term (the best a statistical model does with a trend; LIKELY strong
    on T, uninterpretable as habitat -- regional trend is not dynamics, M6).
    Optional B1-bioclim: the E039c "strongest SDM-style set" (bioclim + squared + land cover) instead of Z, only if those
    covariates already exist at the grid (else skip; do not build a new covariate stack for this).
(c) Metrics: M-A..M-E; M-F by thresholding expected density at the same 1% / 10% of 2010-25 median occupied density
    (for B1-dyn the "front" is where Z-driven suitability crosses the threshold -- tells us how much front timing the
    habitat field alone implies); suitability maps by era compared with the age model's fundamental-lambda maps (rank
    correlation by era, central-east band), as a structural comparison, not a source/sink measure.
(d) Runtime: glmmTMB / sdmTMB (no spatial field) in R or a numpyro/statsmodels GLM in Python: ~150-190k rows x ~50
    columns, minutes per arm. Python preferred so the scorer and table never cross a language boundary.
(e) Risks: an SDM fitted on BBS counts 1966-2025 sees the eastern population AFTER arrival, so B1-dyn may score well on
    T and S simply because 2021-25 and the blocks are mostly inside the equilibrium range; its failure will be
    concentrated in the 1966-94 east (M-C era x region is mandatory, not optional). Overfitting Z quadratics: penalise
    (ridge) and pick the penalty by inner block CV on training blocks only. Zurell's structural lesson applies: the SDM's
    suitability is a density surface, not a vital rate.
Decision informed: if B1-dyn matches the age model on T/S marginal within .01-.02 nats/obs, held-out count fit does not
discriminate process structure, and the ladder's process claims must rest on M-F, M-H and synthetic recovery (B6). If
the age model wins mainly in east x 1966-94 and post-disease cells, that is where its process structure earns its keep.

### B2. Dispersal-limited SDM (MigClim / KISSMig class) -- the Zurell "simple hybrid"
(a) Question: does SDM suitability + colonisation (release 1940 on Long Island + native western range in 1902) explain
    the eastern front timing (M-F) as well as the age model, with no demography, no Allee, no habitat change needed to
    gate the front? This bears directly on the open puzzle: the age model holds the front back by making the early east
    a deep fundamental sink (GRID54 front-speed diagnostic, ESTABLISHED for step 3: with habitat frozen the release fills
    the East 26-30 years early). If a colonisation automaton with a fitted spread rate reproduces the observed arrival
    cohorts on near-static suitability, the front timing does not by itself require early-east habitat change; the age
    model's issue is then its spread/establishment structure (Allee, LDD, settlement) rather than habitat (LIKELY reading
    to be tested, not assumed). If it cannot (e.g. the 1966-74 slow phase and 1975-94 acceleration need a sigmoid the
    automaton cannot give), that is evidence for density-dependent establishment (Veit & Lewis 1996 Allee + LDD, E051).
(b) Inputs: suitability from B1-static (primary) and B1-dyn (secondary), both trained on data that exclude the held-out
    rows; seeds: native-range cells occupied in 1902 (the age model's native seed / hull), release cells 1940 (the same
    4 cells at 54 km, 1 at 27 km). Rules (NumPy, re-implementing MigClim): yearly colonisation probability of an empty
    cell = 1 - prod over occupied sources (1 - p_disp(d) * s_target), p_disp a negative-exponential + fat-tail mixture
    (2-3 parameters), optional maturity lag (1 year), optional extinction where suitability < threshold. Abundance =
    occupancy probability x B1 expected count. Parameters (3-5) fitted by maximising the TRAINING NB likelihood of the
    counts (grid search / Nelder-Mead over a few hundred forward runs). Cross-check: kissmig (R) on binarised suitability
    with matched step count, for M-F only.
(c) Metrics: M-F first (front median by cohort, share never reached), M-A..M-E, M-G (log-density by years since first
    positive -- does the automaton put the cost where the age model does?).
(d) Runtime: 4,000 cells x 124 years is ~0.5M cell-years; with an FFT kernel a forward run is milliseconds in NumPy/JAX
    on CPU; a few thousand runs for fitting = minutes. Trivial at 27 km too (17k cells).
(e) Risks: (1) presence vs density thresholds -- M-F for B2 uses P(occ) >= .5 while the age model uses density
    fractions; report both definitions for the age model too so the comparison is like for like. (2) A free spread rate
    can fit front timing by construction; the fair test is that ONE parameter set must fit all four arrival cohorts and
    the plains crossing, scored on held-out S. (3) Feeding B2 the B1-dyn suitability imports the "early east bad"
    confound if B1 learned it -- run B2 on B1-native suitability too.
Decision informed: whether the front-gating that inflates lambda change (+.58 to +.76 refined) is a property of the
data (any adequate model needs early-east unsuitability) or of the age model's spread structure. Production implication
(reported): if B2 fits the front on static suitability, the ladder's next lever is establishment/dispersal (Allee
profile step 7, S-front / S-disp forms in the misspecification screen), not habitat amplitude.

### B3. Dispersal-aware dynamic occupancy on BBS (Bled et al. 2011 template)
(a) Question: with an honest observation model for zeros (route-level detection) and neighbour-driven colonisation,
    does an occupancy-dynamics model capture range expansion and retreat (west sinks, post-disease losses) better than
    the age model's abundance dynamics on presence data? It separates "where" (occupancy) from "how many".
(b) Inputs: D1 table reduced to route detection (y > 0); cell-level latent occupancy z_{c,t}; colonisation logit =
    a + b*Z + c*log(sum_j z_{j,t-1} K(d_cj)); extinction logit = e + f*Z (+ disease-era term); detection per route as a
    function of the age model's expected count is NOT allowed (would import the age model) -- use p = 1 - exp(-m * dens)
    with a free scale, or a constant per region. Fitted in JAX (forward filtering over cells is not exact with neighbour
    dependence; use the mean-field / expected-occupancy approximation, as E063 does) -- or a smaller exact version on
    54 km.
(c) Metrics: M-A for presence (Bernoulli log density of y > 0 on held-out rows, and Brier/AUC, N1), M-F (P(occ) >= .5),
    M-E, plus colonisation/extinction maps by era. Count metrics only via B1 conditional on occupancy (a hurdle).
(d) Runtime: comparable to B2 per forward run; gradient-based fit in JAX on CPU, minutes to an hour.
(e) Risks: largest build cost of B0-B3 (no package fits neighbour-dependent colonisation at this size; spOccupancy's
    spatio-temporal models have spatial and AR(1) random effects but no spread term -- LIKELY from its documented forms).
    Briscoe et al. 2021 found dynamic occupancy models did not beat flexible SDMs on discrimination; expect the same on
    T/S and judge B3 on M-F and on occupancy-scale source/sink (sec. 5). Skip if B2 already answers the front question.

### B4. Stripped dynamic range model (Pagel & Schurr / LoLiPop class) -- the structural-uncertainty ladder
(a) Question (Zurell's lesson 3): how much do the suitability-to-demography link and the demographic structure change
    held-out fit, front timing and the source/sink maps? Our age model fixes one structure (Z -> survival, fecundity,
    capacity, juvenile survival, adult/juvenile classes, path survival, settlement, relative Allee). B4 asks whether a
    much simpler process model already gets there, and whether the inferred lambda change is structure-dependent.
(b) Inputs: D1 table; unstructured Ricker (or Beverton-Holt with optional Allee, as LoLiPop) with dispersal by one
    kernel; arms by link: B4-r (Z -> r only), B4-K (Z -> K only), B4-rK (both), each with and without Allee. Same NB2 +
    observer effect; same release and native seed; same pseudo-zeros. Implement as a small JAX forward (FFT convolution)
    rather than as config switches of the age model, so it is independent code (a real comparator, not an ablation of
    the same code path) -- but test that B4-rK with matched parameters reproduces an age-model forward within tolerance
    on a toy grid (end-to-end parity, houfin-test-agreed-design-end-to-end).
(c) Metrics: all of M-A..M-H, and the full source/sink suite (sec. 5): B4 is the first baseline that has a lambda.
(d) Runtime: one MAP fit ~ minutes-hours on CPU (5-10 parameters + Z coefficients; 124 years x 4,000 cells); GPU only
    when the local GPU is idle. drmr (if its code is released and runs) is a cross-check on a coarse subset only.
(e) Risks: B4 inherits the same identifiability problems the age model has (deterministic skeleton over a century,
    Wood 2010; E070) -- so expect the same refine contortions; read MAP and refined. Do not reuse any calibration from
    another run (houfin-premise-changing-habitat). Static habitat arms are diagnostics only.
Decision informed: if lambda change and the early-east sink appear in every link structure, the inflation is a data x
dynamic-model property (LIKELY then the observation/spread side); if only some links show it, the link choice is a
structural lever worth a ladder step (production implication, reported).

### B5. Spatiotemporal statistical abundance model (sdmTMB) -- the flexible-statistics ceiling
(a) Question: what is the best held-out count fit available WITHOUT any process (a statistical ceiling on T and S)? A
    gap between B5 and the age model is the price of mechanistic structure; a model can be mechanistically right and
    lose to B5 on forecasts.
(b) Inputs: D1 table exported to R (Windows R 4.5.1, project-local library); NB2 (or delta-truncated NB2), Z dims +
    quadratics, observer-route random intercept, spatial + AR1 spatiotemporal SPDE field on a ~300-600 knot mesh,
    years 1966-2020; forecast 2021-25 from the AR1; spatial blocks predicted from the field.
(c) Metrics: M-A..M-E. NOT M-F (an AR1 field starting in 1966 has no front from 1940) and NOT lambda; it does yield
    apparent growth N_{t+1}/N_t of the smoothed field (sec. 5, measure S6).
(d) Runtime: ~150k rows, 60 time slices, 500 knots: LIKELY tens of minutes to a few hours per fit single-threaded.
(e) Risks: observer random effect + spatiotemporal field may trade off; spatial blocks are 216 km so the field must
    extrapolate -- its block score tests interpolation of the field, which is exactly what the age model's habitat is
    meant to beat. The field can absorb front dynamics as "trend", which is fine for a ceiling but must never be read as
    habitat (M6).

### B6. Virtual-ecologist benchmark (runs alongside B2 onward)
(a) Question: on truths where we KNOW the source/sink structure and front mechanism, do our held-out metrics rank
    the true structure first, and which baselines recover which quantities? Zurell et al. built their benchmark on
    virtual species; we already have the tool on the real design.
(b) Truths (all on the real 54 km design, NB2 counts via research/age_model/exp/simulate_observations.py, real holdout
    masks): T1 = GRID54 step-3 refined point (lambda inflation present); T2 = a step-3 variant with flat early-east
    habitat and a strong relative Allee (front slowed by establishment, not habitat); T3 = an alternative-structure
    truth generated outside our code (rangr: Ricker + suitability-driven K + Allee + kernel on our suitability raster;
    sampled at our BBS route-years) -- the out-of-family truth that guards against scoring our own structure as right
    because it generated the data. Each truth stores its fundamental lambda, realised lambda and C_r (sec. 5).
(c) Metrics: same as on real data, plus recovery: rank correlation of each model's source/sink map with the truth's,
    front-timing error vs the truth's front, and lambda-change recovery.
(d) Runtime: B0-B2 and B4 refits per truth are cheap (CPU); age-model refits on synthetic data are GPU fits that queue
    behind the ladder (already in the misspecification suite plan, section A) -- reuse those truths rather than adding.
(e) Risks: T1/T2 favour the age model's family; T3 is essential. rangr is on CRAN but its fit to our grid size and
    route sampling is untested (UNKNOWN).
Decision informed: whether T, S, M-F can tell T1 from T2 at all. If held-out count scores cannot separate "habitat gates
the front" from "Allee gates the front" on synthetic truths, then no amount of real-data held-out score can settle the
lambda-inflation question, and the evidence must come from M-F shape, M-H plausibility and external data (CBC,
historical edges -- M29).

---------------------------------------------------------------------------------------------------------------------

## 5. Source/sink validation suite: several definitions, and which models can produce them

Why several (user, 2026-10-30): "sink" has competing definitions that classify the same cell differently, and the
choice changes conclusions (Runge et al. 2006 show that criteria using apparent survival treat emigration as death,
which mislabels exporting cells as sinks). The age model today reports only fundamental lambda (S1) and, from E063,
an occupied-conditional realised lambda (S2). Definitions to compute (per cell, per era; 1902-15 / 1940-55 / 1966-79 /
1980-94 / 1995-2009 / 2010-25):

- S1 Fundamental lambda (EXISTING): density-independent local eigenvalue of the adult/juvenile matrix, no dispersal,
  disease, density or Allee. Sink = lambda_f < 1 (a habitat-quality sink, Pulliam 1988 in spirit).
- S2 Realised local lambda (EXISTING in E063 form): local births + survival at the realised density (density
  dependence, Allee, disease included), movement excluded. Sink = realised < 1 in occupied cells.
- S3 BIDE classification (Pulliam 1988): source if B > D and E > I over the era; sink if B < D and I > E. Needs the four
  flows per cell-year: births, deaths (incl. journey mortality attributed to the origin), emigrants, immigrants. The
  age-model forward computes all four internally (reproduction, survival, departure, kernel with path survival,
  settlement) -- they need to be exposed as outputs, not re-estimated.
- S4 Runge et al. 2006 contribution C_r (IMPLEMENTED 2026-11-01 in contribution_budget.py/run_contribution_audit.py; see CONTRIBUTION_RESULTS_20261101.md): per-capita contribution of cell r to the whole population one year later
  = (local survivors + local recruits that stay + emigrants that SURVIVE the journey and SETTLE elsewhere) / N_r.
  Source if C_r > 1. Also its local part R_r (self-recruitment) and export C_r - R_r. With adult/juvenile classes the
  contributions must be weighted to a common currency; Runge et al. root C_r in matrix-model theory -- whether they
  weight by stage reproductive value was initially unverified. VERIFIED 2026-11-01: the paper's "Multiple Stage or Age Classes" section specifies weighting stage-specific contributions by current stage distributions for short-term dynamics, or stable stage distributions for asymptotic dynamics; it does not require arbitrary reproductive-value weights. The implementation supplies the actual-stage-weighted postbirth metric and a separate breeder-to-breeder census metric. Computation in
  our model: C_r is a column sum of the one-year spatial projection operator M_t at the realised state; all column sums
  come from ONE transposed (adjoint) pass, 1^T M_t, which for an FFT kernel is the flipped-kernel convolution of the
  settlement-weighted survival field -- about the cost of one forward year (LIKELY; to be confirmed in code). Because
  departure is negatively density-dependent and settlement follows K (E008), M_t depends on N_t: C_r is state-dependent
  and must be reported as realised per-capita flows (exact) rather than a linearisation. Diffendorfer et al. 2019
  generalise C to multiple classes, seasons and density dependence (pubs.usgs.gov/publication/70206024) -- the
  framework that fits our model.
- S5 Network / long-run value (OPTIONAL, NEW): the left dominant eigenvector of M_t (reproductive value of location) or
  multi-year C (contribution over k years) -- separates cells that export now from cells whose descendants persist.
  Only if S4 is unstable year to year.
- S6 Apparent growth (observational, NEW for baselines): N_{t+1}/N_t of the fitted abundance surface (immigration
  included). Not a source/sink measure; a net-change measure any spatiotemporal model produces. Listed so nobody reads
  it as one.

Capability table (what each model can honestly produce):
| Model | S1 fundamental | S2 realised local | S3 BIDE | S4 C_r | S5 | S6 | Notes |
|---|---|---|---|---|---|---|---|
| Age model (current) | yes (exists) | yes (E063) | yes once flows exposed | yes (adjoint pass) | yes | yes | Only model with stage structure + journey survival, so the only one where C_r differs meaningfully from S2 |
| B0 naive | no | no | no | no | no | trend only | no dynamics |
| B1 SDM (static/dyn) | no | no | no | no | no | B1-dyn: change in expected count, purely from Z | suitability is a density, not a rate; can be compared with S1 maps only as a correlate |
| B2 MigClim-type | no | no | no | colonisation-source share only (fraction of new colonisations seeded from each cell; an occupancy analogue of export, NEW, not C_r) | no | no | presence-only |
| B3 dynamic occupancy | no | occupancy analogue: 1 - extinction prob. | no | occupancy analogue of contribution: colonisations caused per occupied cell (rescue / export); not demographic | no | P(occ) change | answers "occupancy sink" = persists only via recolonisation (rescue effect) |
| B4 stripped DRM | yes (r) | yes | yes | yes (unstructured: no stage weighting needed) | yes | yes | best comparator for S4 because the definitions are computed identically in a different structure |
| B5 sdmTMB | no | no | no | no | no | yes | statistical only |
| B6 synthetic truths | yes | yes | yes | yes | yes | yes | truth for recovery tests of every column |

Validation use: (i) agreement between S1/S2/S3/S4 within the age model (where do "habitat sinks" export? The native
west is 46-55% fundamental sink in steps 2-4 while holding many modern BBS positives -- S4 tests whether those cells are
net contributors once successful emigrants count; LIKELY many are, which would reframe the "west sink pathology");
(ii) structural robustness: S1-S4 maps from B4 links vs the age model; (iii) recovery on B6 truths. Implementation note:
S3/S4 need flows that exist inside the age-model forward; exposing them is a read-only diagnostic addition in the
snapshot lineage code (scripts/viz/map_diagnostics.py family), not a model change.

---------------------------------------------------------------------------------------------------------------------

## 6. New metrics proposed (only where the existing ones cannot answer)
- N1 Occupancy scale on held-out rows: Brier score of P(y > 0) (proper) and AUC (secondary, improper; included only
  because the SDM literature, including Zurell et al., reports discrimination). Derived from each model's own
  predictive (NB / ZINB / occupancy), so the age model gets it too. Needed because B2/B3 do not produce counts.
- N2 Source/sink definitions S3-S5 (sec. 5) and their between-definition agreement (share of cells classified the same).
- N3 Range position error (Fredston et al. DRM-forecast precedent): yearly eastern-front position (e.g. 10th percentile
  longitude of occupied cells per latitude band) and occupied-range centroid vs the BBS-observed equivalents. Only if
  M-F's cell-wise arrival differences prove too noisy; otherwise skip.
- N4 Synthetic recovery scores (B6): rank correlation of source/sink maps with truth, front error vs truth.
Proxy species (optional, later): running B1/B2 on sedentary-common proxy species would test whether the same structures
win across species; that reopens the sedentary-primary rule and is out of scope for this plan.

---------------------------------------------------------------------------------------------------------------------

## 7. Sequence, cost, and what each result decides

| Step | Work | Local cost | Reads against | Decision (implication reported, not decided) |
|---|---|---|---|---|
| 0 | D1 design table + D2 scorer with parity test against step-3 per-observation mu | ~1 day, CPU minutes | step 3 T/S marginal & conditional to < .001 | gate for everything after |
| 1 | B0 naive forecasts (MINIMAL FIRST STEP) | hours | step 3, step 4 arms | can T/S rank arms; is T dominated by route memory |
| 2 | B1-static / dyn / native (+ full+yr) | ~1 day | step 3 | is held-out skill just niche; does an SDM also learn "early east bad" |
| 3 | B2 MigClim-type on B1 suitabilities (+ kissmig cross-check) | ~1-2 days | step 3 M-F, front-speed diagnostic | is front timing explainable without habitat change; points ladder at spread/Allee vs habitat |
| 4 | Source/sink S3/S4 exposed in age-model diagnostics; computed on step 3 refined | ~1-2 days code, minutes per fit | S1 west-sink audit | does C_r reframe the west-sink and early-east-sink readings |
| 5 | B4 stripped DRM link ladder | ~3-5 days | step 3, S1-S4 | is lambda inflation structure-specific |
| 6 | B6 truths T1-T3 with B0-B2, B4 refits (age-model synthetic fits from the misspecification suite) | ~2-3 days CPU + queued GPU | truths | can our metrics separate habitat-gating from Allee-gating |
| 7 | B5 sdmTMB ceiling; B3 dynamic occupancy only if B2 leaves the front question open | 1-3 days each | all | price of process structure on forecasts |

Pre-registration: each step gets its own pre-registered prediction and rule in this file (or a child record) before
running, per PROTOCOL / houfin-run-planned-work, citing CONSIDERATIONS M1 (report correlation, scaling and skill
separately), M6 (regional trend is not dynamics), M16 (never quote the covariate GP), M28 (one yardstick for every arm:
the D2 scorer and the D1 table ARE that yardstick), G3/G4 (division of labour between habitat and dynamics) and the
premise that habitat change must stay learnable (static-habitat arms B1-static and the front-speed freeze are
diagnostics only).

## 8. Top risks across the plan
1. Unequal information: persistence and B5 use observer-route memory; compare like with like (conditional vs marginal)
   or the comparison is wrong (ESTABLISHED as a trap in GRID54 step 3).
2. Thresholds: presence-based front metrics vs density-based ones are not interchangeable; report both for the age
   model.
3. Self-confirmation in synthetic tests: truths drawn from our own model favour our structure (B6 T3 guards this).
4. Equilibrium confound in every SDM fitted on invasion-era data (B1, B2 suitability): train-on-native arms are needed
   to read it.
5. Machine stability: CPU-heavy work alongside local GPU fits coincided with crashes (GRID54 run environment notes);
   cap threads, queue at low priority.
6. Scope creep into "improve the baseline until it wins": baselines get the same budget as one ladder step each and
   are frozen after their pre-registered fit.

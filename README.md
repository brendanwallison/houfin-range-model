# houfin-range-model

## Why this project exists

When a species is introduced somewhere it has never lived — a house finch released in New York City in the 1940s, say — its subsequent spread across a continent is not random. It is shaped by climate, by the availability of suitable habitat, by how far individuals of different ages can disperse, by density-dependent competition for space, and by the basic demographic arithmetic of births and deaths along the way. This codebase is an attempt to reconstruct that process as a statistical model: given decades of citizen-science observations of where a species was found and in what numbers, infer the underlying biological rules — survival rates, reproduction rates, dispersal distances, carrying capacity — that could have produced the observed range expansion.

Doing this well runs into two hard sub-problems, and the repository is really organized around solving them one at a time before combining the results:

1. **What does "suitable habitat" even mean, and how does it change over the last 125 years?** Raw climate variables (temperature, precipitation) are an incomplete proxy — two places with identical climate can differ enormously in the community of other species present, competition, and human land use. This project instead learns a *data-driven notion of habitat similarity* directly from where birds actually co-occur (via eBird), and then teaches a model to predict that notion of similarity from climate and urbanization data alone, so it can be extrapolated back to 1900, long before eBird existed. This is the **community encoder** subsystem (`src/community_encoder/`).
2. **Given a map of habitat quality through time, how do you turn that into a mechanistic population model whose parameters you can actually fit to real observations?** This requires an explicit, age-structured, spatially-explicit population dynamics model — with dispersal, density dependence, and reproduction — solved forward in time on a grid, and then fit via Bayesian inference against 50+ years of Breeding Bird Survey (BBS) counts. This is the **age-structured range-limit model** (`src/model/`, `src/processing/`, `src/analysis/`).

The two subsystems are connected at exactly one point: the age-structured model consumes the community encoder's output (a per-pixel, per-year latent vector called `Z`) as its measure of habitat quality, instead of using raw climate covariates directly.

The remainder of this document explains, for each part of the codebase, *why* it's built the way it is, and only then *how* — working from the big picture down to specific files and functions.

---

## 1. Tech stack — why these tools

**Why NumPyro/JAX rather than PyMC.** The project began as a PyMC model (its very first commit), but PyMC's autodiff and array machinery were not well suited to the FFT-based spatial convolutions and long forward-simulation loops the model eventually needed, so it migrated wholesale to **NumPyro on top of JAX** — a grep across `src/` and `scripts/` turns up zero `import pymc` statements anywhere. The package was historically named `range-limits-pymc` and carried a `pymc==5.26.1` dependency; both were leftovers from that earlier version and have since been removed (the package is now `houfin-range-model`, and the unused `pymc` and `scikit-learn` dependencies were dropped).

**Why JAX specifically.** The age-structured model's forward simulation is a decades-long, pixel-by-pixel time loop involving Fourier-transform dispersal kernels; JAX's `lax.scan`, `jax.checkpoint` (gradient checkpointing, needed because the unrolled simulation would otherwise be too memory-hungry to differentiate through), and native GPU execution make this tractable. The code assumes a CUDA GPU is available — it explicitly requests `jax.devices("gpu")[0]`, sets CUDA allocator environment variables, and manually manages device memory (deleting intermediate posterior samples, memory-mapping large arrays to disk) to avoid exhausting VRAM. This is workstation-tuned code, not designed to run comfortably on a laptop CPU.

**Why PyTorch shows up too.** The community-encoder's second stage (DESK, below) is a from-scratch multi-branch autoencoder. It's written in PyTorch rather than JAX simply because it's a much more conventional supervised deep-learning task (train an MLP against a fixed target) with no need for JAX's custom-derivative machinery — a case of using the right tool for a self-contained sub-problem rather than forcing everything into one framework.

**Other libraries, and what they're actually for:**
| Library | Where | Why |
|---|---|---|
| `rasterio` / `rioxarray` | throughout `scripts/` | reading, reprojecting, and aligning every raster data source (PRISM climate, eBird abundance, urbanization index, ocean mask) onto one common 4 km Albers grid |
| `geopandas` / `shapely` | `scripts/ingest_bbs_data.py` | BBS route geometry and convex-hull native-range boundaries |
| `dendropy` | `scripts/avonet_pipeline.py` | parsing a bird phylogeny (Hackett tree) to compute phylogenetic distance from other species to the house finch |
| `optax` | `src/model/age_run_*.py` | JAX-native optimizers (AdamW, cosine decay, gradient clipping) for SVI/MAP fitting |

**Why there's no CLI or notebooks checked in.** Most scripts are run directly rather than through a CLI framework — this is characteristic of an active research codebase where parameters change from run to run faster than a CLI abstraction would be worth building. Filesystem paths, however, are no longer hand-edited per script: they now come from three JSON files under `config/` (see §4), loaded through the shared helper `src/config_utils.py`. A `notebooks/` directory is `.gitignore`d, implying exploratory notebooks exist locally but were deliberately kept out of version control. `main.py` at the repository root is an unused placeholder stub.

---

## 2. Repository structure and workflow

### 2.1 The big picture: two pipelines feeding one model

```
                    ┌─────────────────────────────────────────┐
                    │   COMMUNITY ENCODER  (habitat quality)    │
                    │        src/community_encoder/             │
                    │                                            │
  eBird abundance ──┤  ESK: kernel-PCA on species similarity    │
   (2023 only)      │       → Z.npy  (one year, ground truth)   │
                    │                                            │
  PRISM climate  ───┤  DESK: autoencoder predicts Z from        │
  + urbanization    │        climate/urbanization alone         │
                    │       → Z_latent_{year}.npy, 1900-2024     │
                    └──────────────────┬────────────────────────┘
                                       │  Z (habitat-quality latent vector,
                                       │    every pixel, every year)
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │  AGE-STRUCTURED RANGE-LIMIT MODEL          │
                    │      src/model/, src/processing/            │
                    │                                            │
  BBS survey     ───┤  path-integrated dispersal features        │
  counts             │  + age-structured population dynamics      │
                    │  fit via NumPyro (MAP / SVI / HMC)          │
                    └──────────────────┬────────────────────────┘
                                       ▼
                          src/analysis/, src/vis/
                     (diagnostics, posterior summaries, maps)
```

The **why** behind this split: habitat-quality estimation and population-dynamics inference are different statistical problems with different data requirements (one needs a single richly-sampled year of species co-occurrence data; the other needs a full time series but only sparse survey counts), so they are developed, validated, and run as separate pipelines, joined only through the intermediate `Z` arrays written to disk.

### 2.2 `src/community_encoder/` — building a history of habitat quality

**Why a two-stage design (ESK then DESK) rather than one model.** eBird's species-abundance data — the richest available signal of *which pixels are ecologically similar* — is essentially only usable for recent years (2023 in this project); there's no way to run the same analysis on 1900 because the underlying observational data doesn't exist that far back. Climate and urbanization data, by contrast, exist (or can be reconstructed) for the entire 20th century. The two-stage design exploits this: build a trustworthy "ground truth" of habitat similarity where the rich data exists (2023), then train a *second, simpler* model that learns to reconstruct that same latent space from only the climate/urbanization variables that are available for every year — and use that second model to extrapolate backwards in time.

- **ESK** (`src/community_encoder/train_DESK/esk_kernel.py`) is not a neural network — it's a **kernel PCA** computed on a **Ruzicka similarity kernel** (a generalized Jaccard / Bray–Curtis-style similarity) between every pair of pixels' weekly, per-species eBird abundance vectors. Why Ruzicka: it's a similarity measure well suited to sparse, non-negative abundance data, which is exactly what per-species eBird rasters look like. Because computing a full pairwise kernel over every land pixel is computationally infeasible, the code uses a **Nyström landmark approximation**: it samples a subset of "landmark" pixels (18,000, per `config/esk_desk_config.json`), builds the exact kernel among just those, eigendecomposes it, and projects every other pixel onto the resulting eigenvectors. The result, swept over several temporal-smoothing bandwidths (`sigmas`) and target dimensionalities (`latent_dims` = 8/16/32), is `Z.npy` — a single year's latent habitat-similarity space, along with a `valid_mask.npy` marking which pixels have usable data.
- **DESK** ("Dynamic ESK", `src/community_encoder/train_DESK/desk_training.py`) is a genuine PyTorch model: a `MultiInputAutoencoder` with two encoder branches (one for PRISM climate variables, one for the urbanization/BUI raster), merged and compressed into a latent vector, then decoded back into both inputs. It's trained with three loss terms, matching the `weights` block in `config/esk_desk_config.json`:
  - **stabilizing** — mean-squared error against ESK's 2023 ground-truth `Z`, so the learned latent space actually matches the "real" similarity structure where it's known;
  - **metric** — a metric-learning loss over random pixel pairs that preserves Ruzicka-similarity relationships, so pixels that are ecologically similar stay close in latent space even away from the labeled year;
  - **reconstruction** — standard autoencoder reconstruction loss on the climate/urbanization inputs, computed on both the labeled year and on unlabeled historical years, which is what lets the model be trained semi-supervised across the full 1900–2024 span rather than only on 2023.
  
  Why this matters: once trained, DESK needs only climate and urbanization rasters — available for any year — to produce a `Z`-like vector, sidestepping the fact that eBird data doesn't exist for most of the time period the range-expansion model needs to cover.
- `prism_bui_smoothers.py` is the ETL step feeding DESK: it loads monthly PRISM climate GeoTIFFs (7 variables) and multi-band urbanization ("BUI") rasters, applies a 10-year exponential moving average (to represent gradually-changing climate/land-use rather than noisy year-to-year fluctuation), and writes one `state_{year}_bio_ema10.npz` file per year.
- `build_final_z_cube.py` applies the trained DESK model to every year's smoothed state, producing the final artifact of this subsystem: `Z_latent_{year}.npy` for 1900–2024 — the "spacetime cube" of habitat quality that the population model consumes. Missing or edge-case pixels are filled in three passes (spatial interpolation, backfilling from the static ESK ground truth where available, then nearest-neighbor cleanup).
- The remaining files in this directory (`analyze_final_z_cube.py`, `climate_vs_z_turnover.py`, `urbanization_vs_z_turnover.py`, `latent_interpreter.py`, `generate_z_gif.py`, `sanity_check_Z.py`, `sanity_check_houfin_regression.py`) are diagnostics — they answer "does this latent space actually make ecological sense?" The most important of these, `sanity_check_houfin_regression.py`, fits a closed-form Bayesian linear regression of observed house-finch abundance on `Z` — the project's basic sanity check that the learned latent space is at all predictive of the species this whole project is trying to model. `latent_interpreter.py` is the richest interpretive tool (phenology, per-species loadings, spatial variograms); it previously existed as two near-identical copies (a Hellinger-transform variant and a Ruzicka variant), now consolidated into one module with a `--transform ruzicka|hellinger` flag.
- `analysis_2023/` is a newer, config-driven rewrite of the same house-finch-regression/comparison logic, and is explicitly an in-progress consolidation (visible in the git history as "Incomplete reorganization of ESK/DESK visualization"). All active scripts in this subsystem now read their paths from `config/esk_desk_config.json` (via `src/config_utils.py`) rather than hardcoding them.

### 2.3 `src/model/`, `src/processing/`, `src/analysis/` — the age-structured range-limit model

**Why an age-structured model rather than a single population count.** Juvenile and adult birds disperse very differently — juveniles range much farther after fledging than established adults do in subsequent years — and this difference is central to how a range expands geographically over time. A model that lumps ages together would be structurally unable to capture that. So the population at each pixel is tracked as two numbers, adults and juveniles, each with its own survival and dispersal rules.

The processing pipeline, in dependency order:

```
generate_all_path_features.py  (src/processing/, using build_kernels.py + build_path_features.py)
        reads Z_latent_{year}.npy  →  writes Z_disp_{year}.npz   (path-integrated dispersal features)
                    │
ingest_model_data.py  (src/processing/, using build_kernels.py)
        reads Z_disp_{year}.npz + BBS survey data
                    →  writes metadata.pkl, Z_gathered.dat, Z_disp_gathered.dat
                    │
age_priors.py :: build_model_2d   (src/model/, using age_fields.py + age_forward.py)
        the NumPyro generative model itself
                    │
        ┌───────────┼────────────────┐
        ▼           ▼                ▼
  age_run_map.py  age_run_svi.py  (each followed by a resume/refine step)
        │              │
  age_resume_svi_from_map.py   age_run_hmc.py   (NeuTra-reparameterized NUTS, warm-started from SVI)
  age_resume_hmc.py            (plain HMC, warm-started from MAP)
                    │
       src/analysis/{engine,stats,plots}.py, analyze_svi.py
       src/vis/visualize_{advi,hmc,age}_model.py
```

**Why habitat quality feeds in through two related but distinct pathways.** Two scalar "habitat manifolds" are derived from the same underlying `Z`: a survival manifold `H_s` and a reproduction manifold `H_r`, each a learned linear projection of `Z` (the projection weights `beta_s`/`beta_r` are drawn from a correlated 2-D multivariate normal, letting the model learn whether the features that predict good survival also predict good reproduction, or not). Separately, a *path-integrated* version of `Z` — `Z_disp`, produced by convolving `Z` with the same dispersal kernels used for movement — approximates the habitat quality a disperser experiences while traveling, not just once it arrives, and feeds a third quantity governing in-transit mortality. The **why** here: a juvenile that must cross 300 km of poor habitat to reach a good patch should have a different survival probability than one that starts in good habitat, and a single per-pixel habitat value can't represent that.

**Why dispersal is computed via FFT convolution on a toroidal grid.** Simulating dispersal as literal draws from a probability kernel for millions of pixels over decades of yearly time steps would be computationally prohibitive; convolving a population raster with a dispersal kernel is mathematically equivalent and can be done efficiently in the frequency domain via FFT. The grid is padded and treated as wrapping (toroidal) purely as an implementation convenience for the FFT, and an explicit "edge correction" — a cross-correlation with the land mask — is applied to prevent dispersal probability mass from being silently absorbed into ocean cells that don't exist in the biological system being modeled. Adults use one isotropic kernel (mean dispersal distance ~100 km); juveniles use a **12-cohort directional kernel stack** (4 compass directions × 3 log-spaced distance bins, mean dispersal distance ~330 km) — reflecting the biological fact that young birds disperse roughly three times farther than adults, and that dispersal is not perfectly isotropic.

**Why reproduction includes an explicit Allee effect.** At the leading edge of a range expansion, population densities are low, and a lone disperser may struggle to find a mate at all — a dynamic that a simple density-independent birth rate cannot represent, and one that matters a great deal for correctly modeling how fast (or whether) a range edge advances. The model represents this with an encounter-rate-style mate-finding probability, `1 − exp(−γ·N)`, layered on top of a Beverton–Holt-style density-dependent fecundity term, where `γ` is fit from data (parameterized as the population size giving 50% mate-finding probability, `N50`).

**Why the model is seeded with an explicit invasion pulse.** The house finch population in the eastern US is not endogenous — it began from a documented human-caused release in New York City. Rather than trying to infer an implausible spontaneous-origin scenario, the model hardcodes the invasion's geographic origin (Queens, NY) and lets a learned time series of introduction magnitudes and a learned introduction timestep determine how the invasion pulse enters the simulation.

**Why three different inference procedures (MAP, SVI, HMC) are all present.** This reflects a standard pattern for hard, high-dimensional Bayesian models: MAP (point estimation via optimization) is cheap and used to find a good starting region; SVI with a low-rank multivariate normal guide gives an approximate posterior at moderate cost and is used to both explore and to *warm-start* HMC; full HMC/NUTS (further improved with a NeuTra reparameterization derived from the SVI fit) gives the most trustworthy uncertainty quantification but is by far the most expensive, so it's run last, informed by everything already learned from the cheaper methods. An annealing schedule that progressively tightens prior/noise variances during optimization is used throughout to stabilize what is otherwise a difficult, poorly conditioned posterior geometry.

**Why the likelihood is negative-binomial, not Poisson.** Real ecological count data is almost always overdispersed relative to a Poisson distribution (variance exceeds the mean, due to unmodeled heterogeneity in detection and local conditions); the negative-binomial-2 likelihood adds a dispersion parameter to absorb that, giving more realistic uncertainty estimates than a Poisson model would. (A stale code comment still referring to "Poisson" is a documentation artifact from an earlier version of the model.)

**Why an "identifiability" penalty on the age ratio exists.** Age-structured models like this one can have multiple parameter combinations that produce nearly identical total-population trajectories but wildly different (and biologically implausible) adult/juvenile ratios at equilibrium. Rather than hard-constraining this — which risks ruling out a genuinely correct solution — the model applies a gentle soft penalty (via a nearly-flat Beta prior factor) on the analytically-derived equilibrium juvenile fraction, nudging the model away from implausible regions of parameter space without forbidding them outright.

**A note on the earlier model generation (now removed).** An earlier single-age-class model — driven directly by PCA'd raw bioclim covariates via a Hilbert-Space Gaussian Process approximation, rather than by the community encoder's `Z` — used to live in `src/model.py`, `src/dispersal_precompute.py`, `src/ingest.py`, and `src/ingest_directional.py`, with downstream tooling in `scripts/reconstruct_map_results`, `scripts/plot_maps.py`, and `scripts/sanity_check_forwardsim.py`. That generation was fully superseded by the age-structured, `Z`-driven model described above and has been deleted; nothing in the current codebase imports it. This is noted only so that references to those filenames in old commits or notebooks aren't mistaken for live code.

### 2.4 `src/vis/` — where the two subsystems are explicitly compared

Most of `src/vis/` is diagnostic plotting for one subsystem or the other (`check_bbs_npz.py`, `check_ingested_data.py` for ingestion outputs; `visualize_{advi,hmc,age}_model.py` for the three inference backends). The two sample-based visualizers — `visualize_advi_model.py` (SVI) and `visualize_hmc_model.py` (HMC) — share their common plotting functions through `src/vis/_age_vis_common.py`, keeping only backend-specific plots of their own; `visualize_age_model.py` (MAP) works on point-estimate objects and keeps its own richer plot set. One file, `visualize_community_similarity.py`, is the place where the two halves of the project are put in direct conversation with each other: it loads a fitted age-structured model, extracts the learned survival/reproduction projection weights (`beta_s`/`beta_r`), and projects the *entire* eBird species-abundance stack onto those same directions in the community encoder's `Z` space. The cosine similarity between each other species' community centroid and the house finch's learned niche direction is then interpreted as a measure of ecological "mimicry" — which other species occupy a niche similar to the one the model inferred for the house finch — and cross-referenced against AVONET trait and phylogenetic distances (`scripts/avonet_pipeline.py`) to ask whether ecological similarity tracks evolutionary relatedness or not.

### 2.5 Scripts directory — one-off ETL, experiments, and duplication to be aware of

`scripts/` is largely a flat collection of one-off data-preparation utilities that feed the two subsystems above (data sources are covered in §4). A few things worth flagging for anyone navigating it:

- The ESK/DESK/cube runners live under `scripts/experiments/` (`run_esk.py`, `run_desk.py`, `run_build_final_z_cube.py`, `run_single_year_analysis.py`) — thin launchers that load a config and call into `src/community_encoder/`. (Earlier top-level `scripts/run_esk.py`/`run_desk.py` were byte-identical duplicates and have been removed.)
- `scripts/project_ebird` and `scripts/fft_optimization_test` lack a `.py` extension and must be invoked as `python scripts/project_ebird`, etc.
- `scripts/test_kernel_physics.py` and `scripts/test_path_features_single_year.py` are the closest things to genuine tests in the whole repository: proper `argparse`-driven scripts that check the FFT dispersal kernel's *empirical* mean dispersal distance against its theoretical target, and spot-check the path-integration feature pipeline on a single year. They produce diagnostic plots for a human to inspect rather than pass/fail assertions (see §5).

---

## 3. The science — what is actually being modeled, in plain terms

**The subject.** The historical spread of the house finch (*Haemorhous mexicanus*) across eastern North America, following its introduction in New York City in the 1940s (a well-documented human-caused release, hardcoded into the model as a fixed geographic origin), tracked over subsequent decades by the Breeding Bird Survey.

**The core modeling idea.** Rather than fitting a purely statistical curve to range-expansion data, the model is *mechanistic*: it encodes an explicit hypothesis about the biological processes that generate a range expansion — survival, reproduction, density-dependent dispersal — as a forward simulation on a spatial grid, and then uses Bayesian inference to find the parameter values (survival curves, dispersal distances, carrying capacities, mate-finding thresholds, etc.) under which that simulation best reproduces the BBS observations, while also quantifying the uncertainty in those parameter estimates. This lets the fitted model make biologically interpretable claims (e.g., "the data support a mean juvenile dispersal distance of X km, with credible interval Y–Z") rather than just producing predictions.

**Why habitat quality is learned rather than assumed.** A recurring modeling choice throughout this project is to *not* hand-pick which environmental variables matter and how — instead, a data-driven "habitat similarity" representation is learned from co-occurrence patterns (the community encoder's `Z`), and the population model then learns which directions in that learned space predict survival and reproduction. This two-step "let the data define the feature space, then fit the mechanistic model on top of it" pattern is the throughline connecting the two halves of the repository.

**The mathematical machinery, briefly:**
- *Habitat similarity*: kernel PCA on a Ruzicka similarity kernel (community encoder, §2.2).
- *Extrapolation across time*: a semi-supervised multi-branch autoencoder trained with a combination of supervised (match the known 2023 answer), metric-learning (preserve pairwise similarity structure), and reconstruction losses.
- *Population dynamics*: a two-age-class (Leslie-matrix-like), spatially-explicit simulation with FFT-based dispersal convolution, sigmoid/softplus demographic link functions, Beverton–Holt density dependence, and a Poisson-encounter Allee effect for mate-finding.
- *Inference*: NumPyro's MAP (AutoDelta), SVI (AutoLowRankMultivariateNormal), and NUTS/HMC (including a NeuTra-reparameterized variant), used in sequence as progressively more expensive and more rigorous approximations to the true posterior, against a negative-binomial-2 observation likelihood.

---

## 4. Data pipeline — entry points, formats, and gaps

### 4.1 External data sources and how they enter the pipeline

| Source | What it is | Raw format | Entry script | Downstream form |
|---|---|---|---|---|
| PRISM | monthly gridded US climate (precipitation, temperature, vapor pressure deficit) | NetCDF | `scripts/download_prism.py` → `scripts/project_prism.py` | reprojected 4 km GeoTIFF → yearly EMA `.npz` state (`prism_bui_smoothers.py`) |
| eBird | weekly per-species abundance-median rasters | GeoTIFF | `scripts/project_ebird` | reprojected 4 km GeoTIFF, consumed directly by ESK |
| BBS (Breeding Bird Survey) | route-level bird count survey data | CSV (`Weather.csv`, `Routes.csv`, per-state counts) | `scripts/ingest_bbs_data.py` | `bbs_data_for_python.npz` (gridded counts + native-range/pseudo-zero absence records) |
| AVONET + phylogeny | bird morphological traits + Hackett-tree phylogeny | CSV + Nexus | `scripts/avonet_pipeline.py` | merged/filtered CSVs of trait/phylogenetic distance to house finch |
| Urbanization ("BUI") | multi-band human land-use intensity index | raster (PNG-style multi-band) | `scripts/aggregate_and_interpolate_bui.py`, `bui_lowres.py` | folded into yearly EMA `.npz` state alongside PRISM |
| Ocean/water mask | land/water boundary | derived from BUI nodata, and separately a raw global watermask | `scripts/extract_ocean_mask.py`, `scripts/watermask_project.py` | `ocean_mask_4km.tif`, used throughout as the canonical land mask |

All of these are aligned onto a **common 4 km Albers-projection grid** via `rioxarray.reproject_match` — a deliberate design choice so that every subsequent pipeline stage can treat "pixel index" as a stable, shared coordinate system across all data sources and years.

### 4.2 Format progression through the pipeline

Raw rasters (`.tif`/`.nc`) → yearly smoothed states (`.npz`) → community-encoder latents (`Z.npy`, `Z_latent_{year}.npy`) → path-integrated dispersal features (`Z_disp_{year}.npz`) → flattened model-ready memory-mapped binaries (`Z_gathered.dat`, `Z_disp_gathered.dat`) → fitted-model checkpoints (`.pkl` for MAP/SVI, `.pth` for the DESK network) → diagnostic outputs (`.png`/`.gif`/`.mp4`). This progression — from georeferenced rasters, to dense numeric arrays, to flat memmapped binaries — mirrors the pipeline's shift from "data that needs geographic context" to "data that only needs to be fed into a numerical inference routine," and each stage discards the geographic metadata the next stage no longer needs.

### 4.3 Configuration

Filesystem paths are centralized in three JSON files under `config/`, all loaded through the shared helper `src/config_utils.py` (which also honors an environment-variable override per file):

- **`esk_desk_config.json`** — the community-encoder subsystem: eBird/PRISM/BUI input locations, ESK sweep settings (`sigmas`, `latent_dims`, `n_landmarks`), DESK training hyperparameters and loss weights, the spacetime-cube (`latent_cube`) locations, and the `single_year_analysis` comparison paths. Consumed via `$ESK_DESK_CONFIG`.
- **`age_model_config.json`** — the age-structured model: `input_dir` (holding `metadata.pkl`, `Z_gathered.dat`, `Z_disp_gathered.dat`), the BBS/ocean-mask/eBird inputs, `results_dir` plus a `run_names` map giving the per-backend output/warm-start directory names (templated by `{precision}`), path-feature directories, and the community-similarity bridge paths. Consumed via `$AGE_MODEL_CONFIG`.
- **`data_config.json`** — just `datasets_root` and `processed_root`, the two machine-specific prefixes the one-off ETL scripts compose their paths from. Consumed via `$DATA_CONFIG`.

To run on a different machine, point these files (or the environment variables) at the local data locations; no script edits are needed. The default values still reference the original `/home/breallis/...` development machine, so they must be updated for a fresh checkout.

One deliberate subtlety: in `esk_desk_config.json`, the `single_year_analysis` block compares ESK features computed at `sigma_0.5` against DESK-cube features at `sigma_1.5`. This is **intentional** — the ESK sanity-check pipeline runs at 0.5 while the spacetime cube is built at 1.5, and `compare_esk_desk.py` deliberately cross-compares the two — not a copy-paste error.

### 4.4 Remaining gap: no sample data ships with the repository

`data/`, `misc_outputs/`, `checkpoints/`, and similar directories are all `.gitignore`d, so a fresh clone cannot reproduce any downstream analysis without first running the entire pipeline end-to-end on a machine that already has the raw PRISM/eBird/BBS datasets and enough GPU time to retrain DESK and rebuild the spacetime cube (a multi-hour-to-multi-day undertaking). There is currently no small pre-built sample dataset for onboarding or quick verification — this remains the biggest obstacle to reproducibility.

---

## 5. Tests and validation — why there is no automated suite, and what stands in for one

**Why.** This is an actively-evolving research codebase where the "correctness" of most components is a scientific question (does this dispersal kernel produce realistic spread patterns? does this latent space correlate with real abundance?) rather than a software-engineering one (does this function return the right type?). That kind of correctness is much more naturally checked by rendering a plot and looking at it than by writing an assertion — and the codebase reflects that throughout. There is no `pytest`/`unittest` usage anywhere, no `tests/` directory, and no CI configuration (`.github/workflows` or equivalent).

**What exists instead — manual, visual sanity checks:**
- `scripts/test_kernel_physics.py` and `scripts/test_path_features_single_year.py` are the most rigorous checks present: they verify that the FFT dispersal kernel's empirical mean dispersal distance matches its theoretical target, and spot-check the path-integrated feature pipeline on a single year — but they still emit diagnostic PNGs for human review rather than a pass/fail result.
- `scripts/ingest_bbs_init_check.py`, `src/vis/check_bbs_npz.py`, `src/vis/check_ingested_data.py`, `src/community_encoder/sanity_check_Z.py`, and `src/community_encoder/sanity_check_houfin_regression.py` all follow the same pattern: load an intermediate pipeline artifact, render a diagnostic plot, and rely on a human to notice if something looks wrong.
- `scripts/fft_optimization_test` is an exploratory scratch script for probing FFT/kernel behavior by hand, not validation in any formal sense.

**The gap this leaves.** This approach is appropriate for solo, exploratory research, but it means there is currently no way to detect a regression automatically — e.g., a refactor of the dispersal-kernel code, or a change to the DESK training loop, could silently break something and the only way to notice would be to re-run the relevant diagnostic script and eyeball the output. If this project moves toward collaboration or long-term maintenance, converting the more mechanical of these checks (especially `test_kernel_physics.py`'s theoretical-vs-empirical dispersal distance comparison) into real assertions would be the highest-leverage first step toward an automated test suite.

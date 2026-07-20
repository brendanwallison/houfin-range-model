# Running the data-processing pipeline on TACC Lonestar6

This runs the pipeline end-to-end: download every source, build all 25 km products
(including `climr` climate), assemble the encoder inputs, and train the DESK
community encoder through the `validate` step. The GPU encoder is a separate job
(§3c).

The repo is portable: all config paths are `${HOUFIN_DATA}` / `${HOUFIN_PROCESSED}`
references that expand from the environment (`src/config_utils.py`). You set two
env roots and run — no config edits needed.

Key Lonestar6 facts: login nodes have internet, **compute nodes do not**;
`$SCRATCH` is large but **purged after 10 days**; `$WORK` (1 TB) persists. So:
**downloads run on a login node; preprocessing runs as SLURM jobs; confirmed
products get promoted `$SCRATCH` → `$WORK`.**

### End-to-end order (what runs where)

`climr` needs internet, so the one subtlety is that a **login-node cache warm sits
between preprocessing and the climate stage** (the warm reads the sub-cell centroids
that preprocessing builds; the offline climate stage then reads the warmed cache).
Every submit wrapper injects the `-A` allocation, so use them (a bare `sbatch` fails
on multi-project accounts).

| # | Step | Where | Command |
|---|------|-------|---------|
| 1 | one-time setup | login | §1 |
| 2 | download raw data | login | `bash scripts/tacc/download_all.sh` |
| 3 | preprocess → 25 km grids (+ sub-cell centroids) | compute | `bash scripts/tacc/submit_preprocess.sh` |
| 4 | **warm the climr cache** | **login** | `bash scripts/tacc/warm_climr.sh` |
| 5 | climate downscale + grids | compute | `bash scripts/tacc/submit_climate.sh` |
| 6 | assemble encoder inputs (states, eBird cache, BBS, amplitude) | compute | `bash scripts/tacc/submit_states.sh` |
| 7 | GPU encoder (ESK → DESK → cube → validate) | GPU | `bash scripts/tacc/submit_encoder.sh` |
| 8 | visual-QC quicklooks | compute | `bash scripts/tacc/submit_visualize.sh` |

Steps 3+5+6 can instead be **one** dev-queue job on a warm cache (`00_preprocess_all`,
§3b). The only hard ordering is 2 → 3 → 4 → 5.

## 1. One-time setup (login node)

```bash
ssh <user>@ls6.tacc.utexas.edu
module load python/3.12.11

# Clone onto $WORK (persistent).
mkdir -p $WORK/houfin && cd $WORK/houfin
git clone https://github.com/brendanwallison/houfin-range-model.git houfin-range-model && cd houfin-range-model

# Python env with uv (base deps are enough for data processing — no GPU).
curl -LsSf https://astral.sh/uv/install.sh | sh          # userspace uv

# Throttle uv's threads/concurrency. The LS6 login node has 128 cores, so uv's
# default thread pool trips the per-user process/memory limit (# "failed to initialize global rayon pool ... Resource temporarily unavailable").
export RAYON_NUM_THREADS=4
export UV_CONCURRENT_DOWNLOADS=4
export UV_CONCURRENT_BUILDS=2
export UV_CONCURRENT_INSTALLS=4

uv venv $WORK/houfin/venv --python 3.12        # venv must exist before env.sh can activate it

# Edit the three marked values (allocation, repo path, venv path), then source it:
# env.sh activates the venv, sets HOUFIN_DATA/PROCESSED, and cd's to the repo.
export EDITOR=nano
$EDITOR scripts/tacc/env.sh
source scripts/tacc/env.sh

uv pip install -e .                            # installs into the now-active venv (base deps only)

# Secrets: eBird key (required for the eBird download).
cp config/secrets.example.json config/secrets.json && $EDITOR config/secrets.json
# ...or: export EBIRD_KEY="..."

# R for the climate step (climr). Two reasons TACC's Rstats/4.0.3 module can't do
# this (both verified by trying it): (1) climr's CRAN dependency tree (dplyr, tidyr,
# ggplot2, scales, glue, fs, purrr, ...) now requires R >= 4.1, so on R 4.0.3 those
# come back "not available" and the install dead-ends; (2) climr is GitHub-only
# (bcgov/climr, not on CRAN) and its native deps (terra, sf, RPostgres) need
# GDAL/GEOS/PROJ + libpq. Fix both with a userspace modern R: Lonestar6 has no
# conda/mamba module, so bootstrap the static micromamba binary and pull R 4.3+ plus
# the compiled deps prebuilt from conda-forge. env.sh auto-detects
# $WORK/houfin/renv/bin/Rscript.
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $WORK/houfin bin/micromamba
export MAMBA_ROOT_PREFIX=$WORK/houfin/micromamba
# LS6 login-node gotcha: `ulimit -u` is hard-capped at 300 processes/user, and a
# one-shot create of all ~225 packages spawns more workers than that -> the create
# dies with "Resource temporarily unavailable" (EAGAIN) or a std::bad_alloc/segfault
# with garbled package names. `taskset`/thread-config knobs do NOT fix it. What works
# is splitting the install into smaller transactions (each well under 300 packages),
# with MAMBA_DOWNLOAD_THREADS=1 and taskset for good measure. If a run dies mid-way,
# `micromamba clean --all --yes` first to drop any corrupted cached tarballs.
export MAMBA_DOWNLOAD_THREADS=1
MM="taskset -c 0-3 $WORK/houfin/bin/micromamba"
$MM create  -y -p $WORK/houfin/renv -c conda-forge r-base           # base R + libs closure
$MM install -y -p $WORK/houfin/renv -c conda-forge r-terra r-sf r-rpostgres
$MM install -y -p $WORK/houfin/renv -c conda-forge r-remotes r-data.table r-dplyr r-tidyr \
    r-stringi r-curl r-uuid r-ggplot2 r-plotly     # drop any r-* name that won't resolve
# climr from GitHub; remotes fills the remaining pure-R deps from CRAN. upgrade=never
# keeps the conda-forge binaries rather than rebuilding them from source. If you hit
# a GitHub API rate limit, export GITHUB_PAT=<token> first.
$WORK/houfin/renv/bin/Rscript -e 'options(download.file.method="libcurl"); remotes::install_github("bcgov/climr", upgrade="never")'
$WORK/houfin/renv/bin/Rscript -e 'suppressMessages(library(climr)); cat("climr OK\n")'

# The env.sh you sourced back in step 1 ran before this renv existed, so its
# HOUFIN_RSCRIPT/PROJ_DATA point at the fallback. Re-source it now (env.sh re-detects
# the renv) so the current shell uses it:
source scripts/tacc/env.sh
```

> **R / climr wiring.** Both the batch climate step (`02_climate.slurm`) and the
> login-node cache-warm (`scripts/tacc/warm_climr.sh`) read `$HOUFIN_RSCRIPT`, set in
> `scripts/tacc/env.sh`. That var auto-detects the micromamba env at
> `$WORK/houfin/renv/bin/Rscript` and otherwise falls back to PATH `Rscript`; to use
> a different R, `export HOUFIN_RSCRIPT=/path/to/Rscript` before sourcing env.sh.
> Because that Rscript is invoked directly (not via `micromamba activate`), env.sh
> also exports `PROJ_DATA`/`GDAL_DATA` pointing at the env's `share/` dirs — without
> them terra/sf can't find `proj.db` ("problem with the PROJ installation") and
> reprojections misbehave.
> Why not TACC's `Rstats/4.0.3` module: its R is too old for climr's dependency
> tree (dplyr/tidyr/ggplot2/scales now need R >= 4.1, so they resolve as "not
> available" on 4.0.3), and the generic `RstatsPackages` companion module that
> would supply prebuilt packages isn't deployed on LS6 (`module spider` can't find
> it). The self-contained conda-forge env above sidesteps both.

**Reference species list.** The eBird download reads `species_list`
(`${HOUFIN_DATA}/avonet/reference_community_ranked.csv`), produced by
`scripts/avonet_pipeline.py`. All of its inputs are automated by
`scripts/download_avonet.py`: AVONET traits + the BirdLife/BirdTree crosswalk +
the precomputed Hackett phylogeny (public figshare `ELEData.zip`), the eBird
taxonomy (eBird API), and the urban-tolerance indices (public figshare). So
`download_all.sh` runs AVONET → the ranked species list → the eBird download in
order, no manual staging.

## 2. Download raw data (login node — NOT sbatch)

```bash
source scripts/tacc/env.sh
nohup bash scripts/tacc/download_all.sh > download.log 2>&1 &
tail -f download.log
```

Fetches eBird, BBS (+Mexico), LUH-3 (~8 GB), HYDE, SoilGrids, DEM (ETOPO ~0.5 GB),
and Natural Earth land. It ends with a one-point **climr connectivity check** (is
climr installed + its server reachable?) — **not** the study-region cache warm,
which is a separate login-node step *after* preprocessing (§3a, `warm_climr.sh`),
because it needs the sub-cell centroids preprocessing builds. All idempotent — safe
to re-run. Lands under `$HOUFIN_DATA`.

## 3. Preprocess to 25 km grids (SLURM, compute)

```bash
bash scripts/tacc/submit_preprocess.sh     # 01_preprocess (dev, 2h); injects -A
squeue -u $USER
```

`01_preprocess.slurm` (CPU, RAM-light): ref grid → land mask → eBird → LUH-3 →
HYDE → SoilGrids → elevation → **sub-cell centroids** → BBS ingest, validating each
stage (JSON manifest per product in `$HOUFIN_PROCESSED/validation/<stage>.json`).
The sub-cell centroids (`elevation/subcell_centroids.csv`, ocean-masked) are what
the climr warm and the climate stage tile over, so they must exist before either.

> The login-node climr warm (§3a) must run **between** preprocess and climate, so
> there is no auto-chain of `01 → 02` — use the explicit `submit_preprocess.sh` →
> `warm_climr.sh` → `submit_climate.sh` order (the climate wrappers refuse to queue
> on a cold cache).

## 3a. Warm the climr cache (login node — the one online step after preprocessing)

```bash
source scripts/tacc/env.sh
bash scripts/tacc/warm_climr.sh            # python scripts/climate_climr.py --warm-cache
```

Compute nodes have no internet, so the offline climate stage can only read an
already-warmed cache. `warm_climr.sh` warms the **whole study region**, one R
process **per geographic tile** (`climate.tiles`² tiles; default 8 → up to 64) — a
fresh process per tile because a single long-lived R process leaks GDAL threads
until the login node's `ulimit -u` (~300) kills it. Per tile it caches the reference
map by a **verbatim file copy + a header-only band rename** (`terra::update`, not
`pre_cache()`/`writeRaster`, which the login cgroup can't afford) and records the
**requested** bbox (not the clip's data extent) so the offline read hits at every
tile — including coastal/edge tiles where the refmap is partly nodata. It downloads
the **cru.gpcc** observed surfaces for the full year range and is **resumable**
(re-running fetches only missing tiles).

The cache lives on **persistent `$WORK`** (`R_USER_CACHE_DIR=$HOUFIN_CLIMR_CACHE`,
set in `env.sh`), **not** the 10-day-purged `$SCRATCH`. So this is effectively a
**one-time** step: it survives the purge, is shared to the compute nodes, and every
later run — including the `00_preprocess_all` one-shot (§3d) — reads it warm. You
only re-warm if you delete the cache or climr ships new reference data.

- **`db_option=local` is mandatory offline.** climr's default `db_option="auto"`
  runs the observed time-series on its **remote database server** (→ `Database
  connection issue` on an internet-less compute node). Config sets
  `climate.db_option: "local"` so climr downloads+caches locally; the warm uses the
  same mode + the **same centroids and `climate.tiles` tiling** as the compute
  stage, so the cached tile bboxes match what compute requests.

## 3b. Climate downscale + grids (SLURM, compute)

```bash
bash scripts/tacc/submit_climate.sh                              # dev, offline
# full range on the normal queue if needed:
QUEUE=normal TIME=12:00:00 bash scripts/tacc/submit_climate.sh
```

`02_climate.slurm` runs the offline `climr` downscale (reads the warm cache), then
rasterizes to per-year bio-year grids (`climate_grid`). `submit_climate.sh` (and the
`00_preprocess_all` one-shot) **refuse to queue on a cold cache**
(`check_climr_cache.sh`) and point you at `warm_climr.sh`, so a doomed offline job is
never submitted; the stage also re-checks in-job before doing any work. Notes:

- **Name an observed dataset.** `obs_ts_dataset=cru.gpcc` (config) — without it climr
  returns only the 1961–1990 reference normal, not the annual series.
- **Climate parallelism = PROCESSES per tile (raise the worker count).** Each
  geographic tile runs offline in its own R process via a process pool (no shared-DB
  contention now that reads come from the cache). `HOUFIN_CLIMATE_WORKERS` is the
  number of concurrent tile processes — `climatena.py` default 4, `02_climate.slurm`
  sets **16**, cap ~96 to fill a 128-core node. **Raise it** (`HOUFIN_CLIMATE_WORKERS=32`)
  to use the node; the old advice to keep it at 1 no longer applies. `HOUFIN_CLIMATE_THREADS`
  = climr threads per process (default `cpus/workers`).
- **Climate `mode`.** Default **`subgrid`** (`data_config.json:climate.mode`): a
  `grid`×`grid` mesh (`climate.subgrid.grid`, default 5×5) of true-elevation
  sub-points per cell → per-cell *spatial* q10/q50/q90 (captures coast/rain-shadow
  gradients, not just elevation). Set `climate.mode: "elev_quantile"` for the cheaper
  centroid-at-3-elevations path. **Switching mode requires clearing
  `$HOUFIN_DATA/climate`** (incl. `_chunks` — stale chunks mismatch).
- `climate.tiles` (default 8) sets the geographic tiling shared by warm + process;
  `climr_obs_min_year`/`climr_obs_max_year` (1901/2024) bound the obs range, with a
  `first_year-1` bio-year lookback clamp.

## 3c. Assemble encoder inputs (SLURM, compute)

```bash
bash scripts/tacc/submit_states.sh         # 04_states (dev); injects -A
```

`04_states.slurm` runs the CPU / torch-free pre-encoder assembly —
`states` (per-year N-stream `state_{year}.npz` + `state_schema.json`),
`ebird_cache` (reprojected eBird stack `E`), `bbs` (AOU↔eBird crosswalk + gridded
`community_matrix`), and `amplitude` (the `x = E·anomaly` point cube) — so the GPU
job is purely the encoder. `HOUFIN_STATES_WORKERS` (default 16) sets the parallel
per-year npz compression writers; `build_states` also pre-reads rasters in parallel
and pins native thread pools, so it uses the node cleanly.

## 3d. Or: all preprocessing in one CPU job (warm-cache re-runs)

On an **already-warm** climr cache, §3 + §3b + §3c collectively fit the 2-hour
`development` queue as a single job. `scripts/tacc/pipeline.sh` defines the chain as
selectable stages and `00_preprocess_all.slurm` runs it:

```bash
git -C $WORK/houfin/houfin-range-model fetch && git checkout bbs-spacetime-desk
bash scripts/tacc/submit_preprocess_all.sh     # dev, 2h
squeue -u $USER ; tail -f houfin_preall.o<jobid>
```

Stages: `preprocess → climate → climate_grid → states → ebird_cache → bbs →
amplitude`. This produces everything the encoder needs — N-stream
`state_{year}.npz` (+ `state_schema.json`), the eBird stack cache, the BBS
community matrix, and the amplitude-modulated point set. Select a subset with
`STAGES` (skip what already ran); e.g. if the old preprocess + climate outputs are
present, just build the new artifacts:

```bash
STAGES="climate_grid states ebird_cache bbs amplitude" \
  bash scripts/tacc/submit_preprocess_all.sh
```

For `bbs_mode=off` (eBird-only, no BBS) drop `bbs amplitude`. **This one-shot
assumes the climr cache is already warm** (§3a) — the `climate` stage is offline and
cannot warm it. On a warm cache the whole chain (incl. climate ~minutes with a
raised worker count) fits dev 2h. Cold start: run §3 → §3a (login warm) → then this
one-shot with `STAGES` starting at `climate`, or just use the split
`submit_preprocess.sh`/`submit_climate.sh`/`submit_states.sh` scripts.

**Clean up before a fresh run.** The encoder was rewired from the deprecated
2-stream PRISM/BUI states to N-stream `state_{year}.npz` + `state_schema.json`, so
clear stale encoder artifacts so nothing mixes formats:

```bash
rm -rf $HOUFIN_PROCESSED/encoder     # regenerated downstream
# raw downloads + 25 km products under $HOUFIN_DATA are kept (preprocess reuses them)
```

Confirm the BBS release has `bbs_2026_release/{SpeciesList.csv,Weather.csv,
Routes.csv,States/*.csv}` (SpeciesList.csv drives the AOU↔eBird crosswalk).

## 3e. Encoder (ESK → DESK → cube → validate) — separate GPU job

The encoder is **not** preprocessing: ESK (Nyström kernel-PCA) and DESK
(autoencoder training) use torch and are the heavy stages, so they run on a GPU
queue, separately, and typically **one at a time** so each is sized/queued on its
own. Run after preprocessing:

```bash
STAGES=esk  bash scripts/tacc/submit_encoder.sh        # eBird-only ESK Z (GPU)
STAGES=desk bash scripts/tacc/submit_encoder.sh        # train env→Z (GPU)
STAGES="cube validate" bash scripts/tacc/submit_encoder.sh   # light; CPU is fine too
# or all four at once:  bash scripts/tacc/submit_encoder.sh
```

`validate` writes `$HOUFIN_PROCESSED/encoder/desk/validate_report.json` —
CKA/Mantel/Pearson per period, i.e. how far back the eBird-only model's implicit
predictions reproduce the BBS spatiotemporal structure.

**Env note:** the encoder needs `torch` importable (GPU: a CUDA torch build; CPU
works but slower — `cube`/`validate` are light). The repo's `gpu` extra pins
`jax[cuda12]` for the population model, *not* torch, so confirm torch is in the
venv first (`20_encoder.slurm` checks and aborts if not). `enrich` mode (folding
BBS into training) is not yet wired — decide based on the `validate` report.

## 3f. Visual-QC quicklooks (SLURM, compute)

Render the 25 km products to thumbnail PNGs and tar them for `scp` — a fast visual
sanity check at any point after the grids/states exist:

```bash
bash scripts/tacc/submit_visualize.sh --years-per-var 3 --workers 48   # rasters incl climate_grid
bash scripts/tacc/submit_visualize.sh --states --workers 48            # per-year encoder state grids
```

`03_visualize.slurm` (development queue) runs `quicklook_grids.py`, which stratifies
by variable and renders a few evenly-spaced years each (a legible variable × year
set, not thousands). Output: `$HOUFIN_PROCESSED/quicklooks.tgz`. `--states` renders
the assembled `state_{year}.npz` channels (the actual encoder inputs); `--climate`
additionally re-grids the raw per-centroid CSVs (redundant with the `climate_grid`
tifs the raster path already shows). Pull it back:

```bash
scp <user>@ls6.tacc.utexas.edu:'$WORK/houfin/processed/quicklooks.tgz' .
```

## 4. Validate + retrieve for analysis

The manifests are small and are what you send back for review (large rasters stay
on TACC):

```bash
# from your laptop:
scp -r <user>@ls6.tacc.utexas.edu:'$WORK/houfin/processed/validation' ./tacc_validation
scp <user>@ls6.tacc.utexas.edu:'$WORK/houfin/houfin-range-model/houfin_*.o*' ./tacc_logs
```

Each manifest records, per product: file size + `sha256`, raster CRS / shape /
resolution / nodata, per-band min·median·max + valid-cell count, and csv/npz
shapes. Spot-checks to confirm: land mask looks like North America; LUH-3
fractions ∈ [0,1]; HYDE population totals sane; elevation quantiles monotonic
(q10 ≤ q50 ≤ q90); `climr` temperatures colder at higher elevation.

## 5. Promote confirmed products ($SCRATCH → $WORK)

`$SCRATCH` is purged after 10 days. Once a product's manifest checks out, copy it
to `$WORK` (use `cp`, not `mv`, so striping applies):

```bash
mkdir -p $WORK/houfin/processed/products
cp -r $HOUFIN_DATA/{ref_grid_25km.tif,land_mask,ebird_weekly_2023_grid,luh3_grid,\
hyde35_grid,soilgrids_grid,elevation,climate,bbs_2026_release/bbs_data_for_python.npz} \
   $WORK/houfin/processed/products/
```

Raw downloads can stay on `$SCRATCH` (re-downloadable) or be archived to `$WORK`/Ranch.

## Deferred (later milestone)
The **population-model fit** — the NumPyro negative-binomial range model that
consumes the per-year Z cube (`Z_latent_{year}.npy`) plus the route-level BBS counts
(`bbs_data_for_python.npz`) — is the remaining downstream stage (GPU: `gpu-a100` /
`gpu-h100`; install with `uv pip install -e ".[model,gpu]"` — the `gpu` extra pins
`jax[cuda12]`). The encoder half (states/`run_states`, the `climr`→gridded-climate
step, ESK/DESK, cube, and `validate`) is now wired and covered above; `enrich` mode
(folding BBS into DESK training) is the one encoder path still to wire, gated on the
`validate` report.

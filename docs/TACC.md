# Running the data-processing pipeline on TACC Lonestar6

This covers the **data-processing milestone**: download every source and build all
25 km products (including `climr` climate), saving intermediates for later
DESK-training / model-fit milestones. Those later (GPU) stages are **deferred**.

The repo is portable: all config paths are `${HOUFIN_DATA}` / `${HOUFIN_PROCESSED}`
references that expand from the environment (`src/config_utils.py`). You set two
env roots and run — no config edits needed.

Key Lonestar6 facts: login nodes have internet, **compute nodes do not**;
`$SCRATCH` is large but **purged after 10 days**; `$WORK` (1 TB) persists. So:
**downloads run on a login node; preprocessing runs as SLURM jobs; confirmed
products get promoted `$SCRATCH` → `$WORK`.**

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
> login-node cache-warm (in `download_all.sh`) read `$HOUFIN_RSCRIPT`, set in
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
Natural Earth land, and **warms the `climr` reference cache** (the batch node can't,
having no internet). All idempotent — safe to re-run. Lands under `$HOUFIN_DATA`.

## 3. Preprocess to 25 km (SLURM)

```bash
bash scripts/tacc/submit.sh      # submits 01_preprocess, then 02_climate (afterok)
squeue -u $USER
```

- `01_preprocess.slurm` (CPU, RAM-light): ref grid → land mask → eBird → LUH-3 →
  HYDE → SoilGrids → elevation → BBS ingest, validating each stage.
- `02_climate.slurm`: `climr` downscaling (loads R; reads the warmed cache). The
  heavy/uncertain stage — see the R notes above; if it needs splitting, run it on
  a login node or chain shorter jobs.

Each stage writes a JSON manifest to `$HOUFIN_PROCESSED/validation/<stage>.json`.

## 3b. All preprocessing in one CPU job

All preprocessing (CPU, torch-free) collectively fits the 2-hour `development`
queue. `scripts/tacc/pipeline.sh` defines the chain as selectable stages and
`00_preprocess_all.slurm` runs it:

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

For `bbs_mode=off` (eBird-only, no BBS) drop `bbs amplitude`. Cold `climr` is the
only stage that can exceed 2h — if its cache isn't warm, run it once on `normal`
first (`QUEUE=normal TIME=06:00:00 STAGES=climate bash scripts/tacc/submit_preprocess_all.sh`),
then the rest on dev. (The split `01_preprocess`/`02_climate` scripts remain for
normal-queue / partial runs.)

**Clean up before a fresh run.** The encoder was rewired from the deprecated
2-stream PRISM/BUI states to N-stream `state_{year}.npz` + `state_schema.json`, so
clear stale encoder artifacts so nothing mixes formats:

```bash
rm -rf $HOUFIN_PROCESSED/encoder     # regenerated downstream
# raw downloads + 25 km products under $HOUFIN_DATA are kept (preprocess reuses them)
```

Confirm the BBS release has `bbs_2026_release/{SpeciesList.csv,Weather.csv,
Routes.csv,States/*.csv}` (SpeciesList.csv drives the AOU↔eBird crosswalk).

## 3c. Encoder (ESK → DESK → cube → validate) — separate GPU job

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

## Deferred (later milestones)
DESK training, Z-cube, path features, and the model fit (GPU: `gpu-a100` /
`gpu-h100`; install with `uv pip install -e ".[model,gpu]"` — the `gpu` extra is
pinned to `jax[cuda12]` for these nodes). Wiring `streams.run_states` and the
`climr`-CSV → gridded-climate step + climate streamer also belong to that phase.

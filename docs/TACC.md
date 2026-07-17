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
git clone <repo-url> houfin-range-model && cd houfin-range-model

# Python env with uv (base deps are enough for data processing — no GPU).
curl -LsSf https://astral.sh/uv/install.sh | sh          # userspace uv
uv venv $WORK/houfin/venv --python 3.12
source $WORK/houfin/venv/bin/activate
uv pip install -e .                                        # base deps only

# Edit the three marked values (allocation, repo path, venv path):
$EDITOR scripts/tacc/env.sh
source scripts/tacc/env.sh                                 # sets HOUFIN_DATA/PROCESSED

# Secrets: eBird key (required for the eBird download).
cp config/secrets.example.json config/secrets.json && $EDITOR config/secrets.json
# ...or: export EBIRD_KEY="..."

# R for the climate step (climr):
module load Rstats/4.0.3 RstatsPackages
export R_LIBS_USER=$WORK/houfin/Rlib && mkdir -p $R_LIBS_USER
Rscript -e 'install.packages(c("climr","data.table"), repos="https://cloud.r-project.org")'
```

> **R-version risk.** `climr` may require a newer R than TACC's `Rstats/4.0.3`. If
> the install above fails, use a userspace newer R via mamba and point the climate
> step at it:
> ```bash
> module load mamba 2>/dev/null || true
> mamba create -y -p $WORK/houfin/renv -c conda-forge r-base r-data.table
> $WORK/houfin/renv/bin/Rscript -e 'install.packages("climr", repos="https://cloud.r-project.org")'
> # then run the climate step with:  python scripts/climate_climr.py --rscript $WORK/houfin/renv/bin/Rscript ...
> ```

**Reference species list.** The eBird download reads `species_list`
(`${HOUFIN_DATA}/avonet/reference_community_ranked.csv`), produced by
`scripts/avonet_pipeline.py`. Its inputs are now mostly automated:
`scripts/download_avonet.py` fetches AVONET traits + the BirdLife/BirdTree
crosswalk + the precomputed Hackett phylogeny (public figshare `ELEData.zip`) and
the eBird taxonomy (eBird API). The **one remaining manual input** is the
urban-tolerance table `urban_avian/spp_urban_indices.csv` (a separate paper
supplement with no clean programmatic source) — stage it under
`$HOUFIN_DATA/urban_avian`, then `avonet_pipeline.py` runs and writes the species
list **before** the eBird download. (`download_all.sh` skips the pipeline with a
notice if that file is absent.)

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

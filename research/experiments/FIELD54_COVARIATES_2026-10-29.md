# FIELD54 Stage A: covariate states built natively at 54 km (2026-10-29 / 30)

Stage A of `FIELD54_NATIVE_PLAN_2026-10-29.md`. Every covariate family was rebuilt from its RAW source on
the 54 km reference grid. Nothing averages 27 km transformed values. The 302-channel states for 1902-2025
(EMA from 1882) were then built under a 54 km overlay. Interpretive nuance for anything downstream:
`research/CONSIDERATIONS.md` (M7 scales, G6).

## What was built, and where (TACC)

| Piece | Location |
|---|---|
| 54 km data root | `/scratch/07980/bwa386/houfin/data54` (`$HOUFIN_DATA54`) |
| Raw-source symlinks | `LUH3`, `hyde35`, `HBUI`, `soilgrids_5000m`, `dem`, `land_source`, pointing at `/scratch/.../houfin/data/<same>`. None points at a 27 km output. |
| Grid and mask (copied from `processed/grid54`) | `ref_grid_54km.tif` (sha 828b9b39...); `land_mask/ocean_mask_54km.tif` (sha 8f089228...) |
| Family grids | `climate/` (regrouped `climate_q{10,50,90}.csv` plus `regroup_summary.json` and `subpoint_parent_map.parquet`), `climate_grid_monthly/`, `elevation/`, `bui_grid/`, `hyde35_grid/`, `luh3_grid/` (plus `parent_weighting.json`), `luh3_grid_plainavg_diag/` (diagnostic only; states never read it), `soilgrids_grid/` |
| States | `/work/07980/bwa386/ls6/houfin/processed/field54/states/` (`yearly_states/state_{1902..2025}.npz`, `state_schema.json`, `history_vectors.npy`) |
| Checks | `/work/07980/bwa386/ls6/houfin/processed/field54/field54_checks.json` |
| Code snapshot | `/work/07980/bwa386/ls6/houfin/research/field54/code` (CRLF stripped; file list and sha256 in `SNAPSHOT_MANIFEST.txt`); logs in `../logs/` |

**Mask.** The data54 mask is byte-identical to the mask the 54 km age-model inputs use. Its sha
8f089228... is the one in `processed/grid54/land_mask` and in
`model_inputs/numpyro_input_s64_grid54_obsroute/metadata.pkl`.
- 4,318 land cells.
- Every 54 km land cell has at least 2 land children at 27 km.

**Overlays.**
- `config/overlays/field54_data.json`:
  - `datasets_root` and `grid.ref_raster` point at `${HOUFIN_DATA54}`; `target_res_m` is 54000.
  - `elevation.fine_factor` 30, `bui.fine_factor` 54, `coastline.fine_factor` 50.
  - `luh3.parent_weighted` (the map below).
- `config/overlays/field54_states.json`: the same streams, order, `ema_tau` and transforms as
  `esk_desk_config.json`. Only `grid_dir`, `hist_dir` and `water_mask_path` change.
- Jobs source `research/field54/env54.sh`, which also points `HOUFIN_DATA` and `HOUFIN_PROCESSED` at the
  54 km roots, so nothing can write into the 27 km tree.

## Code changes (uncommitted; 27 km behaviour unchanged)

- `src/data/preprocess/netcdf_grid.py`: new `parent_weighted_reproject`, `_reproject_parent_weighted`,
  `attach_parents` and `index_variables` (lines 144-271). `_reproject_one` dispatches to the new path only
  when an item carries `parents`, which happens only when the config sets `parent_weighted`.
- `src/data/preprocess/luh3.py:67-87`: the opt-in `parent_weighted` and `parent_weighted_plain_dir`
  switch. When it is on, the stage also writes `parent_weighting.json`.
- New module `src/data/preprocess/climate_regroup.py`.
- New `tests/test_field54_covariates.py`, covering:
  - the weighted rule and its fallback;
  - that the switch is off by default;
  - a regroup with a parent that straddles two chunks, which must equal `quantile_aggregate` over the union.
- Test results:
  - These tests plus `test_netcdf_grid`, `test_climate_subgrid`, `test_soilgrids`, `test_bui` and
    `test_regrid`: 27 passed locally.
  - On TACC: 7 passed.
- `research/exp/field54_checks.py` holds the checks; the coordinator added a climate year stride on TACC,
  which is synced here.
- `research/field54/*.slurm` and `env54.sh` are the jobs.

## Rules, family by family

**Climate (240 channels).** Exact union quantiles; climr was not rerun.
- Source: the stored per-sub-point climr output, `climate/_chunks/out_*/climate_points.csv` (5x5 points per
  27 km cell, fine-land filtered).
- Each sub-point goes to its 54 km parent, `parent = (row//2)*112 + col//2`.
- q10, q50 and q90 come from `climatena.quantile_aggregate` (`climatena.py:147`) over the union of the
  children's sub-points. The 54 km 10x10 lattice is the 27 km 5x5 lattice, so the union is the native set.
- The 54 chunks are tiles cut on the 27 km grid. Tile rows break at odd 27 km rows (17, 51, 85, 119), so
  253 parents straddle two chunks. Their rows were set aside and aggregated globally
  (`climate_regroup.py`), not per chunk.
- Fidelity: recomputing 27 km chunk 20 from its points reproduces the stored `climate_q*.csv` to
  2.3e-13, with no NaN mismatch. You had already verified chunk 0.
- `climate_grid` was then run unchanged under the overlay, with 54 km `cell_centroids.csv` ids
  (row*112 + col).
- `climate_grid_monthly` is the grid the states read. The annual `climate_grid` is not used, so it was not
  built.

**Elevation (3 channels).** `elevation.py` reads the fine factor from the config (line 107), and the
factor is relative to the ref grid (`dem_to_fine_grid`, line 22). Setting it to 30 gives the same 1.8 km
lattice as 27 km with factor 15. p10, p50 and p90 come from `regrid.block_quantiles` (`regrid.py:115`).
No override was needed.

**BUI (7 channels).** Fine factor 54, i.e. the same 1 km lattice as 27 km with factor 27 (`bui.py:116`
and 293).
- Quantiles are taken over the union of valid sub-cells (`cell_quantiles`, line 162).
- `bui_avail` is the sub-cell coverage fraction (line 173).
- The neutral fill was recomputed at 54 km (line 182). The q99 fill moved from 121 to 218 and the q05
  fill from 0.078 to 0.042.

**HYDE 3.5 (3 channels).** `hyde.py:24-30` already uses GDAL `average` for `population_density` and `sum`
for the urban and rural counts, applied to the 5' source. log1p is applied only at load
(`covariate_io.py:69-83`); states hold raw values. The same rule holds unchanged at 54 km.

**LUH-3 (33 channels).**
- The 14 state fractions (crop types, primf/primn, secdf/secdn, pastr, range, urban) use GDAL `average`
  (`regrid.py:238`), as at 27 km.
- 21 per-parent layers use the parent-area-weighted mean `sum(r*a)/sum(a)`. It is computed as two GDAL
  `average` reprojections, of `r*a` and of `a`, on a common valid mask, so both use identical weights.
  Where `sum(a) = 0` the plain average is used. Parents, from the netCDF long_name and units:

  | Layers | Meaning | Parent |
  |---|---|---|
  | fertl_X | kg/ha of crop X | crop X |
  | irrig_X | irrigated fraction of crop X | crop X |
  | fharv_c3per, fharv_c4per | harvested fraction of the crop's biomass | c3per, c4per |
  | flood | flooded fraction of c3ann | c3ann |
  | prtct_Y | protected fraction of land use Y | Y |
  | secma, secmb | secondary mean age and mean biomass density | secdf+secdn |
  | pltns_wdprd | fraction of *harvested* plantation biomass | pltns |

  - pltns_wdprd is a proxy and flagged: harvested biomass is not available, so plantation area stands in
    for it.
  - flood, prtct_* and pltns_wdprd go beyond the brief's list. They are per-parent by their metadata, and
    so were treated the same way.
  - Effectively constant layers: in this window fharv is constant (0.1 / 0.9), and prtct_* and pltns_wdprd
    are 0 everywhere. The rule changes nothing for them.
- **The 27 km build used a plain average** (`luh3.py`, `regrid.py:238`). Mechanism (likely): at 27 km a
  cell is roughly one 0.25° pixel, so a plain average is nearly the pixel's own per-parent rate and the
  inconsistency is small. At 54 km a cell spans about 4-6 pixels, and the plain mean dilutes the rate by
  pixels without that land use.

**Soil (16 channels).** GDAL `average` from the 5 km Goode tiles (`soilgrids.py:58`). Then
`fill_terrestrial_nodata` (line 20), a nearest-land fill on the 54 km mask, which comes from
`data54/land_mask`.

## Checks (`field54_checks.json`; land = the 4,318 54 km land cells)

**Schema.** The `state_schema.json` streams are identical to 27 km in names, offsets, dims, types and
variable lists:
- climate 0-240, landuse 240-273, hyde 273-276, bui 276-283, soil 283-299, elevation 299-302;
- 302 channels in total;
- years 1882-2025, sampled from 1902;
- 124 year files;
- 535,432 training vectors.

**NaN on land.** 0 in every stream and every year. Every 54 km land cell has climate sub-points: min 32,
p05 71, median 100.

**Same-rule identities.** Each 54 km value is compared with the 2x2 child aggregate. None is float-exact:

| Family | rms diff / sd | max diff / sd | corr |
|---|---|---|---|
| LUH fractions (14) | 0.018-0.094 | 0.34-3.0 | 0.9957-0.9998 |
| LUH plain-rule copies of the weighted layers | 0.027-0.085 | | ≥0.9964 |
| Soil | 0.021-0.038 | ≤0.59 | ≥0.9993 |
| HYDE density | 0.16 | 10.5 | 0.987 |
| HYDE rural / urban counts | 0.04 / 0.03 | | 0.9992 / 0.9995 |

- HYDE totals over land: the 54 km sums are 0.22% (rural) and 0.19% (urban) above the summed children.
- Restricting to cells where all four children are finite or land changes little, so coast and NaN
  handling is not the cause.
- Mechanism (likely, not tested): GDAL's `average` / `sum` weight the source pixels covering each target
  cell's bounding box in source coordinates. That box is not the cell itself (Albers cells are rotated
  relative to lat/lon and Goode), so the children's footprints do not tile the parent's. This also fits
  sums slightly above the children.

**Exact lattice identities.**

*Elevation.*
- The fine DEM lattices at 54/30 and 27/15 are identical: max diff 0, no NaN mismatch.
- The stored 54 km quantiles equal the quantiles of the union of the 27 km lattice exactly (max diff 0) in
  rows 0-65.
- All differences, and every land cell outside the children's range (7 / 6 / 6 for q10 / q50 / q90), are
  in row 66. Row 66 is the 54 km row padded south of the 27 km box: its 13 land cells natively include
  terrain that is outside the 27 km domain.

*BUI.*
- `bui_avail` equals the children's mean exactly (max diff 0).
- Quantiles inside the covered children's range:
  - q05-q50: 0 outside;
  - q75 / q90: 4 of 82k;
  - q99: 115 of 82k (0.14%).
- Mechanism (likely): with linear interpolation, the union's quantile can step one order statistic past
  the children's range in heavy tails.

*Climate.*
- Checked over 25 years (stride 5, 1902-2022), all 240 channels.
- Median fraction outside the children's range: 0; maximum 0.18% (q10 temperatures), max excess 0.35
  units. Mechanism (likely): the same interpolation effect, with 25 points per child.

**LUH correction (weighted vs plain, at 54 km, land cells, 1882-2024).**

| Layer | mean, weighted vs plain | corr | cells differing by > 10% sd |
|---|---|---|---|
| fertl_c3per | 27.7 vs 13.9 | 0.78 | 21% |
| fertl_c4per | 7.1 vs 2.1 | 0.72 | |
| fertl_c4ann | 47.6 vs 35.6 | 0.87 | |
| fertl_c3ann | 28.8 vs 26.3 | 0.97 | |
| irrig_* | about 0.045-0.050 vs 0.035 | 0.90-0.95 | |
| flood | | 0.95 | 0.6% of cells; max 21 sd |
| secma | | 0.83 | 24% |
| secmb | | 0.98 | 19% |

- For fertl_c3per, rms diff / sd is 1.08.
- In the LUH source, all five irrig_* layers are identical. Only the parent weighting makes them differ.

**States vs the 2x2 block mean of the 27 km states** (per channel, all years x land):

| Stream | min corr | median corr |
|---|---|---|
| climate | 0.992 (PAS / NFFD) | 0.9994 |
| soil | 0.9993 | |
| elevation | 0.994 | |
| hyde | 0.977 (density) | |
| landuse, unweighted | ≥0.996 | |
| landuse, weighted fertl / secma | 0.70-0.84 (by design) | |
| bui | q05 0.43, q25 0.73, q50 0.85, q90 0.96 | |

- landuse fharv: the correlation is reported as -1 / NaN because the layer is constant (float noise).
- BUI mechanism (established for fully covered land): there the grid-level correlation of log1p values
  is still 0.70 (q05) and 0.90 (q50). A low quantile of the union is not the mean of the children's low
  quantiles: one urban child lifts the mean but not the union. That is the reason for building natively.
  The out-of-coverage fill change adds to the difference.

## Jobs

- My normal-queue chain was cancelled while still pending: 3541853-58, submitted 2026-10-29 22:05. The
  same scripts were rerun on `development`, by another session of this account.

| Job | Stage | Started | Wall time | Result |
|---|---|---|---|---|
| 3542248 | climate regroup | 2026-10-30 06:40 | 10:58 | 13.6 GB RSS |
| 3542249 | HYDE + LUH | | 1:58 | |
| 3542250 | elevation, soil, BUI | | 2:45 | |
| 3542271 | climate_grid | | 7:50 | |
| 3542272 | states | | 54:47 | 3.7 GB |
| 3542273 | checks | | | FAILED (27 km climate grids have no 2025 file) |
| 3542345 | checks | | | cancelled |
| 3542418 | checks | | 28:36 | completed, with the year-intersection fix |

## Not done / open

1. **Climate sub-point gap: CLOSED in v2 (section below).** In v1, sub-points existed only in 27 km land
   children, so 5% of land cells had ≤ 71 of 100 points. 182 non-land parents still carry values (4,813
   points); they are masked.
2. **Row 66 (recorded, no change).** The 13 land cells in the padded south row natively include terrain,
   soil, land use and BUI beyond the 27 km box. Their climate comes only from sub-points of 27 km rows
   ≤ 132, because the 27 km lattice has no row 133; the v2 supplement added 0 points in row 66. This is a
   mixed definition, kept as is.
3. **GDAL averaging is not exact.** Same-rule "identities" hold only to rms 2-9% of sd for reprojected
   families, and to 16% for HYDE density. If exact aggregation is required, an exact area-overlap operator
   would be needed at both resolutions.
4. **pltns_wdprd parent** is a proxy; moot here because the layer is 0.
5. **27 km inconsistency.** The 27 km LUH per-parent channels still use the plain rule; the twin's 27 km
   comparison carries it.
6. **Climate year coverage.** `climate_grid` writes 123/124 bio-years. 2025 is missing because climr ends
   in 2024; the streamer carries the last year forward flat, as at 27 km.

## v2 (2026-10-30): climate over ALL fine-land sub-points (coordinator request)

The v2 states are in **`/work/07980/bwa386/ls6/houfin/processed/field54/states_v2`**.
- The v1 states (`.../field54/states`) were not touched, since Stage D may be reading them.
- Switching to v2 is a deliberate config change: point `paths.hist_dir` at `states_v2`. The overlay
  `config/overlays/field54_states_v2.json` does exactly that, and points the climate stream at
  `climate_grid_monthly_v2`.

### 1. Enumerating the missing points

`subcell_centroids.build_subcell_centroids` was rerun with the 27 km run's settings:
- the same 27 km ref grid, `grid` = 5 (the 5.4 km fine lattice), the same ETOPO DEM;
- the same `rasterize_land_fine` filter (Natural Earth land minus lakes, CUB/BHS excluded);
- the parent mask, and only the parent mask, changed: from "27 km cell is land" to "54 km parent is land".

The regeneration matches the stored set:
- Restricted to 27 km land cells, the regenerated set has 417,777 points. This equals the stored
  422,590 minus the 4,813 points whose 54 km parent is not land.
- Ids are fine-lattice indices, consistent for both the stored and the new points.

The missing points:
- **2,082 sub-points**, in 285 27 km ocean cells, inside **219 54 km land cells**. None is in row 66.
- In those 219 cells, the sub-point count rises from min 32 / median 62 to min 50 / median 68.
- After the fix, all land cells have at least 50 points (p05 78, median 100).

Cost estimate: the 27 km run took about 0.13 core-s per point (about 370k points in 377 s on 16 R
processes x 8 threads). For 2,082 points that is about 5 core-minutes plus R start-up, far under 6 h.

### 2. Running climr

`research/exp/field54_climr_extra.py` runs `climatena._run_chunk`, the same code path as the 27 km run:
- obs_years 1901:2024, `cru.gpcc`, `db_option=local`, the same Rscript and climr 0.2.2, the 16 x 8
  layout, the same monthly variables.
- Output went to a NEW chunk dir, `$HOUFIN_DATA54/climate_extra/_chunks` (chunk ids 100-151). The 27 km
  `_chunks` were not touched.

The compute node has no internet, so every chunk had to hit the existing climr cache:
- The points were grouped so that each chunk's bounding box lies inside one already-cached refmap tile
  and one cached obs_ts area (`chunk_manifest.json`).
- All 52 chunks logged "Retrieving from cache". Nothing was downloaded and the shared cache was not
  modified.

Result: job 3542727, 83 s of climr; 258,168 rows (2,082 points x 124 periods), with 0 NaN.

### 3. Regroup, grids and states

- **Regroup.** `climate_regroup` now accepts several `--chunks` dirs and unions them. A new test covers
  this; the field54 tests are 4 passed, locally and on TACC.
  - Inputs: the 27 km `_chunks` plus `climate_extra/_chunks`.
  - Result: 424,672 sub-points (= 422,590 + 2,082); 456 parents straddle chunks.
  - Output: `$HOUFIN_DATA54/climate_v2`.
- **Grids.** `climate_grid` wrote to `climate_grid_monthly_v2`: 240 channels, 123/124 bio-years, as in v1.
- **States.** `build_states` with `field54_states_v2.json` wrote `states_v2` (job 3542912, 1:03:52). The
  whole states set is rebuilt; build_states has no per-stream mode.

### Results

**v2 vs v1, 54 km states** (`research/exp/field54_climate_v2_compare.py`, job 3542913,
`processed/field54/field54_climate_v2_compare.json`):
- **219 of 4,318 land cells changed**; all 219 affected cells changed.
- Everything else is bit-identical: every stream outside the affected cells (max diff 0), and every
  non-climate stream inside them (max diff 0).
- Climate channels over affected cells x 124 years:

  | Measure | min | median | max |
  |---|---|---|---|
  | corr(v1, v2) | 0.9928 (Tmax_b12m07_q10) | 0.9996 | |
  | rms change / channel sd | | 0.032 | 0.12 (Tmax_b12m07_q10) |
  | max change / channel sd | | 0.26 | 2.48 (PAS_b11m06_q50: 4.8 mm snow) |

  No channel has corr below 0.99.
- Mechanism (likely): the added points are coastal fine-land points, so the change is largest in
  q10/q90 temperatures and snow.

**Checks on v2** (`field54_checks_v2.json`, climate + states sections, climate every 5th year):
- Mask identical.
- Schema identical to 27 km (302 channels).
- 0 NaN on land, in any stream or year.
- Climate state corr with the 27 km block mean: min 0.9920, median 0.9994 (v1: 0.9921 / 0.9994).

**Within the children's range.** Overall this now reports a median of 0.5% outside and a maximum of 3.4%
(v1: 0 / 0.18%). That is expected rather than an error: in the affected cells the union contains points
whose 27 km parents are ocean and have no 27 km value, so the 27 km land children's range no longer
bounds the cell. Split by cell (4 years x 240 channels):

| Cells | outside | max excess |
|---|---|---|
| unaffected | 0.007% | 0.14 (unchanged from v1) |
| affected | 15% | 41.7 |

### Jobs

| Job | Stage | Wall time |
|---|---|---|
| 3542727 | climr supplement, regroup, climate_grid | 21:09 |
| 3542912 | states_v2 | 1:03:52 |
| 3542913 | compare + checks | 21:43 |

All three ran on `development`. Code changes:
- `src/data/preprocess/climate_regroup.py` (multi-dir `--chunks`; a single dir behaves as before).
- New: `research/exp/field54_climr_extra.py`, `research/exp/field54_climate_v2_compare.py`,
  `research/field54/{f,g,h}_*.slurm`, `config/overlays/field54_states_v2.json`.
- `field54_checks.py` gained `--clim-dir` / `--clim-grid`.
- TACC file hashes: `SNAPSHOT_MANIFEST_climate_v2.txt`.

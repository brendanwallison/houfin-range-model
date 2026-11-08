# Data sources — formats, resolutions, projections, cadence

Every external product, what it is when it lands, and what we keep. "Target" is
the model grid (equal-area Albers at `grid.target_res_m`; see
[TEMPORAL.md](TEMPORAL.md) for the time axis). Acquire modules live in
`src/data/acquire/`, preprocess in `src/data/preprocess/`, and paths/knobs in
`config/data_config.json`.

| Product | Access (acquire) | Format | Native res | Native CRS | Cadence | Covariates kept | → Target |
|---|---|---|---|---|---|---|---|
| **eBird S&T** | REST API (`ebird.py`) | GeoTIFF, 1-band | ~2.96 km | EPSG:8857 (Equal Earth) | weekly, 2023 (**opt-in**: only the legacy `trend.anchor_mode=weekly`) | `abundance_median` per species×week | reproject **average** → Albers grid |
| **eBird S&T *trends*** | same REST API, `--trends` (`ebird.py`) | Parquet (tabular, one row per cell) | 27 km | WGS84 centroids | 2012–2022 window | `abd_ppy` (%/yr) and `abd` (mid-window rel. abundance, **the production anchor**) | nearest/Voronoi → Albers grid (`ebird_trend.py`) |
| **BBS trend maps** | ScienceBase `67507ae5…`, DOI 10.5066/P1DPJPSI (`acquire/bbs.py`) | GeoTIFF `tr{AOU}.tif` | 27 km | **ESRI:102003** | 1966–2022 long-term | geometric-mean %/yr population change | **nearest clip/pad, zero resampling** — the ref grid is snapped to this lattice (`bbs_trend.py`) |
| **BBS abundance maps** | same item (`bbs_abundance`) | GeoTIFF `ra{AOU}.tif` | 27 km | ESRI:102003 | 2018–2022 mean | relative abundance (birds/route) | nearest clip/pad (`bbs_abund.py`) |
| **Climate (ClimateNA via `climr`)** | R `climr` over a DEM (`climatena.py`→`climate_climr.R`) | computed (GeoTIFF out) | downscaled to query pts | lon/lat in, Albers out | monthly, 1901→`end_year` | Tmin/Tmax/Tave/PPT (+ derived incl. DD\*/NFFD/CMD/Eref); **all 12 months kept as separate channels** (see TEMPORAL.md), ×3 elevation quantiles for temperatures / q50 only otherwise | built directly on Albers grid |
| **LUH-3** (v1.2 CMIP7 hist.) | Zenodo `19261724` (`zenodo.py`) | netCDF4 | 0.25° (~28 km) | WGS84 geographic | annual, 850–2024 | `states` (12 land-use fractions) + `management` | reproject → Albers (~1:1) |
| **HYDE 3.5** (baseline, apr2025) | Utrecht vault HTTP (`hyde.py`) | netCDF (per var) | 5′ (~9.3 km) | WGS84 geographic | annual time points near-present, to 2025 | popd, urban pop, rural pop | reproject **average** (density) / **sum** (counts) → Albers |
| **HISDAC-US BUI** | Harvard Dataverse `10.7910/DVN/CSLOJP` (`dataverse.py`) | GeoTIFF (float64, `nodata` unset) | 250 m | **EPSG:5070** (Albers, lat_0 23) | semi-decadal, 1810–2020 | indoor building gross area per pixel, ft² — **CONUS only** | reproject→fine sub-grid, block-quantile (6 bands) + `bui_avail` → Albers (`bui.py`) |
| **SoilGrids** (aggregated) | ISRIC HTTP (`soilgrids.py`) | COG GeoTIFF | 5000 m | **Goode Homolosine** ESRI:54052 | **static** | sand/silt/clay/phh2o/soc/bdod/cec/nitrogen × 2 depths | reproject **average** → Albers |
| **BBS US/Canada** | ScienceBase item `6a0b…` (`acquire/bbs.py`) | CSV (+ States.zip) | point routes | WGS84 (lat/lon) | annual, 1966–**2025** | House-Finch counts; `RunType`/`RPID` QC | rasterize to Albers land cells |
| **BBS Mexico** (unprocessed) | ScienceBase item `5f32…`, DOI 10.5066/P9L4KBDC | CSV | point routes | WGS84 (lat/lon) | annual, 2008–2018 | counts (`SpeciesData`), runs (`RouteData`), loc (`RouteDetails`) | rasterize; **no RunType/RPID** → quality covariate |
| **DEM** (elevation) | NOAA ETOPO 2022 HTTP (`dem.py`) | GeoTIFF | 60″ (~1.85 km) | WGS84 geographic | static | surface elevation → p10/p50/p90 per cell | reproject→fine sub-grid, block-quantile → Albers |
| **Land/water** | Natural Earth 10 m (`download_all.sh`) | vector (shapefile) | fine | WGS84 | static | coastline → land fraction | threshold τ → land mask |
| **AVONET + phylogeny** | figshare `16586228` `ELEData.zip` (`acquire/avonet.py`) | CSV + Nexus tree | species | — | static | traits, BirdLife/BirdTree crosswalk, Hackett MCC phylogeny | trait/phylo distance to house finch |
| **eBird taxonomy** | eBird API `ref/taxonomy/ebird` (`acquire/avonet.py`) | CSV | species | — | per taxonomy release | SPECIES_CODE ↔ scientific name crosswalk | join key for AVONET/urban |
| **Urban tolerance** | figshare `19182503` (`acquire/avonet.py`) | CSV | species | — | static | 6 urban-association/night-light indices | defines the species universe; reference-community ranking |

Not every product above enters the current model directly. The age model reads DESK latent
features built from these covariates (plus species data), and its likelihood uses the data sets
listed in the next section.

## Data entering the current model (candidate LA1, revised 2026-11-08)

Source: `docs/model_spec/MODEL_SPEC.md` §1, §4–5, `DESK_PROVENANCE.md` and
`RESOLVED_CONFIG.json`. Paths are relative to the 54 km processed root
(`${HOUFIN_PROCESSED}/grid54`, = `E:\Datasets\houfin\grid54` locally) unless given in full.

| Data | Role in the model | File / build | Size in the candidate |
|---|---|---|---|
| **DESK latent habitat Z** (FIELD54 production, v1 states) | habitat predictors H_ℓ for every vital rate and capacity | `field54prod_ema` s64 cube. The chain: native 54 km DESK (EMA output, half-life 10.9 yr) → z_assim field correction (`field_d`) → summer dims 0–63 (`stageF/encoder/cube_field54prod_ema_s64`) | 124 yr × 4,318 cells × 64; rank 64, no truncation |
| **Path (journey) features Z^d** | juvenile journey survival Q_k | **current: `inputs_shape/J200_established1951_vlF_nativeMX`** (V&L-faithful J200 kernel, FIELD54 land-share floor 1e-4; built by `research/exp/invasion_sandbox/build_vl.py --land_share_floor 1e-4`). **Superseded: `..._vl_nativeMX`** (same kernel, no land-share floor; differs at about 170 coastal cell-band pairs, maximum absolute difference .31) | 124 × 4,318 × 12 kernels × 64; the `_vlF` all-data refits (L0F1–L0F3) have landed and the change is inert (< 0.001 nats at fixed parameters) |
| **BBS US/Canada + Mexico** | NB2 counts, route effects Laplace-integrated | `bbs/bbs_data_obsroute_for_python.npz` → model inputs | 123,191 surveyed rows (US/Canada 1966–2025; 236 Mexico rows 2008–18, quality tier 1), 19,127 observer×routes; together with the 68,704 pseudo-zeros below, 191,895 BBS rows in all. 228 pre-1940 Mexican pseudo-zero rows were dropped for the native-Mexico hull |
| **BBS pseudo-zeros 1902–39** | NB2 zeros with route effect 0 | built into the model inputs (no route id) | 68,704 rows (1,808 cells × 38 yr). **Pending decision:** drop or keep (docs/PRIORS.md#observation-model) |
| **Range absences** | −log NB2(0) cost, t ≤ 1965 | `sandbox/ROUTEMARG/range_absence_halo400_nativeMX.npz` (`native_hull_build.py`, halo 400 km + native Mexico) | 74,306 cell-years |
| **NestWatch** | nest survival + fledgling counts vs realised fecundity | `nestwatch/nests_hofi_v6.npz` | 1,089 attempts, 330 cells; data span 1968–2020 (filter window 1902–2020) |
| **CBC counts 1902–65** | NB2 circle counts, latent effort, circle effects Laplace-integrated | `research/grid54/inputs/cbc_early_20261102/cbc_early.npz` (TACC path) | 10,511 circle-years, 1,823 circles |
| **BBL summer–winter pairs** | CBC migrant-destination kernel | `sandbox/ROUTEMARG/bbl_summer_winter_pairs.csv` | The kernel was estimated offline from these pairs (`p14_bbl_sw.py`), and its migrant shares, distances and bearing are hard-coded in `cbc_obs.py`: East LogNormal(log 265 km, .83), bearing 218°; others LogNormal(log 337 km, .83). Only κ_bear (6.24 at LA1) is computed from the CSV at run time |
| **BBL adult dead recoveries** | m-array survival likelihood (with disease) | `sandbox/R83_bbl/bbl_adult_marray.npz` | 8,778 cohorts, 543 recoveries, to 2025 |
| **GBIF museum specimens + Aves effort** | NB2 specimen counts per cell-decade | `sandbox/ROUTEMARG/specimens_cache_collectors.npz` (collector filter: 571 of 5,522 rows weight 0; rule not re-read [U]; decade window = mean over the decade's model years, 1880–1901 read against 1902) | 5,522 cell-decades (582 positive), 1880–1940 |
| **Native-viability regions** | one-sided penalty on 1902 potential growth | `sandbox/ROUTEMARG/native_viable_cells.npz` | Mexico 88 cells, US 257 cells |
| **Disease arrival map** | onset of the epizootic added mortality | `disease/disease_arrival_year_54km.tif` (`scripts/build_disease_arrival_map.py`) | per-cell year, 1994–2006 spread history |
| **Native seed shape I_0** | 1902 seed and native departure/disease terms | model-input `initpop_latent` (`_nativeMX`: + Mexican hull, 299 margin cells at .5) | 1,409 seeded cells (350 core) |

**Pending decision: FIELD54 v1 → v2 states.** The candidate's Z uses the interim **v1** states.
A rebuild on **`field54prod_v2_ema`** exists on TACC; it fixes coastal climate and changes 219
cells. Whether to move to it is *pending* with the user. Moving would change Z (and hence Z^d),
so every fit would need re-running.

**Reproducibility notes** (DESK_PROVENANCE §c–§d).
- DESK training code (the frozen `coord_code`) exists only on TACC.
- The input metadata's `dispersal_spec` misdescribes the V&L-faithful kernel. Only
  `KERNEL_DISCRETISATION.json` in the input directory records it.
- The overlay's `path_features.input_dir` (`cube_s64_grid54`) is stale and is not the Z that runs.

## Aggregation method by native:target ratio

The model grid is **not** an integer multiple of every native resolution, so the aggregation
method is chosen per product. The base `config/data_config.json` grid is 27 km
(`grid.target_res_m = 27000`), and the worked ratios below were written for it. The **current
candidate runs on the 54 km grid**: `config/overlays/grid54_data.json` sets
`target_res_m = 54000`, giving 67 × 112 cells and 4,318 land cells (ESRI:102003). Its habitat
features are built natively at 54 km by FIELD54, not coarsened from 27 km (see "Data entering the
current model" above).

- **Integer ratio + quantiles needed** → `regrid.block_reduce` / block-quantile.
  The DEM takes this path: ETOPO 60″ (~1.85 km) is first reprojected to a
  `elevation.fine_factor = 15` sub-grid of the model grid, then block-quantiled, so the
  block factor is exact by construction rather than by luck of the native resolution.
  BUI takes the same path at `bui.fine_factor = 27`. Note 27000/250 = 108 exactly, so
  BUI *could* be block-reduced directly — but only in **its own** CRS on its own origin,
  which is not the model lattice. Reprojecting onto a ref sub-grid first is what makes the
  output alignable; the sub-grid, not the native ratio, is what makes the blocks nest.
- **Non-integer or ~1:1** → `rioxarray.reproject_match` with `Resampling.average`
  for continuous fields (eBird ~2.96 km, HYDE ~9.3 km, SoilGrids 5 km→27 km,
  LUH-3 0.25° ~1:1), `Resampling.nearest`/`mode` for categorical masks.

Rationale: `block_reduce` requires an integer block factor; `reproject_match`
handles arbitrary ratios while `Resampling.average` is the linear areal mean (the
deferral-safe aggregate — apply any nonlinear transform *after*, at target res).

## Assumptions that are validated at runtime

These were previously hardcoded/unchecked; the code now asserts them and fails
loudly on mismatch rather than silently mis-ingesting:

- eBird raster CRS actually equals EPSG:8857 (not blindly `write_crs`).
- SoilGrids native CRS is Goode Homolosine (must be reprojected, not assumed
  Albers).
- Ocean/land mask band count.
- BBS quality fields: US/Canada carry `RunType` and `RPID`. Mexico carries `RPID`
  but **not** `RunType`, so it cannot be screened the same way and enters as an
  unscreened tier (→ quality covariate, not a protocol filter).
- Mexico `SpeciesData.csv` ships stop 46 with a damaged header (`Sto 46`). The parser
  absorbs it and **refuses a gap** in the stop numbering rather than summing 49 of 50,
  which would undercount invisibly.
- BUI cannot report its own extent: `nodata` is unset and ocean, Canada, Mexico and
  genuinely unbuilt CONUS land are all exactly `0.0`. Absence is therefore established
  from the Natural Earth admin-0 polygon (read as a USA *inclusion*), and `bui.py` fails
  rather than proceeding if that polygon is missing.

## Provenance / licensing

eBird S&T (Cornell, access-key terms); ClimateNA/`climr` (CC-BY, bcgov);
LUH-3 (CC-BY, Zenodo); HYDE 3.5 (CC-BY 3.0, Utrecht/PBL); SoilGrids (CC-BY,
ISRIC); HISDAC-US BUI (CC0, Harvard Dataverse — cite Ahn, Leyk, Uhl & McShane 2023
and Leyk & Uhl 2018 *Sci. Data* 5:180175); BBS (USGS public domain);
DEM/coastline (public).

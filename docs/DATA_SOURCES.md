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

## Aggregation method by native:target ratio

The model grid (27 km, `grid.target_res_m = 27000`) is **not** an integer multiple of every native resolution,
so the aggregation method is chosen per product:

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
- BBS quality fields present for US/Canada (`RunType`,`RPID`); absent for Mexico
  (→ quality covariate, not a protocol filter).
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

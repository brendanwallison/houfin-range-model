# Data sources — formats, resolutions, projections, cadence

Every external product, what it is when it lands, and what we keep. "Target" is
the model grid (equal-area Albers at `grid.target_res_m`; see
[TEMPORAL.md](TEMPORAL.md) for the time axis). Acquire modules live in
`src/data/acquire/`, preprocess in `src/data/preprocess/`, and paths/knobs in
`config/data_config.json`.

| Product | Access (acquire) | Format | Native res | Native CRS | Cadence | Covariates kept | → Target |
|---|---|---|---|---|---|---|---|
| **eBird S&T** | REST API (`ebird.py`) | GeoTIFF, 1-band | ~2.96 km | EPSG:8857 (Equal Earth) | weekly, 2023 | `abundance_median` per species×week | reproject **average** → Albers grid |
| **Climate (ClimateNA via `climr`)** | R `climr` over a DEM (`climatena.py`→`climate_climr.R`) | computed (GeoTIFF out) | downscaled to query pts | lon/lat in, Albers out | monthly, 1901→`end_year` | Tmin/Tmax/Tave/PPT (+ derived), ×3 elevation quantiles | built directly on Albers grid |
| **LUH-3** (v1.2 CMIP7 hist.) | Zenodo `19261724` (`zenodo.py`) | netCDF4 | 0.25° (~28 km) | WGS84 geographic | annual, 850–2024 | `states` (12 land-use fractions) + `management` | reproject → Albers (~1:1) |
| **HYDE 3.5** (baseline) | Utrecht vault HTTP (`hyde.py`) | netCDF (per var) | 5′ (~9.3 km) | WGS84 geographic | annual time points, to 2023 | popd, urban pop, rural pop | reproject **average** (density) / **sum** (counts) → Albers |
| **SoilGrids** (aggregated) | ISRIC HTTP (`soilgrids.py`) | COG GeoTIFF | 5000 m | **Goode Homolosine** ESRI:54052 | **static** | sand/silt/clay/phh2o/soc/bdod/cec/nitrogen × 2 depths | reproject **average** → Albers |
| **BBS US/Canada** | ScienceBase item `6a0b…` (`acquire/bbs.py`) | CSV (+ States.zip) | point routes | WGS84 (lat/lon) | annual, 1966–**2025** | House-Finch counts; `RunType`/`RPID` QC | rasterize to Albers land cells |
| **BBS Mexico** (unprocessed) | ScienceBase item `5f32…`, DOI 10.5066/P9L4KBDC | CSV | point routes | WGS84 (lat/lon) | annual, 2008–2018 | counts (`SpeciesData`), runs (`RouteData`), loc (`RouteDetails`) | rasterize; **no RunType/RPID** → quality covariate |
| **DEM** (elevation) | TBD (GMTED2010 / MERIT) | GeoTIFF | ~1 km | TBD | static | elevation → p10/p50/p90 per cell | block-quantile → Albers |
| **Land/water** | Natural Earth / GSHHG | vector | fine | WGS84 | static | coastline → land fraction | threshold τ → land mask |

## Aggregation method by native:target ratio

The model grid (25 km) is **not** an integer multiple of every native resolution,
so the aggregation method is chosen per product:

- **Integer ratio + quantiles needed** → `regrid.block_reduce` / block-quantile
  (BUI 250 m→25 km = 100; DEM 1 km→25 km = 25).
- **Non-integer or ~1:1** → `rioxarray.reproject_match` with `Resampling.average`
  for continuous fields (eBird 4 km/~2.96 km, HYDE ~9.3 km, SoilGrids 5 km→25 km,
  LUH-3 0.25° ~1:1), `Resampling.nearest`/`mode` for categorical masks.

Rationale: `block_reduce` requires an integer block factor; `reproject_match`
handles arbitrary ratios while `Resampling.average` is the linear areal mean (the
deferral-safe aggregate — apply any nonlinear transform *after*, at target res).

## Assumptions that are validated at runtime

These were previously hardcoded/unchecked; the code now asserts them and fails
loudly on mismatch rather than silently mis-ingesting:

- eBird raster CRS actually equals EPSG:8857 (not blindly `write_crs`).
- PRISM/legacy netCDF variable name + CRS.
- SoilGrids native CRS is Goode Homolosine (must be reprojected, not assumed
  Albers).
- Ocean/land mask band count.
- BBS quality fields present for US/Canada (`RunType`,`RPID`); absent for Mexico
  (→ quality covariate, not a protocol filter).

## Provenance / licensing

eBird S&T (Cornell, access-key terms); ClimateNA/`climr` (CC-BY, bcgov);
LUH-3 (CC-BY, Zenodo); HYDE 3.5 (CC-BY 3.0, Utrecht/PBL); SoilGrids (CC-BY,
ISRIC); BBS (USGS public domain); DEM/coastline (public).

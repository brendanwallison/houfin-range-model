# Deprecated data products (PRISM + HISDAC-US BUI)

**Status: deprecated, kept runnable for old-vs-new comparison.** These are the
original **CONUS-only** environmental covariates. They forced the model grid to
CONUS and silently dropped Canada/Mexico observations at the border — the reason
the pipeline moved to continental products (ClimateNA/`climr`, LUH-3, HYDE,
SoilGrids) at 25 km with a de-dilated coastline. See the top-level plan and
`docs/DATA_SOURCES.md`.

## What lives here

| Module | Replaced by |
| --- | --- |
| `acquire/prism.py` | `acquire/climatena.py` (`climr`) |
| `preprocess/prism.py` | climate written straight to the model grid by `climate_climr.R` |
| `preprocess/bui.py` | `preprocess/luh3.py` + `preprocess/hyde.py` |
| `preprocess/ocean_mask.py` (BUI-NaN ocean rule) | `preprocess/land_mask.py` (continental land fraction) |
| `preprocess/watermask.py` (BUI-derived) | `preprocess/land_mask.py` |
| `combine/states.py` (hardcoded PRISM+BUI 2-stream) | `combine/streams.py` (config-driven N-stream registry) |

Thin launchers are under `scripts/deprecated/`.

## Why the BUI ocean rule was wrong

HISDAC-US BUI encodes ocean as **0**, not nodata, so *every* cell reads as
finite — a BUI-derived ocean mask cannot distinguish sea from unbuilt land.
`land_mask.py` instead rasterizes a continental land/water source (Natural
Earth) and thresholds a per-cell land fraction. See `land_mask.py`.

## Running the deprecated pipeline (comparison only)

These modules are unchanged from the 16 km / 4 km-BUI-grid era and assume that
grid. To A/B against the new pipeline, run them under a config whose
`grid.target_res_m` and product paths point at the old 4 km BUI grid (they use
integer `block_reduce`, which requires the target to be an integer multiple of
the native resolution — true for the 4 km/16 km grid, not for 25 km). The shared
encoder (`MultiStreamAutoencoder`), `model_inputs`, and cube code are **not**
duplicated here; only the product-specific code is preserved.

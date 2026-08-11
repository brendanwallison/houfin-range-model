# HISDAC-US BUI — not currently wired

**Status: deferred, not retired.** BUI (built-up intensity, HISDAC-US) was dropped from
the covariate set because it is **US-only**, while the model grid spans CONUS + southern
Canada. The intent is to bring it back so the model can choose between HYDE figures, which
exist everywhere, and BUI figures where they are available — so this code is kept
deliberately, not as an archive.

Nothing on the active pipeline imports it. It is not in `pipeline.sh`, `download_all.sh`,
or any `STAGES` default.

## What lives here

| Module | Role | Its continental counterpart |
| --- | --- | --- |
| `preprocess/bui.py` | BUI snapshots → model-grid rasters, with quantile bands and optional year interpolation | `preprocess/luh3.py` + `preprocess/hyde.py` |
| `preprocess/watermask.py` | water mask derived from BUI/HBUI | `preprocess/land_mask.py` |

Acquisition is **not** here: BUI comes from `src/data/acquire/dataverse.py`
(`scripts/download_dataverse.py`), which is also currently unwired.
Thin launchers for the two modules above are under `scripts/deprecated/`.

## How BUI would come back

Add a streamer entry to `states.streams` in `config/esk_desk_config.json`, pointing at the
grid dir `preprocess/bui.py` writes. The stream registry in `src/data/combine/streams.py`
is config-driven and stream-agnostic, so no ETL rewrite is needed.

The old hardcoded PRISM+BUI two-stream ETL (`combine/states.py`) is **not** the path back
and has been deleted: it could not run at any committed config (it resolved
`block_factor(4000, 27000)`, ratio 6.75, and raised before reading data) and it hardcoded
`START_YEAR`/`END_YEAR`/`EMA_TAU`, which `docs/TEMPORAL.md` forbids.

A caveat to carry: these modules date from the 4 km / 16 km BUI-grid era and use integer
`block_reduce`, which requires the target resolution to be an integer multiple of the
native one. That holds for 4 km → 16 km; it does **not** hold for the current 27 km grid,
so the aggregation path needs revisiting (`regrid.reproject_to_ref` is the modern route)
before BUI can be re-ingested.

## Why the BUI ocean rule was wrong

HISDAC-US BUI encodes ocean as **0**, not nodata, so *every* cell reads as finite — a
BUI-derived ocean mask cannot distinguish sea from unbuilt land. `land_mask.py` instead
rasterizes a continental land/water source (Natural Earth) and thresholds a per-cell land
fraction. The BUI-NaN ocean-rule builder that relied on this (`preprocess/ocean_mask.py`)
has been deleted; `land_mask.py` is the sole producer of `ocean_mask_{res}km.tif`.

## Also removed

The PRISM climate modules (`acquire/prism.py`, `preprocess/prism.py`) are gone. Climate now
comes from ClimateNA/`climr` via `acquire/climatena.py` and is written straight to the model
grid; see `docs/DATA_SOURCES.md`.

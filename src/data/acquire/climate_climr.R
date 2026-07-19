#!/usr/bin/env Rscript
# Downscale monthly observed climate at model-grid cell centroids, for three
# representative elevations per cell (p10/p50/p90 from preprocess/elevation.py),
# using climr (pure-R ClimateNA downscaling; Linux/HPC-native, no Windows .exe).
#
# Reads cell_centroids.csv (id,row,col,long,lat,elev_q10,elev_q50,elev_q90) and
# writes one long-format CSV per elevation level: climate_<lvl>.csv with columns
# id, PERIOD (year) and the monthly variables. The Python side joins these back
# onto the model grid. climr fetches reference surfaces from its remote backend
# and caches locally, so pre-populate the cache on a networked node before an
# offline run.
#
# Usage:
#   Rscript climate_climr.R <centroids.csv> <out_dir> <start_year> <end_year>

suppressMessages({
  library(climr)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("usage: climate_climr.R <centroids.csv> <out_dir> <start_year> <end_year> [obs_ts_dataset] [nthread]")
centroids_csv <- args[[1]]; out_dir <- args[[2]]
start_year <- as.integer(args[[3]]); end_year <- as.integer(args[[4]])
# CRITICAL: obs_years alone returns ONLY the 1961-1990 reference normal (PERIOD
# "1961_1990"); climr pulls the annual observed *time series* only when an
# observed dataset is named. cru.gpcc = CRU TS temp + GPCC precip, global,
# ~1901-present (matches the model timeline). Overridable via the 5th arg.
obs_ts_dataset <- if (length(args) >= 5) args[[5]] else "cru.gpcc"
# nthread = climr's in-process parallelism over the point table. The Python driver
# runs FEW processes with many threads each (one DuckDB handle per process, threads
# split the points) rather than many single-threaded processes contending on the
# shared DB. data.table matches nthread; oversubscription is bounded by the driver.
nthread <- if (length(args) >= 6) as.integer(args[[6]]) else 1L
setDTthreads(nthread)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cen <- fread(centroids_csv)
years <- start_year:end_year
monthly_vars <- list_vars("Monthly")   # 12 months x base vars (Tmin/Tmax/Tave/PPT + derived)

# climr 0.2.2 downscale() takes `...` and forwards it to downscale_(db_)core, which
# accept nthread (in-process parallelism over the point table). setDTthreads above
# threads climr's internal data.table ops too.
run_downscale <- function(xyz) {
  downscale(xyz, obs_years = years, obs_ts_dataset = obs_ts_dataset,
            vars = monthly_vars, return_refperiod = FALSE, nthread = nthread)
}

if ("elev" %in% names(cen)) {
  # SUBGRID mode: one true-elevation point per sub-point. Downscale all of them
  # once and write the per-sub-point series; the Python driver takes spatial
  # quantiles per parent cell (real horizontal + elevation variability).
  message(sprintf("Subgrid: downscaling %d sub-points (%d years, obs_ts=%s, nthread=%d)...",
                  nrow(cen), length(years), obs_ts_dataset, nthread))
  xyz <- data.frame(lon = cen$long, lat = cen$lat, elev = cen$elev, id = cen$id)
  fwrite(run_downscale(xyz), file.path(out_dir, "climate_points.csv"))
  message("Done. Wrote climate_points.csv (per sub-point) to ", out_dir)
} else {
  # ELEV_QUANTILE mode: centroid at three representative elevations per cell.
  for (lvl in c("q10", "q50", "q90")) {
    message(sprintf("Downscaling elevation level %s (%d cells, %d years, obs_ts=%s, nthread=%d)...",
                    lvl, nrow(cen), length(years), obs_ts_dataset, nthread))
    xyz <- data.frame(
      lon  = cen$long, lat = cen$lat,
      elev = cen[[paste0("elev_", lvl)]],
      id   = cen$id
    )
    fwrite(run_downscale(xyz), file.path(out_dir, sprintf("climate_%s.csv", lvl)))
  }
  message("Done. Wrote climate_{q10,q50,q90}.csv to ", out_dir)
}

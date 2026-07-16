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
if (length(args) < 4) stop("usage: climate_climr.R <centroids.csv> <out_dir> <start_year> <end_year>")
centroids_csv <- args[[1]]; out_dir <- args[[2]]
start_year <- as.integer(args[[3]]); end_year <- as.integer(args[[4]])
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cen <- fread(centroids_csv)
years <- start_year:end_year
monthly_vars <- list_vars("Monthly")   # 12 months x base vars (Tmin/Tmax/Tave/PPT + derived)

for (lvl in c("q10", "q50", "q90")) {
  message(sprintf("Downscaling elevation level %s (%d cells, %d years)...",
                  lvl, nrow(cen), length(years)))
  xyz <- data.frame(
    lon  = cen$long, lat = cen$lat,
    elev = cen[[paste0("elev_", lvl)]],
    id   = cen$id
  )
  ds <- downscale(xyz, obs_years = years, vars = monthly_vars, return_refperiod = FALSE)
  fwrite(ds, file.path(out_dir, sprintf("climate_%s.csv", lvl)))
}
message("Done. Wrote climate_{q10,q50,q90}.csv to ", out_dir)

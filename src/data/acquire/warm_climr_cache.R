#!/usr/bin/env Rscript
# Download + cache climr's reference map and observed time-series rasters for the
# region's bounding box, WITHOUT downscaling. Run on a networked node (login).
#
# Why not just run downscale() to warm the cache: downscale(db_option="local")
# loads/merges the continent-scale 800m reference raster into memory, which OOMs a
# memory-capped login node ("std::bad_alloc"). The input_*() functions only fetch
# the raster files into the cache (terra keeps them disk-backed), so this stays
# low-memory. Compute nodes then read the warmed cache offline via
# downscale(db_option="local"), where the 251 GB of RAM handles the processing.
#
# Usage: warm_climr_cache.R <centroids.csv> <obs_ts_dataset> <start_year> <end_year>
suppressMessages({
  library(climr)
  library(terra)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("usage: warm_climr_cache.R <centroids.csv> <obs_ts_dataset> <start_year> <end_year>")
centroids_csv <- args[[1]]
obs_ts_dataset <- args[[2]]
start_year <- as.integer(args[[3]]); end_year <- as.integer(args[[4]])

cen <- fread(centroids_csv)
elev_col <- if ("elev" %in% names(cen)) "elev" else "elev_q50"
xyz <- data.frame(lon = cen$long, lat = cen$lat, elev = cen[[elev_col]], id = cen$id)
bb <- get_bb(xyz)

message(sprintf("Caching refmap_climr for bbox (%d points define the extent)...", nrow(xyz)))
invisible(input_refmap(bb, reference = "refmap_climr"))

message(sprintf("Caching obs time series %s for %d-%d...", obs_ts_dataset, start_year, end_year))
invisible(input_obs_ts(dataset = obs_ts_dataset, bbox = bb, years = start_year:end_year, cache = TRUE))

message("climr cache warmed (refmap + ", obs_ts_dataset, "). Compute nodes can now run offline.")

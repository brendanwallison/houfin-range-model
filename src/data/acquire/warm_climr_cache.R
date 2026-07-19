#!/usr/bin/env Rscript
# Warm the climr cache for the study region WITHOUT downscaling or an in-memory
# merge. Run on a networked node (login).
#
# The genuine download/processing split: climr's pre_cache() downloads the
# high-resolution reference map "all at once ... using server-side clipping via
# gdalwarp, written directly to disk without merging into memory" (its own docs) --
# so the RAM-heavy 800m refmap merge that OOMs a memory-capped login node never
# happens here; it's deferred to the compute node (251 GB) reading this cache
# offline. pre_cache() doesn't cover the observed series, but that data is coarse
# (~0.5 deg) and small, so input_obs_ts() caches it without trouble.
#
# Usage: warm_climr_cache.R <centroids.csv> <obs_ts_dataset> <start_year> <end_year> [tiles(ignored)]
suppressMessages({
  library(climr)
  library(terra)
  library(data.table)
})
setDTthreads(1L)   # login ulimit: keep single-threaded (GDAL threads pinned via env)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("usage: warm_climr_cache.R <centroids.csv> <obs_ts_dataset> <start_year> <end_year>")
centroids_csv <- args[[1]]
obs_ts_dataset <- args[[2]]
start_year <- as.integer(args[[3]]); end_year <- as.integer(args[[4]])

cen <- fread(centroids_csv)
elev_col <- if ("elev" %in% names(cen)) "elev" else "elev_q50"
xyz <- data.frame(lon = cen$long, lat = cen$lat, elev = cen[[elev_col]], id = cen$id)
bb <- get_bb(xyz)

message("pre_cache(): downloading reference map for the region (server-side clip, no in-memory merge)...")
pre_cache(bbox = bb)

message(sprintf("input_obs_ts(): caching %s %d-%d (coarse ~0.5deg)...", obs_ts_dataset, start_year, end_year))
invisible(input_obs_ts(dataset = obs_ts_dataset, bbox = bb, years = start_year:end_year, cache = TRUE))

message("climr cache warmed (refmap via pre_cache + ", obs_ts_dataset, "). Compute reads it offline.")

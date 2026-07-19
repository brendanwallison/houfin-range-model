#!/usr/bin/env Rscript
# Download + cache climr's reference map and observed time-series rasters, TILED by
# geographic blocks, so no single tile's raster merge exceeds memory. Run on a
# networked node (login).
#
# Why tiled: even input_refmap() merges the requested bounding box's reference tiles
# into one raster in memory, so a continent-scale bbox OOMs a memory-capped login
# node ("std::bad_alloc"). Warming one small tile at a time (freed each iteration)
# keeps peak memory low. The tiling MUST match climatena.py's _split_centroids_spatial
# (same centroids + tiles-per-axis) so the cached bounding boxes match what the
# compute node requests offline via downscale(db_option="local").
#
# Usage: warm_climr_cache.R <centroids.csv> <obs_ts_dataset> <start_year> <end_year> [tiles_per_axis]
suppressMessages({
  library(climr)
  library(terra)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("usage: warm_climr_cache.R <centroids.csv> <obs_ts_dataset> <start_year> <end_year> [tiles_per_axis]")
centroids_csv <- args[[1]]
obs_ts_dataset <- args[[2]]
start_year <- as.integer(args[[3]]); end_year <- as.integer(args[[4]])
tiles_per_axis <- if (length(args) >= 5) as.integer(args[[5]]) else 8L

cen <- fread(centroids_csv)
elev_col <- if ("elev" %in% names(cen)) "elev" else "elev_q50"

# Same tiling as climatena.py: tile = (row %/% th) * tiles + (col %/% tw).
nrow_ <- max(cen$row) + 1L; ncol_ <- max(cen$col) + 1L
th <- as.integer(ceiling(nrow_ / tiles_per_axis)); tw <- as.integer(ceiling(ncol_ / tiles_per_axis))
cen[, tile := (row %/% th) * tiles_per_axis + (col %/% tw)]
tile_ids <- sort(unique(cen$tile))
message(sprintf("Warming %d geographic tiles (refmap + %s %d-%d)...",
                length(tile_ids), obs_ts_dataset, start_year, end_year))

for (i in seq_along(tile_ids)) {
  sub <- cen[tile == tile_ids[i]]
  xyz <- data.frame(lon = sub$long, lat = sub$lat, elev = sub[[elev_col]], id = sub$id)
  bb <- get_bb(xyz)
  invisible(input_refmap(bb, reference = "refmap_climr"))
  invisible(input_obs_ts(dataset = obs_ts_dataset, bbox = bb, years = start_year:end_year, cache = TRUE))
  message(sprintf("  tile %d/%d cached (%d points)", i, length(tile_ids), nrow(sub)))
}
message("climr cache warmed (tiled): refmap + ", obs_ts_dataset, ". Compute reads it offline.")

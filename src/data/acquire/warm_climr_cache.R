#!/usr/bin/env Rscript
# Warm the climr cache for the study region WITHOUT downscaling or an in-memory
# merge, TILED so each server-side clip/download is small. Run on a networked node.
#
# Two constraints, two fixes:
#  - MEMORY: input_refmap() merges the 800m refmap in RAM -> OOMs a login node.
#    climr's pre_cache() instead clips server-side via gdalwarp straight to disk
#    ("without merging into memory", per its docs), so no local merge.
#  - DOWNLOAD SIZE: clipping the whole continent bbox at once is a huge transfer
#    and times out ("<1 byte/sec"). So we pre_cache one geographic TILE at a time.
# The tiling MUST match climatena.py's _split_centroids_spatial (same centroids +
# tiles-per-axis) so the cached tile bounding boxes match what compute requests
# offline via downscale(db_option="local"). pre_cache() covers the reference map;
# input_obs_ts() caches the coarse (~0.5deg) observed series per tile.
#
# Usage: warm_climr_cache.R <centroids.csv> <obs_ts_dataset> <start_year> <end_year> [tiles_per_axis]
suppressMessages({
  library(climr)
  library(terra)
  library(data.table)
})
setDTthreads(1L)   # login ulimit: single-threaded (GDAL threads pinned via env)

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
message(sprintf("Warming %d geographic tiles (pre_cache refmap + %s %d-%d)...",
                length(tile_ids), obs_ts_dataset, start_year, end_year))

for (i in seq_along(tile_ids)) {
  sub <- cen[tile == tile_ids[i]]
  xyz <- data.frame(lon = sub$long, lat = sub$lat, elev = sub[[elev_col]], id = sub$id)
  bb <- get_bb(xyz)
  pre_cache(bbox = bb)                                                    # refmap, small clip, no merge
  invisible(input_obs_ts(dataset = obs_ts_dataset, bbox = bb,
                         years = start_year:end_year, cache = TRUE))      # coarse obs
  message(sprintf("  tile %d/%d cached (%d points)", i, length(tile_ids), nrow(sub)))
}
message("climr cache warmed (tiled: pre_cache refmap + ", obs_ts_dataset, "). Compute reads it offline.")

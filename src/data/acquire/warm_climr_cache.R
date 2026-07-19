#!/usr/bin/env Rscript
# Warm the climr cache for the study region WITHOUT the heavy raster re-encode,
# TILED so each server-side clip/download is small. Run on a networked node.
#
# Two constraints, two fixes:
#  - MEMORY / THREADS: climr's own cache builders (pre_cache, input_refmap) end in
#    terra::writeRaster(), which DECODES + RE-ENCODES the 73-band 800m refmap tile.
#    On a resource-capped TACC login node that write crashes the very first tile
#    (CPLCreateJoinableThread "Resource temporarily unavailable" -> CPLGetTLSList
#    fails -> segfault: the per-user cgroup can't spare a thread/stack for it).
#    But the file the server hands back is ALREADY a complete GeoTIFF, so we cache
#    it by a verbatim file COPY plus an in-place, header-only band rename
#    (terra::update(names=TRUE)) -- no pixel decode/encode, near-zero memory, no
#    GDAL thread pool. Validated to produce byte-identical downscale output to
#    pre_cache(). (input_obs_ts's coarse ~0.5deg rasters are tiny, so its own
#    writeRaster is cheap and left as-is.)
#  - DOWNLOAD SIZE: clipping the whole continent bbox at once is a huge transfer
#    and times out ("<1 byte/sec"). So we cache one geographic TILE at a time.
# The tiling MUST match climatena.py's _split_centroids_spatial (same centroids +
# tiles-per-axis) so the cached tile bounding boxes cover what compute requests
# offline via downscale(db_option="local").
#
# Even with the refmap re-encode gone, running all 64 tiles in ONE long-lived R
# process still crashes the login node: GDAL joinable threads from each tile's
# input_obs_ts() writeRaster accumulate (not reaped) until the per-user thread cap
# (ulimit -u ~300) is hit, ~tile 30. Fix: PROCESS ISOLATION -- the Python driver
# (--warm-cache) invokes this script once PER TILE (passing [tile_index]); each
# process warms one tile and exits, so the OS reaps its threads and none pile up.
# With no tile_index this loops all tiles (fine on a resourced node / locally).
#
# Usage: warm_climr_cache.R <centroids.csv> <obs_ts_dataset> <start_year> <end_year> [tiles_per_axis] [tile_index]
suppressMessages({
  library(climr)
  library(terra)
  library(data.table)
  library(httr)
  library(curl)
  library(uuid)
})
setDTthreads(1L)   # login ulimit: single-threaded (GDAL threads pinned via env)

# The 73 refmap band names climr assigns on download; band 73 must be "dem2_WNA"
# in the cached file because input_refmap does NOT rename on the cache-read path,
# and downscale indexes the elevation band by that name (res[, "dem2_WNA"]).
CLIPR_URL <- "http://146.190.244.244:8000/clipr"

# Cache the reference map for one bbox WITHOUT terra::writeRaster: server clips it,
# we copy the returned GeoTIFF verbatim into the cache and rename band 73 in place.
cache_refmap_light <- function(bb) {
  cPath <- file.path(cache_path(), "reference", "refmap_climr")
  dir.create(cPath, recursive = TRUE, showWarnings = FALSE)
  # Resume: skip if an already-cached tile fully covers this bbox.
  meta_f <- file.path(cPath, "meta_data.csv")
  if (file.exists(meta_f)) {
    m <- fread(meta_f)
    covered <- m[bb[1] >= xmin & bb[2] <= xmax & bb[3] >= ymin & bb[4] <= ymax]
    if (nrow(covered) > 0L) { message("  refmap already cached; skip clip"); return(invisible()) }
  }
  res <- GET(CLIPR_URL, query = list(rname = "climr_mosaic_clamped.tif",
             xmin = bb[1], xmax = bb[2], ymin = bb[3], ymax = bb[4]))
  url <- content(res)$url[[1]]
  tmp <- tempfile(fileext = ".tif")
  curl_download(url, tmp)                                    # streams to disk, low memory
  uid <- UUIDgenerate()
  dest <- file.path(cPath, paste0(uid, ".tif"))
  if (!file.copy(tmp, dest)) stop("failed to copy refmap clip into cache: ", dest)
  r <- rast(dest); names(r)[73] <- "dem2_WNA"
  update(r, names = TRUE)                                    # header-only rename, in place
  # Record the REQUESTED bbox, not ext() of the clip. climr clips to the DATA
  # extent, so an edge/coastal tile (DEM says land but climr's refmap is nodata
  # there) gets ext() < the requested bbox -> input_refmap decides "not fully
  # cached" and tries to fetch the rest, which crashes offline (pgGetTerra "Empty
  # tile - not enough data"). The process side requests exactly this bbox, so
  # recording it guarantees an offline cache hit; sub-points outside the refmap's
  # real data read back as NaN (correct for no-data regions; masked downstream).
  fwrite(data.table(uid = uid, ymax = bb[4], ymin = bb[3], xmax = bb[2], xmin = bb[1]),
         file = meta_f, append = TRUE)
  invisible(NULL)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("usage: warm_climr_cache.R <centroids.csv> <obs_ts_dataset> <start_year> <end_year> [tiles_per_axis] [tile_index]")
centroids_csv <- args[[1]]
obs_ts_dataset <- args[[2]]
start_year <- as.integer(args[[3]]); end_year <- as.integer(args[[4]])
tiles_per_axis <- if (length(args) >= 5) as.integer(args[[5]]) else 8L
tile_index <- if (length(args) >= 6) as.integer(args[[6]]) else NA_integer_  # NA = loop all tiles

cen <- fread(centroids_csv)
elev_col <- if ("elev" %in% names(cen)) "elev" else "elev_q50"

# Same tiling as climatena.py: tile = (row %/% th) * tiles + (col %/% tw).
nrow_ <- max(cen$row) + 1L; ncol_ <- max(cen$col) + 1L
th <- as.integer(ceiling(nrow_ / tiles_per_axis)); tw <- as.integer(ceiling(ncol_ / tiles_per_axis))
cen[, tile := (row %/% th) * tiles_per_axis + (col %/% tw)]

warm_one_tile <- function(sub, label) {
  xyz <- data.frame(lon = sub$long, lat = sub$lat, elev = sub[[elev_col]], id = sub$id)
  bb <- get_bb(xyz)
  cache_refmap_light(bb)                                                 # refmap: copy + header rename
  invisible(input_obs_ts(dataset = obs_ts_dataset, bbox = bb,
                         years = start_year:end_year, cache = TRUE))      # coarse obs (small write)
  message(sprintf("  %s cached (%d points)", label, nrow(sub)))
}

if (!is.na(tile_index)) {
  # PROCESS-ISOLATED: warm exactly this geographic tile, then exit (threads reaped).
  sub <- cen[tile == tile_index]
  if (nrow(sub) == 0L) { message(sprintf("  tile %d empty; skip", tile_index)); quit(status = 0) }
  warm_one_tile(sub, sprintf("tile %d", tile_index))
} else {
  # Single-process loop (resourced node / local): warm every non-empty tile.
  tile_ids <- sort(unique(cen$tile))
  message(sprintf("Warming %d geographic tiles (copy refmap + %s %d-%d)...",
                  length(tile_ids), obs_ts_dataset, start_year, end_year))
  for (i in seq_along(tile_ids)) {
    warm_one_tile(cen[tile == tile_ids[i]], sprintf("tile %d/%d", i, length(tile_ids)))
  }
  message("climr cache warmed (tiled: copy refmap + ", obs_ts_dataset, "). Compute reads it offline.")
}

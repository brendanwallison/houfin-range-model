# kissmig cross-check of SDM baseline B2 (pre-registered in research/experiments/SDM_BASELINE_KISSMIG_B2ALLEE_2026-10-30.md)
# Usage (repo root): Rscript research/r/kissmig_crosscheck.R <export dir>   (export from research/exp/sdm_b2_export.py)
.libPaths(c(normalizePath("research/r/lib"), .Library))
suppressPackageStartupMessages({ library(terra); library(kissmig) })
args <- commandArgs(trailingOnly = TRUE)
d <- if (length(args)) args[1] else "E:/Datasets/houfin/grid54/baselines/b0/kissmig"
gj <- paste(readLines(file.path(d, "grid.json"), warn = FALSE), collapse = "")
Ny <- as.integer(sub('.*"Ny": ([0-9]+).*', "\\1", gj)); Nx <- as.integer(sub('.*"Nx": ([0-9]+).*', "\\1", gj))
years <- 1902:2025
stopifnot(grepl('"years": \\[1902', gj))
rd <- function(f, L) {
  v <- readBin(file.path(d, f), what = "numeric", size = 4, n = L * Ny * Nx, endian = "little")
  r <- rast(nrows = Ny, ncols = Nx, nlyrs = L, xmin = 0, xmax = Nx, ymin = 0, ymax = Ny)
  values(r) <- matrix(v, ncol = L)
  r
}
na0 <- function(r) { r[is.na(r)] <- 0; r }
E <- list(static = na0(rd("E_static.f32", 1)), dyn = na0(rd("E_dyn.f32", length(years))))
S01 <- list(static = na0(rd("S01_static.f32", 1)), dyn = na0(rd("S01_dyn.f32", length(years))))
O_nat <- na0(rd("origin_native.f32", 1)); O_rel <- na0(rd("origin_release.f32", 1))
cells <- read.csv(file.path(d, "b2_front_cells.csv"))
cells$idx <- cells$row * Nx + cells$col + 1                   # terra cell index (row-major, 1-based)
ok <- cells[cells$ok == 1, ]
cat("ok cells", nrow(ok), "\n")

# first-occurrence YEAR per cell from one origin starting at year y0, k iterations per year
foc_year <- function(S, O, y0, k, pext, seed) {
  if (nlyr(S) > 1) S <- S[[which(years >= y0)]]
  nyr <- if (nlyr(S) > 1) nlyr(S) else (2025 - y0 + 1)
  it <- if (nlyr(S) > 1) k else k * nyr
  f <- kissmig(O, S, it = it, type = "FOC", pext = pext, pcor = 0.2, seed = seed, n_threads = 2)
  v <- values(f)[, 1]
  yr <- ifelse(v > 0, y0 + floor((v - 1) / k), NA)
  yr[values(O)[, 1] > 0] <- y0                                # origin cells occupied from the start
  yr
}
run_arm <- function(S, k, pext, nrep = 20) {
  m <- sapply(seq_len(nrep), function(s) pmin(foc_year(S, O_nat, 1902, k, pext, 1000 + s),
                                             foc_year(S, O_rel, 1940, k, pext, 2000 + s), na.rm = TRUE))
  apply(m, 1, function(x) if (all(is.na(x))) NA else median(x, na.rm = TRUE))
}
cohorts <- list(c(1966, 1974), c(1975, 1984), c(1985, 1994), c(1995, 2009))
stats <- function(yr, b2col) {
  lag <- yr[ok$idx] - ok$first_obs
  b2lag <- ok[[b2col]] - ok$first_obs
  coh <- sapply(cohorts, function(cc) median(lag[ok$first_obs >= cc[1] & ok$first_obs <= cc[2]], na.rm = TRUE))
  b2coh <- sapply(cohorts, function(cc) median(b2lag[ok$first_obs >= cc[1] & ok$first_obs <= cc[2]], na.rm = TRUE))
  c(median = median(lag, na.rm = TRUE), never = mean(is.na(lag)),
    c1 = coh[1], c2 = coh[2], c3 = coh[3], c4 = coh[4],
    spearman = suppressWarnings(cor(yr[ok$idx], ok[[b2col]], method = "spearman", use = "complete.obs")),
    mad_cell = median(abs(yr[ok$idx] - ok[[b2col]]), na.rm = TRUE),
    max_coh_diff = max(abs(coh - b2coh), na.rm = TRUE),
    b2_median = median(b2lag, na.rm = TRUE), b2c1 = b2coh[1], b2c2 = b2coh[2], b2c3 = b2coh[3], b2c4 = b2coh[4])
}
out <- list()
for (arm in c("static", "dyn")) {
  b2col <- paste0("b2_", arm, "_p50")
  for (k in 1:3) for (pext in c(0.05, 0.2)) {
    t0 <- Sys.time()
    yr <- run_arm(E[[arm]], k, pext)
    s <- stats(yr, b2col)
    out[[length(out) + 1]] <- data.frame(arm = arm, surface = "E", k = k, pext = pext, t(s))
    cat(arm, "E k", k, "pext", pext, "median", s["median"], "cohorts", round(s[c("c1", "c2", "c3", "c4")], 1),
        "spearman", round(s["spearman"], 3), "| B2", s["b2_median"], round(s[c("b2c1", "b2c2", "b2c3", "b2c4")], 1),
        format(Sys.time() - t0), "\n")
  }
}
res <- do.call(rbind, out)
# calibrated (k, pext) per arm = closest median to B2's own; then the secondary S01 surface at that setting
for (arm in c("static", "dyn")) {
  ra <- res[res$arm == arm, ]
  best <- ra[which.min(abs(ra$median - ra$b2_median)), ]
  yr <- run_arm(S01[[arm]], best$k, best$pext)
  s <- stats(yr, paste0("b2_", arm, "_p50"))
  res <- rbind(res, data.frame(arm = arm, surface = "S01", k = best$k, pext = best$pext, t(s)))
  cat("CALIBRATED", arm, "k", best$k, "pext", best$pext, "median", best$median, "cohorts",
      round(unlist(best[c("c1", "c2", "c3", "c4")]), 1), "spearman", round(best$spearman, 3), "max cohort diff",
      round(best$max_coh_diff, 1), "| S01 variant median", s["median"], "spearman", round(s["spearman"], 3), "\n")
}
write.csv(res, file.path(d, "kissmig_crosscheck.csv"), row.names = FALSE)
cat("versions: kissmig", as.character(packageVersion("kissmig")), "terra", as.character(packageVersion("terra")), "\nDONE\n")

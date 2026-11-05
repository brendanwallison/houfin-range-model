# B6: outside-our-code synthetic truths with rangr (pre-registered in research/experiments/SDM_BASELINE_B6_RANGR_2026-10-31.md)
# Usage (repo root): Rscript research/r/b6_rangr_truth.R <b6 dir> <truth: static|dyn> <seed>
.libPaths(c(normalizePath("research/r/lib"), .Library))
suppressPackageStartupMessages({ library(terra); library(rangr) })
args <- commandArgs(trailingOnly = TRUE)
d <- args[1]; truth <- args[2]; seed <- as.integer(args[3]); seeding <- if (length(args) >= 4) args[4] else "native"
set.seed(seed)
gj <- paste(readLines(file.path(d, "grid.json"), warn = FALSE), collapse = "")
Ny <- as.integer(sub('.*"Ny": ([0-9]+).*', "\\1", gj)); Nx <- as.integer(sub('.*"Nx": ([0-9]+).*', "\\1", gj))
years <- 1902:2025; nT <- length(years); RES <- 54
rd <- function(f, L) {
  v <- readBin(file.path(d, f), what = "numeric", size = 4, n = L * Ny * Nx, endian = "little")
  r <- rast(nrows = Ny, ncols = Nx, nlyrs = L, xmin = 0, xmax = Nx * RES, ymin = 0, ymax = Ny * RES, crs = "local")
  values(r) <- matrix(v, ncol = L); r
}
KAPPA <- 10; A_ALLEE <- if (length(args) >= 5 && args[5] == "noallee") NA else if (length(args) >= 5 && grepl("^A", args[5])) as.numeric(sub("^A", "", args[5])) else 2
KMEAN <- if (length(args) >= 6) as.numeric(args[6]) else 30; R_GROW <- if (length(args) >= 7) log(as.numeric(args[7])) else log(2); MAXD <- 540; FOUNDERS <- c(13, 13, 12, 12)
S <- if (truth == "static") rd("S_static.f32", 1) else rd("S_dyn.f32", nT)
K <- round(KAPPA * S)
native <- rd("native.f32", 1); release <- rd("release.f32", 1)
K1 <- K[[1]]
n1 <- K1 * (native > 0) * (seeding == "native"); n1[is.na(K1)] <- NA
t0 <- Sys.time()
mk <- function(n1map, Kmap) initialise(n1_map = n1map, K_map = Kmap, r = R_GROW, growth = "ricker", A = A_ALLEE,
                                       dens_dep = "K2N", border = "reprising", kernel_fun = "rexp", rate = 1 / KMEAN,
                                       max_dist = MAXD, progress_bar = FALSE, quiet = TRUE)
# segment 1: 1902-1939 (38 years); rangr: the first layer of N_map is time step 1 = 1902
i1 <- which(years <= 1939)
if (seeding == "native") {
  sd1 <- mk(n1, if (truth == "static") K1 else K[[i1]])
  s1 <- sim(sd1, time = length(i1), progress_bar = FALSE, quiet = TRUE)
  N1 <- s1$N_map                                                  # Ny x Nx x 38
  cat("segment 1 done", format(Sys.time() - t0), "extinct", s1$extinction, "\n")
} else {                                                          # release only: nothing alive before 1940
  N1 <- array(0, c(Ny, Nx, length(i1)))
}
# segment 2: 1940-2025 from the 1939 state plus founders
last <- N1[, , dim(N1)[3]]
start2 <- rast(K1); values(start2) <- as.vector(t(last))           # matrix (row-major) -> raster values
rel_idx <- which(values(release)[, 1] > 0)
v <- values(start2)[, 1]; v[rel_idx] <- v[rel_idx] + FOUNDERS[seq_along(rel_idx)]
v[is.na(values(K1)[, 1])] <- NA; values(start2) <- v                  # same NA cells as K (rangr requires it)
i2 <- which(years >= 1940)
sd2 <- mk(start2, if (truth == "static") K1 else K[[i2]])
s2 <- sim(sd2, time = length(i2), progress_bar = FALSE, quiet = TRUE)
N2 <- s2$N_map
cat("segment 2 done", format(Sys.time() - t0), "extinct", s2$extinction, "\n")
# rangr's first output layer is the start map; segment-2 layer 1 = 1940 start (1939 state + founders)
N <- array(0, c(Ny, Nx, nT)); N[, , i1] <- N1; N[, , i2[seq_len(dim(N2)[3])]] <- N2
# observation: binomial thinning at every design row, p matched to the real 2010-20 mean count over real rows
obs <- read.csv(file.path(d, "obs.csv"))
Nobs <- N[cbind(obs$row + 1, obs$col + 1, obs$year - 1901)]
Nobs[is.na(Nobs)] <- 0
m <- obs$real == 1 & obs$year >= 2010 & obs$year <= 2020
p <- min(1, mean(obs$y_real[m]) / max(mean(Nobs[m]), 1e-9))
y <- rbinom(length(Nobs), size = as.integer(round(Nobs)), prob = p)
cat("p", p, "mean real 2010-20", mean(obs$y_real[m]), "mean synthetic", mean(y[m]), "share zero real/syn",
    mean(obs$y_real[obs$real == 1] == 0), mean(y[obs$real == 1] == 0), "\n")
tag <- paste0(truth, if (seeding == "native") "" else "_rel", if (is.na(A_ALLEE)) "_noallee" else if (A_ALLEE != 2) paste0("_A", A_ALLEE) else "", if (length(args) >= 6) paste0("_k", KMEAN, "_l", args[7]) else "", "_s", seed)
writeBin(as.numeric(aperm(N, c(3, 1, 2))), file.path(d, paste0("N_", tag, ".f64")), size = 8, endian = "little")
write.csv(data.frame(i = obs$i, y = y, mu = Nobs * p), file.path(d, paste0("y_", tag, ".csv")), row.names = FALSE)
tot <- apply(N, 3, sum, na.rm = TRUE)
cat("total N 1902/1939/1940/1966/1990/2025:", tot[c(1, 38, 39, 65, 89, 124)], "\n")
cat("params: kappa", KAPPA, "r", R_GROW, "A", A_ALLEE, "kernel mean km", KMEAN, "max_dist", MAXD, "p", p, "\nDONE\n")

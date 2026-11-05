# Project-local R library for the SDM baselines (never system-wide, never the user library).
# Usage: Rscript research/r/install_rlib.R   (from the repo root; library at research/r/lib, untracked)
lib <- normalizePath(file.path("research", "r", "lib"), mustWork = FALSE)
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .Library))           # project lib first, then base R only (no user library)
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 2)
pkgs <- c("kissmig", "rangr")
res <- list()
for (p in pkgs) {
  r <- tryCatch({
    install.packages(p, lib = lib, dependencies = c("Depends", "Imports", "LinkingTo"), type = "binary")
    "ok"
  }, error = function(e) paste("ERROR:", conditionMessage(e)), warning = function(w) paste("WARNING:", conditionMessage(w)))
  res[[p]] <- r
}
cat("\n==== install results\n"); print(res)
cat("==== versions\n")
for (p in c(pkgs, "terra")) {
  v <- tryCatch(as.character(packageVersion(p, lib.loc = lib)), error = function(e) "NOT INSTALLED")
  cat(p, v, "\n")
}
cat("library:", lib, "\n")

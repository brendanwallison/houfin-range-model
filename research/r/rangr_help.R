.libPaths(c(normalizePath("research/r/lib"), .Library))
suppressPackageStartupMessages(library(rangr))
db <- tools::Rd_db("rangr")
for (f in c("initialise.Rd", "sim.Rd", "get_observations.Rd")) {
  tf <- tempfile(); tools::Rd2txt(db[[f]], out = tf, options = list(underline_titles = FALSE))
  txt <- readLines(tf)
  cat("=====", f, "\n"); cat(txt[1:min(length(txt), 260)], sep = "\n")
}

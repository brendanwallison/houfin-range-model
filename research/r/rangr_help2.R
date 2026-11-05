.libPaths(c(normalizePath("research/r/lib"), .Library))
suppressPackageStartupMessages(library(rangr))
db <- tools::Rd_db("rangr")
print(names(db))
for (f in intersect(c("growth.Rd", "update.sim_data.Rd", "update.Rd", "disp.Rd"), names(db))) {
  tf <- tempfile(); tools::Rd2txt(db[[f]], out = tf, options = list(underline_titles = FALSE))
  txt <- readLines(tf); cat("=====", f, "\n"); cat(txt[1:min(length(txt), 120)], sep = "\n")
}
print(body(rangr:::ricker))
print(rangr:::ricker)

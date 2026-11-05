#!/usr/bin/env Rscript
# IDAT route: raw GEO IDATs -> per-batch beta matrices that `methylsparse prepare`
# can stream side by side (identical probe order, GSM ids as column names).
#
# Use this if the series has no usable processed beta matrix, or if you want
# full control over normalisation. Sample annotation handling follows
# https://github.com/mwsill/mnp_training/blob/master/preprocessing.R
#
# Usage:
#   Rscript scripts/preprocess_idat.R GSE90496 data/raw/GSE90496/GSE90496_RAW data/beta_batches/GSE90496 [batch_size]
#
# Prerequisites (once):
#   BiocManager::install(c("minfi", "GEOquery",
#     "IlluminaHumanMethylation450kmanifest",
#     "IlluminaHumanMethylation450kanno.ilmn12.hg19"))
#   install.packages("data.table")
#   GSE90496_RAW.tar downloaded from GEO and extracted into the idat dir.
#
# Note: noob (single-sample) normalisation is used so batches are independent.
# Capper et al. used Illumina normalisation + FFPE/frozen batch correction; the
# material column is kept in the annotation if you want to model it later.

suppressPackageStartupMessages({
  library(minfi)
  library(GEOquery)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: preprocess_idat.R GSE IDAT_DIR OUT_DIR [batch_size]")
gse_id   <- args[1]
idat_dir <- args[2]
out_dir  <- args[3]
bsize    <- if (length(args) >= 4) as.integer(args[4]) else 200L
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

gse  <- getGEO(gse_id, GSEMatrix = TRUE, getGPL = FALSE)
anno <- pData(gse[[1]])
base <- gsub("_Grn.*", "", gsub(".*suppl/", "", anno$supplementary_file))
paths <- file.path(idat_dir, base)
ok <- file.exists(paste0(paths, "_Grn.idat")) | file.exists(paste0(paths, "_Grn.idat.gz"))
if (!all(ok)) message(sum(!ok), " samples without IDATs are skipped")
anno <- anno[ok, ]; paths <- paths[ok]

batches <- split(seq_along(paths), ceiling(seq_along(paths) / bsize))
probes_ref <- NULL
for (b in seq_along(batches)) {
  idx <- batches[[b]]
  out <- file.path(out_dir, sprintf("beta_batch_%03d.csv.gz", b))
  if (file.exists(out)) { message("skip ", out); next }
  message(sprintf("[%s] batch %d/%d (%d samples)", Sys.time(), b, length(batches), length(idx)))
  rg <- read.metharray(paths[idx], verbose = FALSE, force = TRUE)
  ms <- preprocessNoob(rg)
  beta <- getBeta(ms, offset = 100)
  beta <- beta[grepl("^cg", rownames(beta)), , drop = FALSE]
  beta <- beta[order(rownames(beta)), , drop = FALSE]
  if (is.null(probes_ref)) probes_ref <- rownames(beta)
  stopifnot(identical(rownames(beta), probes_ref))
  colnames(beta) <- anno$geo_accession[idx]
  dt <- data.table(ID_REF = rownames(beta), round(beta, 4))
  fwrite(dt, out, sep = "\t")
  rm(rg, ms, beta, dt); gc()
}
message("done. Next: methylsparse prepare --beta ", out_dir, "/beta_batch_*.csv.gz --series <series_matrix> ...")

# ============================================================
# Case-study preprocessing (arrays) with party split + collinearity pruning
# - Passive party: ONLY features listed in featureModality2.txt
# - Active party:  ALL remaining selected features (modality1,3,4,5) + RESPONSE
# - Collinearity check on PASSIVE features (patient-level summaries) and prune
# - Outputs:
#     X_passive_arr (T x N x P_passive_kept) -> predictors_passive.RData
#     X_active_arr  (T x N x P_active)       -> predictors_active.RData
#     Y_arr         (T x N)                  -> response.RData
#     passive_kept_names.txt / passive_dropped_names.txt
# ============================================================

setwd("~/OneDrive - Inside MD Anderson/Personal/VFL_code/case_study")
suppressPackageStartupMessages({
  library(fda)
})
############################################################################
yMapping = function(arr){
  shape = dim(arr)
  
  for(i in 1:shape[1]){
    for(j in 1:shape[2]){
      arr[i, j] = AHIMapping(arr[i, j])
    }
  }
  return(arr)
}
############################################################################
AHIMapping = function(x){
  if(x<=5){
    return(1)
  }else if(x<=14){
    return(2)
  }else if(x<=30){
    return(3)
  }else{
    return(4)
  }
}
# ------------------------- utilities -------------------------
clean_names <- function(x) {
  y <- as.character(x)
  y <- sub("\\.csv$", "", y, ignore.case = TRUE)
  y <- trimws(y)
  y[nzchar(y)]
}
read_names_file <- function(path) {
  if (!file.exists(path)) stop("Missing names file: ", path)
  nm <- tryCatch(read.table(path, stringsAsFactors = FALSE)[,1], error = function(e) character(0))
  clean_names(nm)
}
read_matrix <- function(name_noext, dir = "data") {
  f <- file.path(dir, paste0(name_noext, ".csv"))
  if (!file.exists(f)) stop("Missing feature file: ", f)
  M <- as.matrix(read.csv(f, check.names = FALSE))
  if (anyNA(M)) stop("NA detected in file: ", f)
  M
}

# Passive collinearity pruning (patient-level mean across time -> N-vector per feature)
prune_collinear <- function(X_list, feature_names, thr = 0.95) {
  stopifnot(length(X_list) == length(feature_names))
  if (length(X_list) <= 1) return(list(keep = feature_names, drop = character(0), C = NA))
  # Build patient-level summaries (N x P): mean over time
  N <- ncol(X_list[[1]])
  P <- length(X_list)
  Summ <- matrix(NA_real_, nrow = N, ncol = P)
  colnames(Summ) <- feature_names
  for (j in seq_len(P)) Summ[, j] <- colMeans(X_list[[j]], na.rm = TRUE)
  C <- cor(Summ, use = "pairwise.complete.obs")
  if (anyNA(C)) C[is.na(C)] <- 0
  keep <- colnames(C)
  drop <- character(0)
  # Greedy prune
  repeat {
    C_abs <- abs(C); diag(C_abs) <- 0
    m <- which(C_abs == max(C_abs), arr.ind = TRUE)
    if (length(m) == 0) break
    rmax <- C_abs[m[1, 1], m[1, 2]]
    if (rmax < thr) break
    i1 <- rownames(C_abs)[m[1, 1]]; i2 <- colnames(C_abs)[m[1, 2]]
    # drop the one with larger average absolute correlation
    avg1 <- mean(C_abs[i1, ])
    avg2 <- mean(C_abs[i2, ])
    drop_i <- if (avg1 >= avg2) i1 else i2
    drop  <- c(drop, drop_i)
    keep  <- setdiff(keep, drop_i)
    C <- C[keep, keep, drop = FALSE]
    if (length(keep) <= 1) break
  }
  list(keep = keep, drop = unique(drop), C = C)
}

# ------------------------- load selections -------------------------
featureSelectedModality1 <- read.table("featureModality1.txt", header = F)
featureSelectedModality2 <- read_names_file("featureModality2.txt")
featureSelectedModality3 <- read.table("featureModality3.txt", header = F)
featureSelectedModality4 <- read.table("featureModality4.txt", header = F)
featureSelectedModality5 <- read.table("featureModality5.txt", header = F)

# response file
resp_name <- "count"   # corresponds to data/count.csv

# Active/Passive split per requirement
passive_names <- unique(setdiff(featureSelectedModality2, resp_name))
active_names  <- unique(setdiff(c(featureSelectedModality1,
                                  featureSelectedModality3,
                                  featureSelectedModality4,
                                  featureSelectedModality5), resp_name))

cat("[INFO] Passive features (raw): ", length(passive_names), "\n")
cat("[INFO] Active features  (raw): ", length(active_names),  "\n")

# ------------------------- load response -------------------------
xyfine <- 70
response <- as.matrix(read.csv(file = file.path("data", paste0(resp_name, ".csv")), check.names = FALSE))
if (anyNA(response)) stop("NA detected in response file.")
if (nrow(response) != xyfine) {
  stop("Unexpected response time length: got ", nrow(response), ", expected ", xyfine)
}
# column-center each patient, then map
for (ii in 1:ncol(response)) response[, ii] <- response[, ii] - mean(response[, ii])
response <- yMapping(12 * response)   # keep your mapping
Y_arr <- response
save(Y_arr, file = "response.RData")

shape <- dim(response)   # T x N
T_obs <- shape[1]; N_pat <- shape[2]

# ------------------------- load passive feature matrices -------------------------
# Build a list of matrices for passive (for pruning)
passive_list <- list()
keep_order   <- character(0)
for (nm in passive_names) {
  M <- read_matrix(nm, dir = "data")
  if (!all(dim(M) == shape)) {
    stop("Shape mismatch for passive feature ", nm, ": got ", paste(dim(M), collapse="x"),
         " expected ", paste(shape, collapse="x"))
  }
  passive_list[[nm]] <- M
  keep_order <- c(keep_order, nm)
}

# ------------------------- collinearity pruning (passive) -------------------------
COL_THRES <- 0.95
pr <- prune_collinear(passive_list, keep_order, thr = COL_THRES)
passive_keep   <- pr$keep
passive_drop   <- pr$drop
cat("[INFO] Passive pruning: kept ", length(passive_keep), " / ", length(keep_order),
    " (drop=", length(passive_drop), ", thr=", COL_THRES, ")\n", sep = "")
writeLines(passive_keep,  con = "passive_kept_names.txt")
writeLines(passive_drop,  con = "passive_dropped_names.txt")

# ------------------------- assemble passive array -------------------------
P_passive_kept <- length(passive_keep)
X_passive_arr <- array(NA_real_, dim = c(T_obs, N_pat, P_passive_kept))
for (j in seq_along(passive_keep)) {
  X_passive_arr[ , , j] <- passive_list[[ passive_keep[j] ]]
}
save(X_passive_arr, file = "predictors_passive.RData")

# ------------------------- assemble active array -------------------------
# Active party holds: (active_names) + response (Y_arr is separate object)
active_keep <- active_names  # no pruning requested on active side
P_active <- length(active_keep)
X_active_arr <- array(NA_real_, dim = c(T_obs, N_pat, P_active))
for (j in seq_along(active_keep)) {
  M <- read_matrix(active_keep[j], dir = "data")
  if (!all(dim(M) == shape)) {
    stop("Shape mismatch for active feature ", active_keep[j], ": got ", paste(dim(M), collapse="x"),
         " expected ", paste(shape, collapse="x"))
  }
  X_active_arr[ , , j] <- M
}
save(X_active_arr, file = "predictors_active.RData")

# ------------------------- also provide full predictor cube if needed -------------------------
# (passive kept + active, order documented)
predictorName <- c(paste0(passive_keep, ".csv"), paste0(active_keep, ".csv"))
X_arr <- array(NA_real_, dim = c(T_obs, N_pat, length(predictorName)))
if (length(passive_keep)) X_arr[ , , seq_along(passive_keep)] <- X_passive_arr
if (length(active_keep))  X_arr[ , , length(passive_keep) + seq_along(active_keep)] <- X_active_arr
save(X_arr, predictorName, file = "predictors_all.RData")

cat("[DONE] Saved:\n",
    "  - predictors_passive.RData (X_passive_arr: ", T_obs, "x", N_pat, "x", P_passive_kept, ")\n",
    "  - predictors_active.RData  (X_active_arr : ", T_obs, "x", N_pat, "x", P_active, ")\n",
    "  - predictors_all.RData     (X_arr + predictorName)\n",
    "  - response.RData           (Y_arr: ", T_obs, "x", N_pat, ")\n",
    "  - passive_kept_names.txt / passive_dropped_names.txt\n", sep = "")

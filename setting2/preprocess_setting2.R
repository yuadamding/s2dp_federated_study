# ============================================
# Setting 2 (EASY + Stronger Preprocessing ONLY)
# ============================================
setwd("~/OneDrive - Inside MD Anderson/Personal/VFL_code")
suppressPackageStartupMessages({ library(fda) })

# -------------------- Configuration --------------------
set.seed(2025)
data_dir <- "setting2_raw"
out_dir  <- "setting2"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

duration <- 10
rangeval <- c(0, duration)

Q <- 12
basisobj <- create.bspline.basis(rangeval, nbasis = Q)
fdParobj <- fdPar(basisobj, Lfdobj = int2Lfd(2), lambda = 0)

predictor_map <- list(
  ecg12pc1 = "ecg12",  # active
  ppg      = "ppg",
  rsp      = "rsp",    # active
  eda      = "eda",
  emg      = "emg"
)
p <- length(predictor_map)

active_idx   <- c(1, 3)           # ecg12pc1, rsp
weights_act  <- c(+3.0, -2.0)
sigma_y_coef <- 0.02
normalize_preds <- TRUE

# --- Preprocessing toggles (ONLY preprocessing; method unchanged) ---
winsor_prob              <- 0.01     # 1% tails
keep_dc_active           <- TRUE     # keep DC in actives to preserve mean contrast
remove_dc_passive        <- TRUE     # remove DC in passives
scale_mad_all            <- TRUE     # robust scale by MAD per subject
decorrelate_passives     <- "project"  # "project" | "shift" | "none"
shift_frac_range         <- c(0.15, 0.35)  # if "shift": circular shift fraction of length
# --------------------------------------------------------------------

subjects_per_rep <- 1000

# -------------------- Utilities ------------------------
read_signal_vector <- function(path) {
  if (!file.exists(path)) stop("Missing file: ", path)
  as.numeric(scan(path, what = double(), sep = ",", quiet = TRUE))
}
read_ecg12_pc1 <- function(path) {
  if (!file.exists(path)) stop("Missing file: ", path)
  df <- tryCatch(read.csv(path, check.names = FALSE), error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0L) stop("Cannot read ecg12 DataFrame: ", path)
  num_cols <- vapply(df, is.numeric, logical(1))
  X <- as.matrix(df[, num_cols, drop = FALSE])
  drop_candidates <- grepl("time|index", tolower(colnames(X)))
  if (any(drop_candidates)) X <- as.matrix(X[, !drop_candidates, drop = FALSE])
  if (ncol(X) < 1L) stop("No numeric ECG-lead columns found in: ", path)
  pc <- prcomp(X, center = TRUE, scale. = TRUE)
  as.numeric(pc$x[, 1])
}

winsorize_vec <- function(x, prob=0.01){
  q <- quantile(x, c(prob, 1-prob), na.rm=TRUE, names=FALSE)
  x[x < q[1]] <- q[1]; x[x > q[2]] <- q[2]; x
}
circshift <- function(x, k){
  n <- length(x); if (n <= 1) return(x)
  k <- ((k %% n) + n) %% n
  if (k == 0) x else c(tail(x, k), head(x, n-k))
}
preproc_raw <- function(x, keep_dc, do_scale=TRUE){
  x <- winsorize_vec(x, winsor_prob)
  if (!keep_dc) x <- x - mean(x, na.rm=TRUE)
  if (do_scale && scale_mad_all) {
    s <- mad(x, center=0, constant=1.4826, na.rm=TRUE)
    if (is.finite(s) && s > 1e-8) x <- x / s
  }
  x
}

smooth_one_subject <- function(ti, yi, fdParobj) {
  stopifnot(length(ti) == length(yi), length(ti) >= 4)
  fdobj <- smooth.basis(argvals = ti, y = matrix(yi, length(ti), 1), fdParobj = fdParobj)$fd
  as.numeric(fdobj$coefs)
}

discover_subject_ids <- function(root_dir, suffixes) {
  get_ids <- function(suf) {
    files <- list.files(root_dir, pattern = paste0("^output_\\d+_", suf, "\\.csv$"), full.names = FALSE)
    if (!length(files)) return(integer(0))
    as.integer(sub(paste0("^output_(\\d+)_", suf, "\\.csv$"), "\\1", files))
  }
  Reduce(intersect, lapply(suffixes, get_ids))
}

# functional energy with Gram M
energy_cols <- function(C, M) colSums(C * (M %*% C))

# per-subject projection: remove projection of c (Q) onto span(Phi) (QxK) w.r.t. M
project_out_subject <- function(c, Phi, M) {
  if (is.null(Phi) || ncol(Phi) == 0) return(c)
  G <- crossprod(Phi, M %*% Phi); G <- (G + t(G))/2
  # robust invert
  ev <- eigen(G, symmetric=TRUE, only.values=TRUE)$values
  ridge <- 1e-8
  if (!all(is.finite(ev)) || min(ev) < ridge) G <- G + (abs(min(ev, na.rm=TRUE)) + ridge) * diag(ncol(G))
  alpha <- solve(G, crossprod(Phi, M %*% c))
  c - Phi %*% alpha
}

# -------------------- Assemble and Save ----------------
required_suffixes <- unlist(predictor_map, use.names = FALSE)
all_ids <- sort(discover_subject_ids(data_dir, required_suffixes))
if (!length(all_ids)) stop("No complete subject sets found in ", data_dir)

num_duplicate <- floor(length(all_ids) / subjects_per_rep)
if (num_duplicate < 1L) stop("Not enough subjects to make at least one replicate.")

message(sprintf("[INFO] Found %d complete subjects. Making %d replicates of %d subjects each.",
                length(all_ids), num_duplicate, subjects_per_rep))

M <- inprod(basisobj, basisobj)  # Gram once

for (hh in seq_len(num_duplicate)) {
  message(sprintf("==== Setting 2 | EASY+Preproc | Replicate %d ====", hh))
  idx_start <- (hh - 1L) * subjects_per_rep + 1L
  idx_end   <- hh * subjects_per_rep
  subj_ids  <- all_ids[idx_start:idx_end]
  Nsub      <- length(subj_ids)
  
  pred_coefs <- lapply(seq_len(p), function(.) matrix(NA_real_, nrow = Q, ncol = Nsub))
  names(pred_coefs) <- names(predictor_map)
  
  for (m in seq_along(subj_ids)) {
    sid <- subj_ids[m]
    
    # Load raw signals
    path_ecg12 <- file.path(data_dir, sprintf("output_%d_%s.csv", sid, predictor_map$ecg12pc1))
    ecg12_pc1  <- read_ecg12_pc1(path_ecg12)
    
    L <- length(ecg12_pc1)
    ti <- seq(rangeval[1], rangeval[2], length.out = L)
    
    read1d <- function(suf) read_signal_vector(file.path(data_dir, sprintf("output_%d_%s.csv", sid, suf)))
    x_ppg <- read1d(predictor_map$ppg)
    x_rsp <- read1d(predictor_map$rsp)
    x_eda <- read1d(predictor_map$eda)
    x_emg <- read1d(predictor_map$emg)
    
    # Trim lengths
    Lmin <- min(L, length(x_ppg), length(x_rsp), length(x_eda), length(x_emg))
    if (Lmin < 10) stop("Too few samples after trimming for subject ", sid)
    if (L != Lmin) { ecg12_pc1 <- ecg12_pc1[seq_len(Lmin)] }
    if (length(x_ppg) != Lmin) x_ppg <- x_ppg[seq_len(Lmin)]
    if (length(x_rsp) != Lmin) x_rsp <- x_rsp[seq_len(Lmin)]
    if (length(x_eda) != Lmin) x_eda <- x_eda[seq_len(Lmin)]
    if (length(x_emg) != Lmin) x_emg <- x_emg[seq_len(Lmin)]
    if (length(ti)   != Lmin)  ti    <- seq(rangeval[1], rangeval[2], length.out = Lmin)
    
    # Optional decorrelation by circular shift (time-domain) for passives
    if (decorrelate_passives == "shift") {
      k <- round(runif(1, min=shift_frac_range[1], max=shift_frac_range[2]) * Lmin)
      x_ppg <- circshift(x_ppg, k)
      x_eda <- circshift(x_eda, k)
      x_emg <- circshift(x_emg, k)
      # keep actives unshifted
    }
    
    # ---------- Robust preprocessing ----------
    ecg12_pc1 <- preproc_raw(ecg12_pc1, keep_dc = keep_dc_active)
    x_ppg     <- preproc_raw(x_ppg,     keep_dc = !remove_dc_passive)
    x_rsp     <- preproc_raw(x_rsp,     keep_dc = keep_dc_active)
    x_eda     <- preproc_raw(x_eda,     keep_dc = !remove_dc_passive)
    x_emg     <- preproc_raw(x_emg,     keep_dc = !remove_dc_passive)
    
    # Smooth to coefficients
    pred_coefs$ecg12pc1[, m] <- smooth_one_subject(ti, ecg12_pc1, fdParobj)
    pred_coefs$ppg[, m]      <- smooth_one_subject(ti, x_ppg,      fdParobj)
    pred_coefs$rsp[, m]      <- smooth_one_subject(ti, x_rsp,      fdParobj)
    pred_coefs$eda[, m]      <- smooth_one_subject(ti, x_eda,      fdParobj)
    pred_coefs$emg[, m]      <- smooth_one_subject(ti, x_emg,      fdParobj)
    
    # ---------- NEW: per-subject projection of passives out of active span ----------
    if (decorrelate_passives == "project") {
      Phi <- cbind(pred_coefs[[ active_idx[1] ]][, m, drop=FALSE],
                   pred_coefs[[ active_idx[2] ]][, m, drop=FALSE])  # Q x 2
      # project each passive column-wise
      for (nm in setdiff(names(pred_coefs), names(predictor_map)[active_idx])) {
        pred_coefs[[nm]][, m] <- project_out_subject(pred_coefs[[nm]][, m], Phi, M)
      }
    }
  }
  
  # -------- Optional per-predictor normalization (robust energy) ----------
  if (normalize_preds) {
    for (nm in names(pred_coefs)) {
      C <- pred_coefs[[nm]]
      s <- sqrt(median(energy_cols(C, M), na.rm = TRUE))
      if (!is.finite(s) || s <= 1e-12) s <- 1
      pred_coefs[[nm]] <- C / s
    }
  }
  
  # ---------------- EASY Y: constant linear combo in coef space -----------
  w <- rep(0, p); names(w) <- names(predictor_map)
  w[active_idx] <- weights_act
  
  y_coefs <- matrix(0, nrow = Q, ncol = Nsub)
  for (nm in names(predictor_map)) if (w[[nm]] != 0) y_coefs <- y_coefs + w[[nm]] * pred_coefs[[nm]]
  y_coefs <- y_coefs + matrix(rnorm(Q * Nsub, sd = sigma_y_coef), nrow = Q, ncol = Nsub)
  
  # Wrap fd objects
  predictorList <- lapply(pred_coefs, function(C) fd(coef = C, basisobj = basisobj))
  predictorList <- predictorList[names(predictor_map)]
  yfdobj <- fd(coef = y_coefs, basisobj = basisobj)
  
  # Save
  save(yfdobj,        file = file.path(out_dir, sprintf("yfdobj_%d.RData", hh)))
  save(predictorList, file = file.path(out_dir, sprintf("predictorList_%d.RData", hh)))
  message(sprintf("[OK] Saved %s and %s",
                  file.path(out_dir, sprintf("yfdobj_%d.RData", hh)),
                  file.path(out_dir, sprintf("predictorList_%d.RData", hh))))
}
message("[DONE] Setting 2 EASY data prepared (stronger preprocessing; method unchanged).")

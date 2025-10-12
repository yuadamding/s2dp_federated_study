# ---------------- Accuracy vs Privacy Budget (whitened one-shot DP; per-party & global eps)
suppressPackageStartupMessages({
  library(fda)
  library(future.apply)
})

source("functions.R")  # uses the whitened one-shot DP implementation

# ---------------------- Settings ----------------------------
num_duplicate     <- 10
folds             <- 4
N_global          <- 500
p                 <- 20                      # one predictor per party
rangeval          <- c(0, 100)
t_basis           <- 20
basisobj          <- create.bspline.basis(rangeval, nbasis = t_basis)
tgrid             <- seq(rangeval[1], rangeval[2], by = 1)

# DP sweep & accounting
sx_grid           <- c(0.0, 0.1, 0.2, 0.3)   # whitened noise std
Sx_mode           <- "fixed"                 # "fixed" or "empirical"
Sx_fixed          <- 3.0                     # clipping radius in whitened norm
delta_total       <- 1e-5
compose_over_cv   <- FALSE                   # one-shot reused across folds

# FoF penalties / boosting controls
lambda_s <- 5e-2; lambda_t <- 5e-2; lambda_st <- 0
nu <- 0.3; max_steps <- 30
use_aic <- "spherical"; use_aic_c <- TRUE; df_K <- 5
patience <- 6; min_steps <- 10

# ---------------------- Helpers -----------------------------
subset_fd <- function(fdobj, idx) {
  co <- fdobj$coefs
  fd(coef = co[, idx, drop = FALSE], basisobj = fdobj$basis, fdnames = fdobj$fdnames)
}

metrics_fd <- function(yhat_fd, ytrue_fd, grid) {
  Yhat <- eval.fd(grid, yhat_fd)
  Ytru <- eval.fd(grid, ytrue_fd)
  eps <- 1e-8
  rmse  <- sqrt(mean((Yhat - Ytru)^2))
  nrmse <- rmse / (sd(as.numeric(Ytru)) + eps)
  smape <- mean(2 * abs(Yhat - Ytru) / (abs(Yhat) + abs(Ytru) + eps)) * 100
  wmape <- sum(abs(Yhat - Ytru)) / (sum(abs(Ytru)) + eps) * 100
  mape  <- mean(abs(Yhat - Ytru) / (abs(Ytru) + eps)) * 100
  
  My <- inprod(ytrue_fd$basis, ytrue_fd$basis)
  C_hat  <- yhat_fd$coefs; C_true <- ytrue_fd$coefs
  C_diff <- C_hat - C_true
  il2  <- sum(colSums(C_diff * (My %*% C_diff))) / ncol(C_diff)
  denom <- sum(colSums(C_true * (My %*% C_true))) / ncol(C_true)
  ril2 <- il2 / (denom + eps)
  list(wmape = wmape, smape = smape, nrmse = nrmse, mape = mape, rmse = rmse, il2 = il2, ril2 = ril2)
}

# one-shot, per party: Ntrain*Qx doubles
comm_cost_mb <- function(Ntrain, Qx, p) {
  bytes <- p * (Ntrain * Qx) * 8
  bytes / (1024^2)
}

adapt_Sx <- function(Xlist, mode = c("fixed", "empirical"), Sx_fixed = 3.0) {
  mode <- match.arg(mode)
  if (mode == "fixed") return(rep(Sx_fixed, length(Xlist)))
  Sx_vec <- numeric(length(Xlist))
  for (j in seq_along(Xlist)) {
    fdj <- Xlist[[j]]
    Mx  <- inprod(fdj$basis, fdj$basis)
    Z   <- sym_eigen_sqrt(Mx)$invhalf %*% fdj$coefs   # whitened coords
    norms <- sqrt(colSums(Z * Z))
    Sx_vec[j] <- as.numeric(quantile(norms, 0.95, na.rm = TRUE))
    if (!is.finite(Sx_vec[j]) || Sx_vec[j] <= 0) Sx_vec[j] <- 3.0
  }
  Sx_vec
}

# ---- NEW: classification helpers (train-calibrated threshold) ----
score_fd_mean <- function(fdobj, grid) {
  M <- eval.fd(grid, fdobj)  # |grid| x N
  colMeans(M)
}
sens_spec_from_fd <- function(yhat_test_fd, y_test_fd, y_train_fd, grid) {
  thr <- median(score_fd_mean(y_train_fd, grid), na.rm = TRUE)
  s_true <- score_fd_mean(y_test_fd, grid)
  s_pred <- score_fd_mean(yhat_test_fd, grid)
  y_true <- as.integer(s_true >= thr)
  y_pred <- as.integer(s_pred >= thr)
  TP <- sum(y_pred == 1 & y_true == 1)
  TN <- sum(y_pred == 0 & y_true == 0)
  FP <- sum(y_pred == 1 & y_true == 0)
  FN <- sum(y_pred == 0 & y_true == 1)
  sens <- if ((TP + FN) > 0) TP / (TP + FN) else NA_real_
  spec <- if ((TN + FP) > 0) TN / (TN + FP) else NA_real_
  list(sensitivity = sens, specificity = spec, threshold = thr,
       TP = TP, TN = TN, FP = FP, FN = FN)
}

# Build global dataset (no workers; one predictor per party)
build_global_dataset <- function(hh, train_idx, test_idx) {
  env <- new.env(parent = emptyenv())
  load(sprintf("yfdobj_%d.RData", hh), envir = env)
  load(sprintf("predictorList_%d.RData", hh), envir = env)
  yfdobj <- get("yfdobj", envir = env)
  Xlist  <- get("predictorList", envir = env)
  stopifnot(length(Xlist) == p, inherits(yfdobj, "fd"))
  list(
    Xlist_train = lapply(Xlist, subset_fd, idx = train_idx),
    Xlist_test  = lapply(Xlist, subset_fd, idx = test_idx),
    Y_train     = subset_fd(yfdobj, train_idx),
    Y_test      = subset_fd(yfdobj,  test_idx)
  )
}

# zCDP epsilon per-party & global (no fold composition)
eps_job_from_Sx <- function(Sx_vec, sx, delta, compose_over_cv = FALSE) {
  comp <- if (isTRUE(compose_over_cv)) 2L else 1L
  adj  <- sqrt(comp)                     # ρ scales linearly; simple toggle (kept for clarity)
  out  <- eps_from_sx_zcdp_vec(Sx_vec, sx, delta)
  list(
    eps_j      = out$eps_j,                # per-party epsilon (vector length p)
    eps_global = eps_from_rho(out$rho_total, delta),
    rho_j      = out$rho_j,
    rho_total  = out$rho_total
  )
}

# ---------------------- Job table ----------------------------
set <- 1:N_global
fold_size <- N_global / folds; stopifnot(fold_size == floor(fold_size))

tasks <- expand.grid(
  sx  = sx_grid,
  hh  = seq_len(num_duplicate),
  fold = seq_len(folds),
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
)

# ---------------------- Parallel plan -----------------------
n_cores <- parallel::detectCores(logical = TRUE)
future::plan(multisession, workers = max(1, min(n_cores - 1, nrow(tasks))))

# ---------------------- Run sweep in parallel ---------------
job_rows <- future_lapply(
  X = seq_len(nrow(tasks)),
  FUN = function(i_job) {
    library(fda); source("functions.R")
    sx   <- tasks$sx[i_job]
    hh   <- tasks$hh[i_job]
    fold <- tasks$fold[i_job]
    set.seed(1e6 * hh + 1e3 * fold + round(1e3 * sx))
    
    test_idx  <- set[((fold - 1) * fold_size + 1):(fold * fold_size)]
    train_idx <- setdiff(set, test_idx)
    
    ds <- build_global_dataset(hh, train_idx, test_idx)
    X_train <- ds$Xlist_train; X_test <- ds$Xlist_test
    Y_train <- ds$Y_train;     Y_test <- ds$Y_test
    
    # Clipping radii (whitened norm)
    Sx_vec <- adapt_Sx(X_train, mode = Sx_mode, Sx_fixed = Sx_fixed)
    sx_vec <- rep(sx, length(X_train))
    
    # Privacy accounting (per-party and global)
    eps_pack <- eps_job_from_Sx(Sx_vec, sx, delta_total, compose_over_cv)
    eps_j      <- eps_pack$eps_j
    eps_global <- eps_pack$eps_global
    
    # Fit (AIC-stop by default)
    t0 <- Sys.time()
    fit <- vfl_dp_foboost(
      xfd_list = X_train, yfd = Y_train,
      Sx_vec = Sx_vec, sx_vec = sx_vec,
      Omega_x_list = NULL, Omega_y = NULL,
      lambda_s = lambda_s, lambda_t = lambda_t, lambda_st = lambda_st,
      nu = nu, max_steps = max_steps,
      crossfit = TRUE,
      stop_mode = "aic_train",
      min_steps = min_steps,
      aic = use_aic, aic_c = use_aic_c, df_K = df_K,
      patience = patience
    )
    time_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    
    # Predict & metrics (regression)
    yhat_test <- predict_vfl_dp_foboost(fit, X_test)
    mets <- metrics_fd(yhat_test, Y_test, grid = tgrid)
    
    # NEW: Sensitivity & Specificity (threshold from TRAIN)
    cls <- sens_spec_from_fd(yhat_test, Y_test, Y_train, grid = tgrid)
    
    # Communication (one-shot)
    comm_mb <- comm_cost_mb(Ntrain = length(train_idx), Qx = t_basis, p = p)
    
    # Assemble row (include per-party eps as separate columns)
    out <- data.frame(
      sx = sx, hh = hh, fold = fold,
      WMAPE = mets$wmape, SMAPE = mets$smape, NRMSE = mets$nrmse,
      MAPE  = mets$mape,  RMSE  = mets$rmse,
      IL2   = mets$il2,   RIL2  = mets$ril2,
      Sensitivity = cls$sensitivity,         # NEW
      Specificity = cls$specificity,         # NEW
      Eps_global = eps_global,
      Time_sec = time_sec, Comm_MB = comm_mb,
      stringsAsFactors = FALSE, check.names = FALSE
    )
    # add per-party eps as wide columns: Eps_p1 ... Eps_pP
    for (j in seq_len(p)) out[[sprintf("Eps_p%d", j)]] <- eps_j[j]
    out
  },
  future.seed = TRUE
)

valid_idx <- vapply(job_rows, function(x) is.data.frame(x) && nrow(x) == 1, logical(1))
if (!any(valid_idx)) {
  cat("\n[WARN] No successful jobs. Check data files.\n")
  sweep_df <- data.frame()
} else {
  sweep_df <- do.call(rbind, job_rows[valid_idx])
}

# ---------------------- Summaries -----------------------------------------
if (nrow(sweep_df) == 0) {
  write.csv(sweep_df, file = "privacy_sweep_perjob.csv", row.names = FALSE)
  cat("\n[INFO] Wrote empty per-job CSV. No summary due to 0 rows.\n")
} else {
  # Mean/sd by sx only (since parties are fixed and one per feature)
  base_cols <- c("WMAPE","SMAPE","NRMSE","MAPE","RMSE","IL2","RIL2",
                 "Sensitivity","Specificity",         # NEW
                 "Time_sec","Comm_MB","Eps_global")
  agg_mean <- aggregate(sweep_df[, base_cols], by = list(sx = sweep_df$sx),
                        FUN = function(x) mean(x, na.rm = TRUE))
  agg_sd   <- aggregate(sweep_df[, base_cols], by = list(sx = sweep_df$sx),
                        FUN = function(x) sd(x, na.rm = TRUE))
  
  names(agg_mean)[-1] <- paste0(names(agg_mean)[-1], "_mean")
  names(agg_sd)[-1]   <- paste0(names(agg_sd)[-1],   "_sd")
  
  summary_df <- merge(agg_mean, agg_sd, by = "sx", all = TRUE)
  
  write.csv(sweep_df,  file = "privacy_sweep_perjob.csv",   row.names = FALSE)
  write.csv(summary_df, file = "privacy_sweep_summary.csv", row.names = FALSE)
  save(sweep_df, summary_df, file = "privacy_sweep_results.RData")
  
  cat("\n=== Accuracy vs Privacy (means ± sd by sx) ===\n")
  print(summary_df[order(summary_df$sx), ])
}

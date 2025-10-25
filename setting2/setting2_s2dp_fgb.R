# ======= setting2_s2dp_fgb.R  =======
# One-shot zCDP VFL FoF boosting — adapted for Setting 2 data
# (yfdobj_<hh>.RData, predictorList_<hh>.RData produced from NeuroKit2)
suppressPackageStartupMessages({ library(fda); library(future.apply) })
source("functions.R")   # uses: vfl_dp_foboost(), rho_from_eps(), eps_from_rho(), etc.

# -------------------- settings --------------------
folds <- 5
# Sweep of total (final) privacy targets (ε, with δ below)
eps_grid    <- c(Inf, 100, 80, 60, 40, 20, 10)     # edit as desired, e.g., c(Inf, 200, 100, 50, 20, 10)
delta_total <- 1e-5

lambda_s <- 5e-1; lambda_t <- 5e-1; lambda_st <- 0
nu <- 0.1; max_steps <- 30; patience <- 6; min_steps <- 2

# -------------------- helpers --------------------
subset_fd <- function(fdobj, idx) fd(coef = fdobj$coefs[, idx, drop = FALSE],
                                     basisobj = fdobj$basis, fdnames = fdobj$fdnames)

metrics_fd <- function(yhat_fd, ytrue_fd, grid) {
  Yhat <- eval.fd(grid, yhat_fd)   # |grid| x N
  Ytru <- eval.fd(grid, ytrue_fd)  # |grid| x N
  eps <- 1e-8
  err  <- Yhat - Ytru
  rmse <- sqrt(mean(err^2))
  rmse_subj <- sqrt(colMeans(err^2))
  denom_sd <- sd(as.numeric(Ytru)) + eps
  nrmse        <- rmse / denom_sd
  nrmse_worst  <- max(rmse_subj) / denom_sd
  smape <- mean(2 * abs(err) / (abs(Yhat) + abs(Ytru) + eps)) * 100
  wmape <- sum(abs(err)) / (sum(abs(Ytru)) + eps) * 100
  mape  <- mean(abs(err) / (abs(Ytru) + eps)) * 100
  My <- inprod(ytrue_fd$basis, ytrue_fd$basis)
  C_hat  <- yhat_fd$coefs; C_true <- ytrue_fd$coefs
  C_diff <- C_hat - C_true
  il2  <- sum(colSums(C_diff * (My %*% C_diff))) / ncol(C_diff)
  denom <- sum(colSums(C_true * (My %*% C_true))) / ncol(C_true)
  ril2 <- il2 / (denom + eps)
  list(wmape = wmape, smape = smape, nrmse = nrmse, nrmse_worst = nrmse_worst,
       mape = mape, rmse = rmse, il2 = il2, ril2 = ril2)
}

score_fd_mean <- function(fdobj, grid) colMeans(eval.fd(grid, fdobj))
sens_spec_from_fd <- function(yhat_test_fd, y_test_fd, y_train_fd, grid) {
  thr <- median(score_fd_mean(y_train_fd, grid), na.rm = TRUE)
  s_true <- score_fd_mean(y_test_fd, grid); s_pred <- score_fd_mean(yhat_test_fd, grid)
  y_true <- as.integer(s_true >= thr); y_pred <- as.integer(s_pred >= thr)
  TP <- sum(y_pred==1 & y_true==1); TN <- sum(y_pred==0 & y_true==0)
  FP <- sum(y_pred==1 & y_true==0); FN <- sum(y_pred==0 & y_true==1)
  list(sensitivity = if ((TP+FN)>0) TP/(TP+FN) else NA_real_,
       specificity = if ((TN+FP)>0) TN/(TN+FP) else NA_real_)
}

sym_eigen_sqrt <- function(M, ridge=1e-8){
  ee <- eigen((M+t(M))/2, symmetric=TRUE)
  lam <- pmax(ee$values, ridge); U <- ee$vectors
  list(half = U %*% diag(sqrt(lam), length(lam)) %*% t(U),
       invhalf = U %*% diag(1/sqrt(lam), length(lam)) %*% t(U))
}

adapt_Sx_empirical <- function(Xlist, q = 0.90) {
  sapply(Xlist, function(fdj) {
    Mx  <- inprod(fdj$basis, fdj$basis)
    Z   <- sym_eigen_sqrt(Mx)$invhalf %*% fdj$coefs
    norms <- sqrt(colSums(Z * Z))
    out <- as.numeric(quantile(norms, q, na.rm = TRUE))
    if (!is.finite(out) || out <= 0) 3.0 else out
  })
}

comm_cost_mb <- function(Ntrain, Qx, p) (p * (Ntrain * Qx) * 8) / (1024^2)

discover_replicates <- function() {
  ys  <- list.files("setting2", pattern = "^yfdobj_(\\d+)\\.RData$", full.names = FALSE)
  xs  <- list.files("setting2", pattern = "^predictorList_(\\d+)\\.RData$", full.names = FALSE)
  idy <- as.integer(sub("^yfdobj_(\\d+)\\.RData$", "\\1", ys))
  idx <- as.integer(sub("^predictorList_(\\d+)\\.RData$", "\\1", xs))
  sort(intersect(idy, idx))
}

# -------------------- jobs table --------------------
rep_ids <- discover_replicates()
if (!length(rep_ids)) stop("No Setting 2 replicates found: expected yfdobj_<hh>.RData and predictorList_<hh>.RData")

tasks <- expand.grid(eps_target = eps_grid,
                     hh = rep_ids,
                     fold = seq_len(folds),
                     KEEP.OUT.ATTRS = FALSE)

n_cores <- parallel::detectCores(TRUE)
future::plan(multisession, workers = max(1, min(n_cores - 1, nrow(tasks))))

# -------------------- run jobs --------------------
job_rows <- future_lapply(seq_len(nrow(tasks)), function(i_job) {
  library(fda); source("functions.R")
  eps_t <- tasks$eps_target[i_job]
  hh    <- tasks$hh[i_job]
  fold  <- tasks$fold[i_job]
  
  # --- load replicate and infer dims/grid ---------------------------------
  env <- new.env(parent = emptyenv())
  load(sprintf("setting2/yfdobj_%d.RData", hh), envir = env)
  load(sprintf("setting2/predictorList_%d.RData", hh), envir = env)
  yfdobj <- get("yfdobj", env); Xlist <- get("predictorList", env)
  N_fd   <- ncol(yfdobj$coefs)
  p_job  <- length(Xlist)
  Qx_job <- nrow(Xlist[[1]]$coefs)
  r      <- yfdobj$basis$rangeval
  tgrid  <- seq(r[1], r[2], length.out = 201)
  
  # --- deterministic fold split -------------------------------------------
  fold_id  <- cut(seq_len(N_fd), breaks = folds, labels = FALSE, include.lowest = TRUE)
  test_idx <- which(fold_id == fold)
  train_idx <- which(fold_id != fold)
  
  subset_fd <- function(fdobj, idx) fd(coef = fdobj$coefs[, idx, drop = FALSE],
                                       basisobj = fdobj$basis, fdnames = fdobj$fdnames)
  X_train <- lapply(Xlist, subset_fd, idx = train_idx)
  X_test  <- lapply(Xlist, subset_fd, idx = test_idx)
  Y_train <- subset_fd(yfdobj, train_idx)
  Y_test  <- subset_fd(yfdobj,  test_idx)
  
  # --- empirical clipping; calibrate per-fold noise to meet total ε --------
  m_cv  <- max(1L, folds - 1L)             # subjects appear in (folds-1) training sets
  Sx_vec <- adapt_Sx_empirical(X_train, q = 0.90)
  
  if (is.finite(eps_t) && eps_t > 0) {
    rho_target <- rho_from_eps(eps_t, delta_total)        # zCDP ρ
    rho_each   <- rho_target / m_cv
    # Correct zCDP Gaussian calibration: ρ = (sum S_j^2) / (2 s^2)
    sx <- sqrt( sum(Sx_vec^2) / (2 * pmax(rho_each, 1e-12)) )
  } else {
    sx <- 0
  }
  sx_vec <- rep(sx, length(X_train))
  
  # Diagnostics (note: the name 'Sigma_prop3_fold' kept for backward compatibility; it is ρ_fold)
  Rho_prop3_fold <- if (is.finite(eps_t) && sx > 0) sum(Sx_vec^2) / (2 * sx^2) else 0
  Eps_prop3_fold <- if (is.finite(eps_t) && sx > 0) eps_from_rho(Rho_prop3_fold, delta_total) else Inf
  Eps_prop3_total_implied <- if (is.finite(eps_t) && sx > 0)
    eps_from_rho(Rho_prop3_fold * m_cv, delta_total) else if (is.finite(eps_t)) Inf else 0
  
  # --- fit (AIC stopping, DP-corrected) ------------------------------------
  fit <- vfl_dp_foboost(
    xfd_list = X_train, yfd = Y_train,
    Sx_vec = Sx_vec, sx_vec = sx_vec,
    Omega_x_list = NULL, Omega_y = NULL,
    lambda_s = lambda_s, lambda_t = lambda_t, lambda_st = lambda_st,
    nu = nu, max_steps = max_steps, crossfit = TRUE,
    stop_mode = "aic_train_dp", min_steps = min_steps,
    aic = "spherical", aic_c = TRUE, df_K = 5, patience = patience
  )
  time_sec <- fit$timing$train_loop_time_s
  
  # --- evaluate on TEST ----------------------------------------------------
  yhat_test <- predict_vfl_dp_foboost(fit, X_test)
  mets <- metrics_fd(yhat_test, Y_test, grid = tgrid)
  cls  <- sens_spec_from_fd(yhat_test, Y_test, Y_train, grid = tgrid)
  
  comm_mb <- comm_cost_mb(Ntrain = length(train_idx), Qx = Qx_job, p = p_job)
  
  data.frame(
    eps_target = eps_t, hh = hh, fold = fold,
    WMAPE = mets$wmape, SMAPE = mets$smape,
    NRMSE = mets$nrmse, NRMSE_worst = mets$nrmse_worst,
    MAPE  = mets$mape,  RMSE = mets$rmse,
    IL2   = mets$il2,   RIL2 = mets$ril2,
    Sensitivity = cls$sensitivity, Specificity = cls$specificity,
    Sigma_prop3_fold = Rho_prop3_fold,             # (kept name; value is zCDP ρ per fold)
    Eps_prop3_fold   = Eps_prop3_fold,
    Eps_prop3_total_implied = Eps_prop3_total_implied,
    Time_sec = time_sec, Comm_MB = comm_mb,
    stringsAsFactors = FALSE, check.names = FALSE
  )
}, future.seed = TRUE)

# -------------------- summaries --------------------
ok <- vapply(job_rows, function(x) is.data.frame(x) && nrow(x) == 1, logical(1))
sweep_df <- if (any(ok)) do.call(rbind, job_rows[ok]) else data.frame()

out_perjob   <- "setting2_privacy_sweep_perjob.csv"
out_summary  <- "setting2_privacy_sweep_summary.csv"

if (!nrow(sweep_df)) {
  write.csv(sweep_df, out_perjob, row.names = FALSE)
  cat("\n[INFO] Wrote empty per-job CSV (no rows). No summary.\n")
} else {
  base_cols <- c("WMAPE","SMAPE","NRMSE","NRMSE_worst","MAPE","RMSE","IL2","RIL2",
                 "Sensitivity","Specificity","Time_sec","Comm_MB",
                 "Sigma_prop3_fold","Eps_prop3_fold","Eps_prop3_total_implied")
  agg_mean <- aggregate(sweep_df[, base_cols],
                        by = list(eps_target = sweep_df$eps_target),
                        FUN = function(x) mean(x, na.rm = TRUE))
  agg_sd   <- aggregate(sweep_df[, base_cols],
                        by = list(eps_target = sweep_df$eps_target),
                        FUN = function(x) sd(x, na.rm = TRUE))
  names(agg_mean)[-1] <- paste0(names(agg_mean)[-1], "_mean")
  names(agg_sd)[-1]   <- paste0(names(agg_sd)[-1],   "_sd")
  summary_df <- merge(agg_mean, agg_sd, by = "eps_target", all = TRUE)
  
  write.csv(sweep_df,  out_perjob,  row.names = FALSE)
  write.csv(summary_df, out_summary, row.names = FALSE)
  
  cat("\n=== Setting 2 — Accuracy vs Privacy (means ± sd by ε_target) ===\n")
  print(summary_df[order(summary_df$eps_target, decreasing = TRUE), ], row.names = FALSE)
}

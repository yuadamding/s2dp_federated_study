# ======= setting1_s2dp_fgb.R (one-shot S2DP-FGB; ε-calibrated) ============
suppressPackageStartupMessages({ library(fda); library(future.apply) })
source("functions.R")

# --- settings --------------------------------------------------------------
num_duplicate <- 20
folds <- 4
N_global <- 500
p <- 20
rangeval <- c(0, 100)
t_basis <- 20
basisobj <- create.bspline.basis(rangeval, nbasis = t_basis)
tgrid <- seq(rangeval[1], rangeval[2], by = 1)

eps_grid <- c(Inf,1000, 500, 100, 80, 60, 40, 30, 20)
delta_total <- 1e-5

lambda_s <- 5e-2; lambda_t <- 5e-2; lambda_st <- 0
nu <- 0.3; max_steps <- 30; patience <- 6; min_steps <- 10

# --- helpers ---------------------------------------------------------------
subset_fd <- function(fdobj, idx) fd(coef = fdobj$coefs[, idx, drop=FALSE],
                                     basisobj = fdobj$basis, fdnames = fdobj$fdnames)
metrics_fd <- function(yhat_fd, ytrue_fd, grid) {
  Yhat <- eval.fd(grid, yhat_fd)   # |grid| x N
  Ytru <- eval.fd(grid, ytrue_fd)  # |grid| x N
  eps <- 1e-8
  
  # global RMSE (your existing one)
  err  <- Yhat - Ytru
  rmse <- sqrt(mean(err^2))
  
  # per-subject RMSEs across the grid
  rmse_subj <- sqrt(colMeans(err^2))
  
  # same normalization as your current NRMSE (global sd of Y)
  denom_sd <- sd(as.numeric(Ytru)) + eps
  nrmse        <- rmse / denom_sd
  nrmse_worst  <- max(rmse_subj) / denom_sd
  
  # SMAPE / WMAPE / MAPE (unchanged)
  smape <- mean(2 * abs(err) / (abs(Yhat) + abs(Ytru) + eps)) * 100
  wmape <- sum(abs(err)) / (sum(abs(Ytru)) + eps) * 100
  mape  <- mean(abs(err) / (abs(Ytru) + eps)) * 100
  
  # Functional IL2 & relative IL2 (unchanged)
  My <- inprod(ytrue_fd$basis, ytrue_fd$basis)
  C_hat  <- yhat_fd$coefs; C_true <- ytrue_fd$coefs
  C_diff <- C_hat - C_true
  il2  <- sum(colSums(C_diff * (My %*% C_diff))) / ncol(C_diff)
  denom <- sum(colSums(C_true * (My %*% C_true))) / ncol(C_true)
  ril2 <- il2 / (denom + eps)
  
  list(
    wmape = wmape, smape = smape,
    nrmse = nrmse, nrmse_worst = nrmse_worst,
    mape = mape, rmse = rmse, il2 = il2, ril2 = ril2
  )
}
score_fd_mean <- function(fdobj, grid) colMeans(eval.fd(grid, fdobj))
sens_spec_from_fd <- function(yhat_test_fd, y_test_fd, y_train_fd, grid) {
  thr <- median(score_fd_mean(y_train_fd, grid), na.rm = TRUE)
  s_true <- score_fd_mean(y_test_fd, grid); s_pred <- score_fd_mean(yhat_test_fd, grid)
  y_true <- as.integer(s_true >= thr); y_pred <- as.integer(s_pred >= thr)
  TP <- sum(y_pred==1 & y_true==1); TN <- sum(y_pred==0 & y_true==0)
  FP <- sum(y_pred==1 & y_true==0); FN <- sum(y_pred==0 & y_true==1)
  list(sensitivity = if ((TP+FN)>0) TP/(TP+FN) else NA_real_,
       specificity = if ((TN+FP)>0) TN/(TN+FP) else NA_real_, threshold = thr,
       TP=TP,TN=TN,FP=FP,FN=FN)
}
sym_eigen_sqrt <- function(M, ridge=1e-8){
  ee <- eigen((M+t(M))/2, symmetric=TRUE)
  lam <- pmax(ee$values, ridge); U <- ee$vectors
  list(half = U %*% diag(sqrt(lam), length(lam)) %*% t(U),
       invhalf = U %*% diag(1/sqrt(lam), length(lam)) %*% t(U))
}
adapt_Sx_empirical <- function(Xlist, q = 0.95) {
  sapply(Xlist, function(fdj) {
    Mx  <- inprod(fdj$basis, fdj$basis)
    Z   <- sym_eigen_sqrt(Mx)$invhalf %*% fdj$coefs
    norms <- sqrt(colSums(Z * Z))
    out <- as.numeric(quantile(norms, q, na.rm = TRUE))
    if (!is.finite(out) || out <= 0) 3.0 else out
  })
}
comm_cost_mb <- function(Ntrain, Qx, p) (p * (Ntrain * Qx) * 8) / (1024^2)

# --- jobs ------------------------------------------------------------------
set <- 1:N_global
fold_size <- N_global / folds; stopifnot(fold_size == floor(fold_size))
tasks <- expand.grid(eps_target = eps_grid, hh = seq_len(num_duplicate),
                     fold = seq_len(folds), KEEP.OUT.ATTRS = FALSE)

n_cores <- parallel::detectCores(TRUE)
future::plan(multisession, workers = max(1, min(n_cores-1, nrow(tasks))))

job_rows <- future_lapply(seq_len(nrow(tasks)), function(i_job) {
  library(fda); source("functions.R")
  eps_t <- tasks$eps_target[i_job]; hh <- tasks$hh[i_job]; fold <- tasks$fold[i_job]
  
  # split
  test_idx  <- set[((fold - 1) * fold_size + 1):(fold * fold_size)]
  train_idx <- setdiff(set, test_idx)
  
  # load replicate
  env <- new.env(parent = emptyenv())
  load(sprintf("yfdobj_%d.RData", hh), envir = env)
  load(sprintf("predictorList_%d.RData", hh), envir = env)
  yfdobj <- get("yfdobj", env); Xlist <- get("predictorList", env)
  
  subset_fd <- function(fdobj, idx) fd(coef = fdobj$coefs[, idx, drop=FALSE],
                                       basisobj = fdobj$basis, fdnames = fdobj$fdnames)
  X_train <- lapply(Xlist, subset_fd, idx = train_idx)
  X_test  <- lapply(Xlist, subset_fd, idx = test_idx)
  Y_train <- subset_fd(yfdobj, train_idx)
  Y_test  <- subset_fd(yfdobj,  test_idx)
  
  # empirical clipping; calibrate sx for target ε
  Sx_vec <- adapt_Sx_empirical(X_train, q = 0.95)
  sx <- if (is.finite(eps_t) && eps_t > 0) {
    K <- sum(2 * (Sx_vec^2)); rho_t <- rho_from_eps(eps_t, delta_total); sqrt(K / pmax(rho_t, 1e-12))
  } else 0
  sx_vec <- rep(sx, length(X_train))
  
  # realized ε
  eps_pack <- if (is.finite(eps_t) && sx > 0) {
    out <- eps_from_sx_zcdp_vec(Sx_vec, sx, delta_total)
    list(eps_j = out$eps_j, eps_global = eps_from_rho(out$rho_total, delta_total))
  } else list(eps_j = rep(Inf, length(Sx_vec)), eps_global = Inf)
  
  # fit with CV stopping (parity)
  fit <- vfl_dp_foboost(
    xfd_list = X_train, yfd = Y_train,
    Sx_vec = Sx_vec, sx_vec = sx_vec,
    Omega_x_list = NULL, Omega_y = NULL,
    lambda_s = lambda_s, lambda_t = lambda_t, lambda_st = lambda_st,
    nu = nu, max_steps = max_steps, crossfit = TRUE,
    stop_mode = "cv", min_steps = min_steps,
    aic = "spherical", aic_c = TRUE, df_K = 5, patience = patience
  )
  time_sec <- fit$timing$train_loop_time_s
  
  yhat_test <- predict_vfl_dp_foboost(fit, X_test)
  mets <- metrics_fd(yhat_test, Y_test, grid = tgrid)
  cls  <- sens_spec_from_fd(yhat_test, Y_test, Y_train, grid = tgrid)
  comm_mb <- comm_cost_mb(Ntrain = length(train_idx), Qx = t_basis, p = p)
  
  out <- data.frame(
    eps_target = eps_t, hh = hh, fold = fold,
    WMAPE = mets$wmape, SMAPE = mets$smape, NRMSE = mets$nrmse,
    MAPE  = mets$mape,  RMSE  = mets$rmse,
    IL2   = mets$il2,   RIL2  = mets$ril2,
    Sensitivity = cls$sensitivity, Specificity = cls$specificity,
    Eps_global_realized = eps_pack$eps_global,
    NRMSE_worst = mets$nrmse_worst,
    Time_sec = time_sec, Comm_MB = comm_mb,
    stringsAsFactors = FALSE, check.names = FALSE
  )
  for (j in seq_len(p)) out[[sprintf("Eps_p%d", j)]] <- eps_pack$eps_j[j]
  out
}, future.seed = TRUE)

ok <- vapply(job_rows, function(x) is.data.frame(x) && nrow(x) == 1, logical(1))
sweep_df <- if (any(ok)) do.call(rbind, job_rows[ok]) else data.frame()

if (!nrow(sweep_df)) {
  write.csv(sweep_df, "privacy_sweep_perjob.csv", row.names = FALSE)
  cat("\n[INFO] Wrote empty per-job CSV. No summary due to 0 rows.\n")
} else {
  base_cols <- c("WMAPE","SMAPE","NRMSE","MAPE","RMSE","IL2","RIL2",
                 "Sensitivity","Specificity","Time_sec","Comm_MB","Eps_global_realized")
  agg_mean <- aggregate(sweep_df[, base_cols], by = list(eps_target = sweep_df$eps_target),
                        FUN = function(x) mean(x, na.rm = TRUE))
  agg_sd   <- aggregate(sweep_df[, base_cols], by = list(eps_target = sweep_df$eps_target),
                        FUN = function(x) sd(x, na.rm = TRUE))
  names(agg_mean)[-1] <- paste0(names(agg_mean)[-1], "_mean")
  names(agg_sd)[-1]   <- paste0(names(agg_sd)[-1],   "_sd")
  summary_df <- merge(agg_mean, agg_sd, by = "eps_target", all = TRUE)
  
  write.csv(sweep_df,  "privacy_sweep_perjob.csv",   row.names = FALSE)
  write.csv(summary_df,"privacy_sweep_summary.csv",   row.names = FALSE)
  
  cat("\n=== Accuracy vs Privacy (means ± sd by ε_target) ===\n")
  print(summary_df[order(summary_df$eps_target, decreasing = TRUE), ], row.names = FALSE)
}

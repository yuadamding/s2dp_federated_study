# ---------------- Roundwise-DP VFL Boosting: run & test ----------------
suppressPackageStartupMessages({
  library(fda)
  library(future.apply)
})

source("functions.R")
source("functions_roundwise_dp.R")

# ---------------- Settings (match your generator) -----------------------
p <- 20
rangeval <- c(0, 100); Q <- 20
basisobj <- create.bspline.basis(rangeval, nbasis = Q)
tgrid <- seq(rangeval[1], rangeval[2], by = 1)

N_global <- 500
folds     <- 4
num_dup   <- 10

# DP per-round (tune as you wish)
Su_res <- 3.0                                # clip radius (whitened Y) for residual broadcast
sx_res <- c(0.10, 0.20, 0.30)                # sweep noise std for residual broadcast
Sdir   <- 2.0                                # clip radius for returned increment
sx_dir <- c(0.10, 0.20)                      # sweep noise std for returned increment
delta_total <- 1e-5

# penalties & boosting
lambda_s <- 5e-2; lambda_t <- 5e-2; lambda_st <- 0
nu <- 0.3; max_rounds <- 30; min_rounds <- 8; patience <- 6

# ---------------- Helpers -----------------------
subset_fd <- function(fdobj, idx) {
  fd(coef = fdobj$coefs[, idx, drop=FALSE], basisobj = fdobj$basis, fdnames = fdobj$fdnames)
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

# NEW: scoring and classification helpers (threshold from TRAIN)
score_fd_mean <- function(fdobj, grid) {
  M <- eval.fd(grid, fdobj)  # matrix: |grid| x N
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

# load a replicate
load_replicate <- function(hh) {
  env <- new.env(parent = emptyenv())
  load(sprintf("yfdobj_%d.RData", hh), envir = env)
  load(sprintf("predictorList_%d.RData", hh), envir = env)
  list(yfdobj = get("yfdobj", envir = env), Xlist = get("predictorList", envir = env))
}

# ---------------- Job table ---------------------
set <- 1:N_global
fold_size <- N_global / folds; stopifnot(fold_size == floor(fold_size))

grid <- expand.grid(
  hh = seq_len(num_dup),
  fold = seq_len(folds),
  su_res = sx_res,
  s_dir  = sx_dir,
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
)

n_cores <- parallel::detectCores(logical = TRUE)
future::plan(multisession, workers = max(1, min(n_cores - 1, nrow(grid))))

# ---------------- Run --------------------------
rows <- future_lapply(seq_len(nrow(grid)), function(i) {
  library(fda); source("functions.R"); source("functions_roundwise_dp.R")
  
  hh   <- grid$hh[i]; fold <- grid$fold[i]
  su_r <- grid$su_res[i]; sd_r <- grid$s_dir[i]
  
  dat <- load_replicate(hh)
  yfd <- dat$yfdobj; Xlist <- dat$Xlist
  
  test_idx  <- set[((fold - 1) * fold_size + 1):(fold * fold_size)]
  train_idx <- setdiff(set, test_idx)
  
  y_train <- subset_fd(yfd, train_idx)
  y_test  <- subset_fd(yfd, test_idx)
  X_train <- lapply(Xlist, subset_fd, idx = train_idx)
  X_test  <- lapply(Xlist, subset_fd, idx = test_idx)
  
  t0 <- Sys.time()
  fit <- roundwise_dp_vfl_boost(
    xfd_list = X_train, yfd = y_train,
    S_res = Su_res, s_res = su_r,        # residual DP
    S_est = 2.0,  s_est = sd_r,          # estimator DP
    Omega_x_list = NULL, Omega_y = NULL,
    lambda_s = lambda_s, lambda_t = lambda_t, lambda_st = lambda_st,
    nu = nu, max_rounds = max_rounds, min_rounds = min_rounds, patience = patience,
    use_crossfit = TRUE,
    delta_total = 1e-5
  )
  time_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  # Proper prediction on TEST using accumulated DP operators
  yhat_test <- predict_roundwise_dp(fit, X_test)
  
  # Evaluate on test (regression)
  mets <- metrics_fd(yhat_test, y_test, grid = tgrid)
  
  # NEW: Sensitivity & Specificity (threshold from TRAIN)
  cls <- sens_spec_from_fd(yhat_test, y_test, y_train, grid = tgrid)
  
  # Communication per round:
  #  - residual broadcast: p * Ntrain x Qy doubles
  #  - operator return   :      Qx x Qy doubles
  Ntr <- length(train_idx)
  Qx <- nrow(X_train[[1]]$coefs); Qy <- nrow(y_train$coefs)
  comm_mb <- ( (p * Ntr * Qy) + (Qx * Qy) ) * 8 * fit$rounds / (1024^2)
  
  # Assemble row
  out <- data.frame(
    hh = hh, fold = fold, su_res = su_r, s_dir = sd_r,
    rounds = fit$rounds,
    WMAPE = mets$wmape, SMAPE = mets$smape, NRMSE = mets$nrmse,
    MAPE = mets$mape,   RMSE  = mets$rmse,
    IL2 = mets$il2,     RIL2  = mets$ril2,
    Sensitivity = cls$sensitivity,       # NEW
    Specificity = cls$specificity,       # NEW
    eps_global = fit$eps$eps_global,
    Time_sec = time_sec, Comm_MB = comm_mb,
    stringsAsFactors = FALSE, check.names = FALSE
  )
  # keep per-party eps
  for (j in seq_len(p)) out[[sprintf("Eps_p%d", j)]] <- fit$eps$eps_per_party[j]
  out
}, future.seed = TRUE)

ok <- vapply(rows, function(x) is.data.frame(x) && nrow(x) == 1, logical(1))
res <- if (any(ok)) do.call(rbind, rows[ok]) else data.frame()

if (nrow(res)) {
  write.csv(res, "roundwise_dp_perjob.csv", row.names = FALSE)
  
  metrics <- c("WMAPE","SMAPE","NRMSE","MAPE","RMSE","IL2","RIL2",
               "Sensitivity","Specificity",          
               "eps_global","Time_sec","Comm_MB")
  
  agg_mean <- aggregate(
    res[, metrics],
    by = list(su_res = res$su_res, s_dir = res$s_dir),
    FUN = function(x) mean(x, na.rm = TRUE)
  )
  agg_sd <- aggregate(
    res[, metrics],
    by = list(su_res = res$su_res, s_dir = res$s_dir),
    FUN = function(x) sd(x, na.rm = TRUE)
  )
  
  names(agg_mean) <- c("su_res","s_dir", paste0(metrics, "_mean"))
  names(agg_sd)   <- c("su_res","s_dir", paste0(metrics, "_sd"))
  
  summary_df <- merge(agg_mean, agg_sd, by = c("su_res","s_dir"), all = TRUE)
  
  write.csv(summary_df, "roundwise_dp_summary.csv", row.names = FALSE)
  print(summary_df[order(summary_df$su_res, summary_df$s_dir), ])
} else {
  cat("[WARN] No successful jobs.\n")
}

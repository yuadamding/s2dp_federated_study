suppressPackageStartupMessages({ library(fda); library(future.apply) })
source("functions.R")
source("functions_roundwise_dp.R")

# ---------------- Settings -----------------------
p <- 5; rangeval <- c(0, 100); Q <- 5
basisobj <- create.bspline.basis(rangeval, nbasis = Q)
tgrid <- seq(rangeval[1], rangeval[2], by = 1)

N_global <- 2500; folds <- 5; num_dup <- 20
R_fixed  <- 10                    
w_split  <- 0.5                   # fraction of zCDP budget to residual (rest to estimator)

eps_grid    <- c(100, 80, 60, 40, 20, 10)
delta_total <- 1e-5

lambda_s <- 5e-2; lambda_t <- 5e-2; lambda_st <- 0
nu <- 0.1

# ---------------- Helpers -----------------------
subset_fd <- function(fdobj, idx) fd(coef = fdobj$coefs[, idx, drop=FALSE],
                                     basisobj = fdobj$basis, fdnames = fdobj$fdnames)

# metrics (with worst-case NRMSE)
metrics_fd <- function(yhat_fd, ytrue_fd, grid) {
  Yhat <- eval.fd(grid, yhat_fd); Ytru <- eval.fd(grid, ytrue_fd); eps <- 1e-8
  err <- Yhat - Ytru
  rmse <- sqrt(mean(err^2))
  rmse_subj <- sqrt(colMeans(err^2))
  denom_sd <- sd(as.numeric(Ytru)) + eps
  nrmse <- rmse / denom_sd
  nrmse_worst <- max(rmse_subj) / denom_sd
  smape <- mean(2 * abs(err) / (abs(Yhat) + abs(Ytru) + eps)) * 100
  wmape <- sum(abs(err)) / (sum(abs(Ytru)) + eps) * 100
  mape  <- mean(abs(err) / (abs(Ytru) + eps)) * 100
  My <- inprod(ytrue_fd$basis, ytrue_fd$basis)
  C_hat <- yhat_fd$coefs; C_true <- ytrue_fd$coefs; C_diff <- C_hat - C_true
  il2 <- sum(colSums(C_diff * (My %*% C_diff))) / ncol(C_diff)
  denom <- sum(colSums(C_true * (My %*% C_true))) / ncol(C_true)
  ril2 <- il2 / (denom + eps)
  list(wmape=wmape, smape=smape, nrmse=nrmse, nrmse_worst=nrmse_worst,
       mape=mape, rmse=rmse, il2=il2, ril2=ril2)
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
empirical_Sres <- function(y_train_fd, q = 0.95){
  My <- inprod(y_train_fd$basis, y_train_fd$basis)
  Y  <- t(y_train_fd$coefs); ybar <- colMeans(Y); Yc <- sweep(Y, 2, ybar, "-")
  Mys <- sym_eigen_sqrt(My); Uw <- Yc %*% t(Mys$invhalf)
  norms <- sqrt(rowSums(Uw * Uw))
  as.numeric(quantile(norms, q, na.rm = TRUE))
}

# ---------------- Jobs -----------------------
set <- 1:N_global
fold_size <- N_global / folds; stopifnot(fold_size == floor(fold_size))
grid <- expand.grid(hh = seq_len(num_dup), fold = seq_len(folds),
                    eps_target = eps_grid, KEEP.OUT.ATTRS = FALSE)

n_cores <- parallel::detectCores(TRUE)
future::plan(multisession, workers = max(1, min(n_cores-1, nrow(grid))))

rows <- future_lapply(seq_len(nrow(grid)), function(i) {
  library(fda); source("functions.R"); source("functions_roundwise_dp.R")
  hh <- grid$hh[i]; fold <- grid$fold[i]; eps_t <- grid$eps_target[i]
  
  # load replicate
  env <- new.env(parent = emptyenv())
  load(sprintf("yfdobj_%d.RData", hh), envir = env)
  load(sprintf("predictorList_%d.RData", hh), envir = env)
  yfd <- get("yfdobj", env); Xlist <- get("predictorList", env)
  
  # split
  test_idx  <- set[((fold - 1) * fold_size + 1):(fold * fold_size)]
  train_idx <- setdiff(set, test_idx)
  
  subset_fd <- function(fdobj, idx) fd(coef = fdobj$coefs[, idx, drop=FALSE],
                                       basisobj = fdobj$basis, fdnames = fdobj$fdnames)
  y_train <- subset_fd(yfd, train_idx)
  y_test  <- subset_fd(yfd, test_idx)
  X_train <- lapply(Xlist, subset_fd, idx = train_idx)
  X_test  <- lapply(Xlist, subset_fd, idx = test_idx)
  
  # empirical S_res; S_est fixed (or make empirical similarly)
  # S_res <- empirical_Sres(y_train, q = 0.95)
  # S_est <- 2.0
  S_res <-  rep(3, p)
  S_est <- 3
  
  # calibrate per-round noises to hit target ε with EXACT R_fixed rounds
  if (is.finite(eps_t) && eps_t > 0) {
    rho_t   <- rho_from_eps(eps_t, delta_total)
    rho_res <- w_split * rho_t
    rho_est <- (1 - w_split) * rho_t
    s_res <- S_res * sqrt(2 * R_fixed / pmax(rho_res, 1e-12))
    s_est <- S_est * sqrt(2 * R_fixed / pmax(rho_est, 1e-12))
  } else { s_res <- 0; s_est <- 0 }
  
  t0 <- Sys.time()
  fit <- roundwise_dp_vfl_boost(
    xfd_list = X_train, yfd = y_train,
    S_res = S_res, s_res = s_res,
    S_est = S_est, s_est = s_est,
    Omega_x_list = NULL, Omega_y = NULL,
    lambda_s = lambda_s, lambda_t = lambda_t, lambda_st = lambda_st,
    nu = nu, R_fixed = R_fixed,
    use_crossfit = TRUE, delta_total = delta_total
  )
  time_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  # predict & metrics
  yhat_test <- predict_roundwise_dp(fit, X_test)
  mets <- metrics_fd(yhat_test, y_test, grid = tgrid)
  cls  <- sens_spec_from_fd(yhat_test, y_test, y_train, grid = tgrid)
  
  # communication per ROUND * R_fixed
  Ntr <- length(train_idx); Qx <- nrow(X_train[[1]]$coefs); Qy <- nrow(y_train$coefs)
  comm_mb <- ( (p * Ntr * Qy) + (Qx * Qy) ) * 8 * R_fixed / (1024^2)
  
  data.frame(
    hh = hh, fold = fold, eps_target = eps_t, rounds = R_fixed,
    WMAPE = mets$wmape, SMAPE = mets$smape, NRMSE = mets$nrmse, NRMSE_worst = mets$nrmse_worst,
    MAPE = mets$mape,   RMSE  = mets$rmse,
    IL2 = mets$il2,     RIL2  = mets$ril2,
    Sensitivity = cls$sensitivity, Specificity = cls$specificity,
    Eps_global = eps_t,                 # <-- fixed, equals target
    Time_sec = time_sec, Comm_MB = comm_mb,
    stringsAsFactors = FALSE, check.names = FALSE
  )
}, future.seed = TRUE)

ok <- vapply(rows, function(x) is.data.frame(x) && nrow(x) == 1, logical(1))
res <- if (any(ok)) do.call(rbind, rows[ok]) else data.frame()

if (nrow(res)) {
  write.csv(res, "roundwise_dp_perjob.csv", row.names = FALSE)
  
  metrics <- c("WMAPE","SMAPE","NRMSE","NRMSE_worst","MAPE","RMSE","IL2","RIL2",
               "Sensitivity","Specificity","Eps_global","Time_sec","Comm_MB")
  
  agg_mean <- aggregate(res[, metrics],
                        by = list(eps_target = res$eps_target),
                        FUN = function(x) mean(x, na.rm = TRUE))
  agg_sd <- aggregate(res[, metrics],
                      by = list(eps_target = res$eps_target),
                      FUN = function(x) sd(x, na.rm = TRUE))
  
  names(agg_mean) <- c("eps_target", paste0(metrics, "_mean"))
  names(agg_sd)   <- c("eps_target", paste0(metrics, "_sd"))
  summary_df <- merge(agg_mean, agg_sd, by = "eps_target", all = TRUE)
  
  write.csv(summary_df, "roundwise_dp_summary.csv", row.names = FALSE)
  print(summary_df[order(summary_df$eps_target, decreasing = TRUE), ], row.names = FALSE)
} else {
  cat("[WARN] No successful jobs.\n")
}

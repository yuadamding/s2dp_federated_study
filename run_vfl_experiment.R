# ============================================================
# Parallel runner for DP-corrected VFL functional boosting
# - VFL semantics: same individuals across workers; workers own
#   disjoint subsets of predictors (features).
# - Jobs = (duplicate x workers x fold). Parallelized with future.apply.
# - Uses vfl_dp_foboost() with DP-aware selection & CV stopping.
# ============================================================

root_dir <- "/Users/yuding/Dropbox/VFL_code"
setwd(root_dir)

suppressPackageStartupMessages({
  library(fda)
  library(future.apply)
})

source("functions.R")     # DP FoF + boosting

# ---------------------- Settings ----------------------------
numworkersseq     <- c(2, 4, 6, 8, 10)
num_duplicate     <- 10
folds_per_worker  <- 4
N_global          <- 500
p                 <- 20
rangeval          <- c(0, 100)
t_basis           <- 20
basisobj          <- create.bspline.basis(rangeval, nbasis = t_basis)
tgrid             <- seq(rangeval[1], rangeval[2], by = 1)

# DP hyperparameters
sx_default <- 0.2
Sx_mode    <- "empirical"   # or "fixed"
Sx_fixed   <- 3.0

# FoF penalties / boosting controls
lambda_s <- 5e-2
lambda_t <- 5e-2
lambda_st <- 0
nu <- 0.3
max_steps <- 30
use_crossfit <- TRUE
use_aic <- "spherical"
use_aic_c <- TRUE
df_K <- 5
patience <- 6
min_steps <- 10
sse_correct_dp <- FALSE

# ---------------------- Helpers -----------------------------
owner_of_feature <- function(j, numworkers) {
  ((j - 1) %% numworkers) + 1L
}

subset_fd <- function(fdobj, idx) {
  stopifnot(is.fd(fdobj))
  co <- fdobj$coefs
  if (is.matrix(co)) {
    fd(coef = co[, idx, drop = FALSE], basisobj = fdobj$basis, fdnames = fdobj$fdnames)
  } else {
    stop("3D fd$coefs not supported.")
  }
}

# VFL-aware per-worker loader: allow NULL for non-owned features
load_worker_data_nullable <- function(l, numworkers, hh, p_expected) {
  fy <- sprintf("yfdobj_%d_%d_%d.RData", l, numworkers, hh)
  fx <- sprintf("predictorLst_%d_%d_%d.RData", l, numworkers, hh)
  if (!file.exists(fy)) stop(sprintf("[load_worker_data] Missing file: %s", fy))
  if (!file.exists(fx)) stop(sprintf("[load_worker_data] Missing file: %s", fx))
  
  env <- new.env(parent = emptyenv())
  load(fy, envir = env)
  load(fx, envir = env)
  
  if (!exists("yfdobj", envir = env))
    stop(sprintf("[load_worker_data] Object 'yfdobj' not found inside %s", fy))
  if (!exists("predictorLst", envir = env))
    stop(sprintf("[load_worker_data] Object 'predictorLst' not found inside %s", fx))
  
  yfdobj       <- get("yfdobj", envir = env)
  predictorLst <- get("predictorLst", envir = env)
  if (!inherits(yfdobj, "fd"))
    stop(sprintf("[load_worker_data] 'yfdobj' in %s is not an 'fd' object", fy))
  if (!is.list(predictorLst))
    stop(sprintf("[load_worker_data] 'predictorLst' in %s is not a list", fx))
  
  if (length(predictorLst) != p_expected) {
    stop(sprintf("[load_worker_data] Length mismatch in %s: length=%d, expected p=%d",
                 fx, length(predictorLst), p_expected))
  }
  # Check owned features only
  for (j in seq_len(p_expected)) {
    owns <- (owner_of_feature(j, numworkers) == l)
    xj <- predictorLst[[j]]
    if (owns && is.null(xj)) {
      stop(sprintf("[load_worker_data] Worker %d should own feature %d but it's NULL (k=%d, hh=%d)",
                   l, j, numworkers, hh))
    }
    if (!is.null(xj)) {
      if (!inherits(xj, "fd"))
        stop(sprintf("[load_worker_data] predictorLst[[%d]] at worker %d not 'fd'", j, l))
      # basis / N check
      if (xj$basis$nbasis != yfdobj$basis$nbasis)
        stop(sprintf("[load_worker_data] nbasis mismatch j=%d at worker %d", j, l))
      if (ncol(xj$coefs) != ncol(yfdobj$coefs))
        stop(sprintf("[load_worker_data] N mismatch j=%d at worker %d", j, l))
    }
  }
  list(yfdobj = yfdobj, predictorLst = predictorLst)
}

# Build per-predictor global X list by selecting the owning worker's fd;
# Y is taken from worker 1 (identical across workers by construction).
build_global_dataset_vfl <- function(numworkers, hh, train_idx, test_idx) {
  p_expected <- p
  # load all workers for (k= numworkers, hh)
  wrk <- lapply(seq_len(numworkers), function(l) {
    load_worker_data_nullable(l, numworkers, hh, p_expected)
  })
  # Y
  Y_full <- wrk[[1]]$yfdobj
  # X per feature j: take from the owning worker
  Xlist_full <- vector("list", p_expected)
  for (j in seq_len(p_expected)) {
    l_owner <- owner_of_feature(j, numworkers)
    Xlist_full[[j]] <- wrk[[l_owner]]$predictorLst[[j]]
    if (is.null(Xlist_full[[j]]))
      stop(sprintf("[build_global_dataset_vfl] Feature %d missing at owner %d (k=%d, hh=%d)",
                   j, l_owner, numworkers, hh))
  }
  # Split
  Xlist_train <- lapply(Xlist_full, subset_fd, idx = train_idx)
  Xlist_test  <- lapply(Xlist_full, subset_fd, idx = test_idx)
  Y_train     <- subset_fd(Y_full,        idx = train_idx)
  Y_test      <- subset_fd(Y_full,        idx = test_idx)
  list(Xlist_train = Xlist_train, Xlist_test = Xlist_test, Y_train = Y_train, Y_test = Y_test)
}

metrics_fd <- function(yhat_fd, ytrue_fd, grid) {
  Yhat <- eval.fd(grid, yhat_fd)
  Ytru <- eval.fd(grid, ytrue_fd)
  stopifnot(all(dim(Yhat) == dim(Ytru)))
  eps <- 1e-8
  rmse  <- sqrt(mean((Yhat - Ytru)^2))
  nrmse <- rmse / (sd(as.numeric(Ytru)) + eps)
  smape <- mean(2 * abs(Yhat - Ytru) / (abs(Yhat) + abs(Ytru) + eps)) * 100
  wmape <- sum(abs(Yhat - Ytru)) / (sum(abs(Ytru)) + eps) * 100
  mape  <- mean(abs(Yhat - Ytru) / (abs(Ytru) + eps)) * 100
  
  # Functional L2 metrics using basis Gram My
  My <- inprod(ytrue_fd$basis, ytrue_fd$basis)
  C_hat  <- yhat_fd$coefs
  C_true <- ytrue_fd$coefs
  C_diff <- C_hat - C_true
  il2  <- sum(colSums(C_diff * (My %*% C_diff))) / ncol(C_diff)
  denom <- sum(colSums(C_true * (My %*% C_true))) / ncol(C_true)
  ril2 <- il2 / (denom + eps)
  
  list(rmse = rmse, nrmse = nrmse, smape = smape, wmape = wmape, mape = mape,
       il2 = il2, ril2 = ril2)
}

compute_sens_spec <- function(selected_idx, active_idx, p) {
  sel <- rep(0L, p)
  if (length(selected_idx) > 0) sel[unique(selected_idx)] <- 1L
  sens <- mean(sel[active_idx] == 1L)
  spec <- mean(sel[setdiff(1:p, active_idx)] == 0L)
  c(sensitivity = sens, specificity = spec)
}

comm_cost_mb <- function(Ntrain, Qx, p) {
  bytes <- p * (Ntrain * Qx + Qx * Qx) * 8
  bytes / (1024^2)
}

adapt_Sx <- function(Xlist_train, mode = c("fixed", "empirical"), Sx_fixed = 3.0) {
  mode <- match.arg(mode)
  if (mode == "fixed") return(rep(Sx_fixed, length(Xlist_train)))
  Sx_vec <- numeric(length(Xlist_train))
  for (j in seq_along(Xlist_train)) {
    fdj <- Xlist_train[[j]]
    Mx <- inprod(fdj$basis, fdj$basis)
    C  <- fdj$coefs
    norms <- sqrt(colSums(C * (Mx %*% C)))
    Sx_vec[j] <- as.numeric(quantile(norms, 0.95, na.rm = TRUE))
    if (!is.finite(Sx_vec[j]) || Sx_vec[j] <= 0) Sx_vec[j] <- 3.0
  }
  Sx_vec
}

# ---------------------- Job table ----------------------------
set <- 1:N_global
fold_size <- N_global / folds_per_worker
stopifnot(fold_size == floor(fold_size))

tasks <- expand.grid(
  hh     = seq_len(num_duplicate),
  nw_i   = seq_along(numworkersseq),
  fold   = seq_len(folds_per_worker),
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
)
tasks$numworkers <- numworkersseq[tasks$nw_i]

# ---------------------- Parallel plan -----------------------
n_cores_avail <- parallel::detectCores(logical = TRUE)
n_workers <- max(1, min(n_cores_avail - 1, nrow(tasks)))
future::plan(multisession, workers = n_workers)

# ---------------------- Run all jobs in parallel ------------
results_list <- future_lapply(
  X = seq_len(nrow(tasks)),
  FUN = function(i_job) {
    # Isolate job specs
    hh <- tasks$hh[i_job]
    numworkers <- tasks$numworkers[i_job]
    fold <- tasks$fold[i_job]
    
    # Local session setup
    setwd(root_dir)
    suppressPackageStartupMessages({ library(fda) })
    source("functions.R")
    load("truth_active_idx.RData")  # active_idx
    
    # Reproducible seed per job
    set.seed(10^6 * hh + 10^3 * numworkers + fold)
    
    # Train/test indices (global; same individuals across workers)
    test_idx  <- set[((fold - 1) * fold_size + 1):(fold * fold_size)]
    train_idx <- setdiff(set, test_idx)
    
    # Build dataset (VFL semantics)
    ds <- build_global_dataset_vfl(numworkers, hh, train_idx, test_idx)
    Xlist_train <- ds$Xlist_train
    Xlist_test  <- ds$Xlist_test
    Y_train     <- ds$Y_train
    Y_test      <- ds$Y_test
    
    # DP clipping & noise
    Sx_vec <- adapt_Sx(Xlist_train, mode = Sx_mode, Sx_fixed = Sx_fixed)
    sx_vec <- rep(sx_default, p)
    
    # Fit
    t0 <- Sys.time()
    fit <- vfl_dp_foboost(
      xfd_list = Xlist_train,
      yfd      = Y_train,
      Sx_vec   = Sx_vec,
      sx_vec   = sx_vec,
      Omega_x_list = NULL, Omega_y = NULL,
      lambda_s = lambda_s, lambda_t = lambda_t, lambda_st = lambda_st,
      nu = nu, max_steps = max_steps,
      crossfit = use_crossfit,
      stop_mode = "cv",
      min_steps = min_steps,
      aic = use_aic, aic_c = use_aic_c,
      df_K = df_K, patience = patience,
      sse_correct_dp = sse_correct_dp
    )
    t1 <- Sys.time()
    time_sec <- as.numeric(difftime(t1, t0, units = "secs"))
    
    # Predict & metrics
    yhat_test <- predict_vfl_dp_foboost(fit, Xlist_test)
    metrics <- metrics_fd(yhat_test, Y_test, grid = tgrid)
    
    # Sensitivity/Specificity
    ss <- compute_sens_spec(fit$selected, active_idx, p)
    
    # Communication cost (MB) for this fold/training size
    Ntrain <- length(train_idx)
    comm_mb <- comm_cost_mb(Ntrain, t_basis, p)
    
    data.frame(
      hh = hh, fold = fold, numworkers = numworkers, nw_i = which(numworkersseq == numworkers),
      WMAPE = metrics$wmape, SMAPE = metrics$smape, NRMSE = metrics$nrmse,
      MAPE  = metrics$mape, RMSE  = metrics$rmse,
      IL2   = metrics$il2,  RIL2  = metrics$ril2,
      Sensitivity = ss["sensitivity"], Specificity = ss["specificity"],
      Time_sec = time_sec, Comm_MB = comm_mb,
      check.names = FALSE
    )
  },
  future.seed = TRUE
)

# ---------------------- Assemble results --------------------
res_df <- do.call(rbind, results_list)

K <- length(numworkersseq)
MAPE_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
SMAPE_arr <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
WMAPE_arr <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
RMSE_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
NRMSE_arr <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
IL2_arr   <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
RIL2_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
Sens_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
Spec_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
Time_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
Comm_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))

for (r in seq_len(nrow(res_df))) {
  i  <- res_df$nw_i[r]
  hh <- res_df$hh[r]
  f  <- res_df$fold[r]
  WMAPE_arr[i, hh, f] <- res_df$WMAPE[r]
  SMAPE_arr[i, hh, f] <- res_df$SMAPE[r]
  NRMSE_arr[i, hh, f] <- res_df$NRMSE[r]
  MAPE_arr[i, hh, f]  <- res_df$MAPE[r]
  RMSE_arr[i, hh, f]  <- res_df$RMSE[r]
  IL2_arr[i, hh, f]   <- res_df$IL2[r]
  RIL2_arr[i, hh, f]  <- res_df$RIL2[r]
  Sens_arr[i, hh, f]  <- res_df$Sensitivity[r]
  Spec_arr[i, hh, f]  <- res_df$Specificity[r]
  Time_arr[i, hh, f]  <- res_df$Time_sec[r]
  Comm_arr[i, hh, f]  <- res_df$Comm_MB[r]
}

res <- data.frame(
  workers              = numworkersseq,
  WMAPE_mean           = apply(WMAPE_arr, 1, mean, na.rm = TRUE),
  WMAPE_sd             = apply(WMAPE_arr, 1, sd,   na.rm = TRUE),
  WMAPE_worst          = apply(WMAPE_arr, 1, max,  na.rm = TRUE),
  sMAPE_mean           = apply(SMAPE_arr, 1, mean, na.rm = TRUE),
  NRMSE_mean           = apply(NRMSE_arr, 1, mean, na.rm = TRUE),
  MAPE_mean            = apply(MAPE_arr,  1, mean, na.rm = TRUE),
  IL2_mean             = apply(IL2_arr,   1, mean, na.rm = TRUE),
  RIL2_mean            = apply(RIL2_arr,  1, mean, na.rm = TRUE),
  Sensitivity_mean     = apply(Sens_arr,  1, mean, na.rm = TRUE),
  Specificity_mean     = apply(Spec_arr,  1, mean, na.rm = TRUE),
  Time_hours_mean      = apply(Time_arr,  1, mean, na.rm = TRUE) / 3600,
  Comm_MB_mean         = apply(Comm_arr,  1, mean, na.rm = TRUE)
)

print(res)

save(MAPE_arr, SMAPE_arr, WMAPE_arr, RMSE_arr, NRMSE_arr,
     IL2_arr, RIL2_arr,
     Sens_arr, Spec_arr, Time_arr, Comm_arr, res, res_df,
     file = "vfl_dp_foboost_results.RData")

future::plan(sequential)

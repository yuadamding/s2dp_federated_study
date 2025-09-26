# ============================================================
# Parallel runner for DP-corrected VFL functional boosting
# - Each (duplicate x workers x fold) is a parallel job
# - Keeps DP-aware selection & CV early stopping
# ============================================================

root_dir <- "/Users/yuding/Dropbox/VFL_code"
setwd(root_dir)

suppressPackageStartupMessages({
  library(fda)
  library(future.apply)   # <-- parallel
})

source("functions.R")     # your DP FoF + boosting implementation

# ---------------------- Settings ----------------------------
numworkersseq     <- c(2, 4, 6, 8, 10, 20, 50)
num_duplicate     <- 10
folds_per_worker  <- 4
samplesPerWorker  <- 100
p                 <- 20
rangeval          <- c(0, 100)
t_basis           <- 20
basisobj          <- create.bspline.basis(rangeval, nbasis = t_basis)
tgrid             <- seq(rangeval[1], rangeval[2], by = 1)

# DP hyperparameters
sx_default <- 0.2   # tune with ε,δ; can go down to 0.1 for sanity checks

# FoF penalties / boosting controls
lambda_s <- 5e-2
lambda_t <- 5e-2
lambda_st <- 0
nu <- 0.3
max_steps <- 30
use_crossfit <- TRUE   # selection
use_aic <- "spherical"
use_aic_c <- TRUE
df_K <- 5
patience <- 6
min_steps <- 10
sse_correct_dp <- FALSE  # not used for CV stopping

# ---------------------- Helpers -----------------------------

subset_fd <- function(fdobj, idx) {
  stopifnot(is.fd(fdobj))
  co <- fdobj$coefs
  if (is.matrix(co)) {
    fd(coef = co[, idx, drop = FALSE], basisobj = fdobj$basis, fdnames = fdobj$fdnames)
  } else {
    stop("3D fd$coefs not supported.")
  }
}

build_global_dataset <- function(numworkers, hh, train_idx_global, test_idx_global) {
  Xlist_full <- vector("list", p)
  Y_full <- NULL
  for (l in 1:numworkers) {
    load(sprintf("yfdobj_%d_%d_%d.RData", l, numworkers, hh))
    load(sprintf("predictorLst_%d_%d_%d.RData", l, numworkers, hh))
    if (length(predictorLst) != p) {
      stop(sprintf("predictorLst length (%d) != p (%d) for worker %d", length(predictorLst), p, l))
    }
    if (is.null(Y_full)) {
      Y_full <- yfdobj
    } else {
      Y_full$coefs <- cbind(Y_full$coefs, yfdobj$coefs)
    }
    for (j in 1:p) {
      if (is.null(Xlist_full[[j]])) {
        Xlist_full[[j]] <- predictorLst[[j]]
      } else {
        Xlist_full[[j]]$coefs <- cbind(Xlist_full[[j]]$coefs, predictorLst[[j]]$coefs)
      }
    }
  }
  Xlist_train <- lapply(Xlist_full, subset_fd, idx = train_idx_global)
  Xlist_test  <- lapply(Xlist_full, subset_fd, idx = test_idx_global)
  Y_train     <- subset_fd(Y_full, train_idx_global)
  Y_test      <- subset_fd(Y_full, test_idx_global)
  list(Xlist_train = Xlist_train, Xlist_test = Xlist_test, Y_train = Y_train, Y_test = Y_test)
}

setGenerator_vfl <- function(base_idx, samplesPerWorker, numworkers) {
  offs <- (0:(numworkers - 1)) * samplesPerWorker
  as.vector(unlist(lapply(offs, function(o) base_idx + o)))
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
  list(rmse = rmse, nrmse = nrmse, smape = smape, wmape = wmape, mape = mape)
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

adapt_Sx <- function(Xlist_train) {
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

# ---------------------- Build job table ---------------------------
set <- 1:samplesPerWorker
fold_size <- samplesPerWorker / folds_per_worker
stopifnot(fold_size == floor(fold_size))

tasks <- expand.grid(
  hh     = seq_len(num_duplicate),
  nw_i   = seq_along(numworkersseq),
  fold   = seq_len(folds_per_worker),
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
)
tasks$numworkers <- numworkersseq[tasks$nw_i]

# ---------------------- Parallel plan ---------------------------
# Use up to (cores - 1) workers, but not more than #tasks
n_cores_avail <- parallel::detectCores(logical = TRUE)
n_workers <- max(1, min(n_cores_avail - 1, nrow(tasks)))
future::plan(multisession, workers = n_workers)

# ---------------------- Run all jobs in parallel ----------------
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
    load("truth_active_idx.RData")  # provides active_idx
    
    # Reproducible seed per job (drives A/B split and any RNG inside)
    set.seed(10^6 * hh + 10^3 * numworkers + fold)
    
    # Train/test indices (global)
    test_base  <- set[((fold - 1) * fold_size + 1):(fold * fold_size)]
    train_base <- setdiff(set, test_base)
    test_idx_global  <- setGenerator_vfl(test_base,  samplesPerWorker, numworkers)
    train_idx_global <- setGenerator_vfl(train_base, samplesPerWorker, numworkers)
    
    # Build dataset
    ds <- build_global_dataset(numworkers, hh, train_idx_global, test_idx_global)
    Xlist_train <- ds$Xlist_train
    Xlist_test  <- ds$Xlist_test
    Y_train     <- ds$Y_train
    Y_test      <- ds$Y_test
    
    # DP clipping & noise
    Sx_vec <- adapt_Sx(Xlist_train)
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
      stop_mode = "cv",      # CV stopping
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
    Ntrain <- length(train_idx_global)
    comm_mb <- comm_cost_mb(Ntrain, t_basis, p)
    
    list(
      hh = hh, fold = fold, numworkers = numworkers, nw_i = which(numworkersseq == numworkers),
      WMAPE = metrics$wmape, SMAPE = metrics$smape, NRMSE = metrics$nrmse,
      MAPE  = metrics$mape, RMSE  = metrics$rmse,
      Sensitivity = ss["sensitivity"], Specificity = ss["specificity"],
      Time_sec = time_sec, Comm_MB = comm_mb
    )
  },
  future.seed = TRUE   # independent, reproducible RNG streams
)

# ---------------------- Assemble results ----------------------
# Convert to data.frame
res_df <- do.call(rbind, lapply(results_list, function(x) as.data.frame(x, check.names = FALSE)))

# Preallocate arrays to match your previous output format
K <- length(numworkersseq)
MAPE_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
SMAPE_arr <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
WMAPE_arr <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
RMSE_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
NRMSE_arr <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
Sens_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
Spec_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
Time_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
Comm_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))

# Fill arrays from res_df
for (r in seq_len(nrow(res_df))) {
  i <- res_df$nw_i[r]
  hh <- res_df$hh[r]
  f  <- res_df$fold[r]
  WMAPE_arr[i, hh, f] <- res_df$WMAPE[r]
  SMAPE_arr[i, hh, f] <- res_df$SMAPE[r]
  NRMSE_arr[i, hh, f] <- res_df$NRMSE[r]
  MAPE_arr[i, hh, f]  <- res_df$MAPE[r]
  RMSE_arr[i, hh, f]  <- res_df$RMSE[r]
  Sens_arr[i, hh, f]  <- res_df$Sensitivity[r]
  Spec_arr[i, hh, f]  <- res_df$Specificity[r]
  Time_arr[i, hh, f]  <- res_df$Time_sec[r]
  Comm_arr[i, hh, f]  <- res_df$Comm_MB[r]
}

# Summaries
res <- data.frame(
  workers              = numworkersseq,
  WMAPE_mean           = apply(WMAPE_arr, 1, mean, na.rm = TRUE),
  WMAPE_sd             = apply(WMAPE_arr, 1, sd,   na.rm = TRUE),
  WMAPE_worst          = apply(WMAPE_arr, 1, max,  na.rm = TRUE),
  sMAPE_mean           = apply(SMAPE_arr, 1, mean, na.rm = TRUE),
  NRMSE_mean           = apply(NRMSE_arr, 1, mean, na.rm = TRUE),
  MAPE_mean            = apply(MAPE_arr,  1, mean, na.rm = TRUE),
  Sensitivity_mean     = apply(Sens_arr,  1, mean, na.rm = TRUE),
  Specificity_mean     = apply(Spec_arr,  1, mean, na.rm = TRUE),
  Time_hours_mean      = apply(Time_arr,  1, mean, na.rm = TRUE) / 3600,
  Comm_MB_mean         = apply(Comm_arr,  1, mean, na.rm = TRUE)
)

print(res)

save(MAPE_arr, SMAPE_arr, WMAPE_arr, RMSE_arr, NRMSE_arr,
     Sens_arr, Spec_arr, Time_arr, Comm_arr, res, res_df,
     file = "vfl_dp_foboost_results.RData")

# (Optional) reset to sequential plan
future::plan(sequential)

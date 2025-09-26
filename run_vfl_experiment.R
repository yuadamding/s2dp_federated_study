# ============================================================
# Parallel runner for DP-corrected VFL functional boosting
# - Varies number of parties J (vertical split of predictors), constant N
# - Each (replicate x J x fold) is a parallel job
# - DP-aware selection + CV early stopping
# ============================================================

root_dir <- "/Users/yuding/Dropbox/VFL_code"
setwd(root_dir)

suppressPackageStartupMessages({
  library(fda)
  library(future.apply)
})

source("functions.R")     # DP FoF + boosting implementation

# ---------------------- Settings ----------------------------
parties_seq       <- c(2, 4, 6, 8, 10)   # number of parties J
num_duplicate     <- 10
folds_per_worker  <- 4
p                 <- 20
rangeval          <- c(0, 100)
t_basis           <- 20
basisobj          <- create.bspline.basis(rangeval, nbasis = t_basis)
tgrid             <- seq(rangeval[1], rangeval[2], by = 1)

# DP hyperparameters
sx_default <- 0.2         # base noise std per predictor (whitened mechanism)
keep_total_privacy <- FALSE
# If TRUE: scale noise ~ sqrt(J) within each party to keep total zCDP ~ constant across J
# (approximate, assumes one mechanism per party; we still release per predictor)

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
sse_correct_dp <- FALSE   # not used in CV mode

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

load_replicate <- function(hh) {
  load(sprintf("yfdobj_%d.RData", hh))        # yfdobj
  load(sprintf("predictorLst_%d.RData", hh))  # predictorLst (length p)
  list(Y_full = yfdobj, Xlist_full = predictorLst)
}

assign_to_parties <- function(p, J, mode = c("roundrobin", "contiguous")) {
  mode <- match.arg(mode)
  if (mode == "roundrobin") {
    split(1:p, ((0:(p-1)) %% J) + 1)
  } else {
    # contiguous blocks
    cuts <- cut(1:p, breaks = J, labels = FALSE)
    split(1:p, cuts)
  }
}

# Build train/test subsets (same indices for all predictors)
build_dataset_split <- function(Xlist_full, Y_full, train_idx, test_idx) {
  Xlist_train <- lapply(Xlist_full, subset_fd, idx = train_idx)
  Xlist_test  <- lapply(Xlist_full, subset_fd, idx = test_idx)
  Y_train     <- subset_fd(Y_full, train_idx)
  Y_test      <- subset_fd(Y_full, test_idx)
  list(Xlist_train = Xlist_train, Xlist_test = Xlist_test,
       Y_train = Y_train, Y_test = Y_test)
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

# Communication cost if each party transmits its block once:
#  For party g with m_g predictors of Qx basis each:
#   bytes_g ≈ (Ntrain * (m_g * Qx) + (m_g * Qx)^2) * 8
comm_cost_mb_parties <- function(Ntrain, Qx, parties) {
  bytes <- 0
  for (g in seq_along(parties)) {
    m_g <- length(parties[[g]])
    qg  <- m_g * Qx
    bytes <- bytes + (Ntrain * qg + qg * qg) * 8
  }
  bytes / (1024^2)
}

# Per-predictor clipping radii
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

# Scale per-predictor sx to keep total privacy roughly constant across J (optional)
scale_sx_by_parties <- function(sx_base, parties, keep_total_privacy = FALSE) {
  if (!keep_total_privacy) return(rep(sx_base, p))
  # Approximation: if each party would use one Gaussian mechanism on its block,
  # and we instead add noise per predictor, scale noise as ~ sqrt(J) to keep total zCDP ~ const.
  rep(sx_base * sqrt(length(parties)), p)
}

# ---------------------- Build job table ---------------------------
load("truth_active_idx.RData")  # active_idx from generator

tasks <- expand.grid(
  hh   = seq_len(num_duplicate),
  Ji   = seq_along(parties_seq),
  fold = seq_len(folds_per_worker),
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
)
tasks$J <- parties_seq[tasks$Ji]

# ---------------------- Parallel plan ---------------------------
n_cores_avail <- parallel::detectCores(logical = TRUE)
n_workers <- max(1, min(n_cores_avail - 1, nrow(tasks)))
future::plan(multisession, workers = n_workers)

# ---------------------- Run all jobs in parallel ----------------
results_list <- future_lapply(
  X = seq_len(nrow(tasks)),
  FUN = function(i_job) {
    hh <- tasks$hh[i_job]
    J  <- tasks$J[i_job]
    fold <- tasks$fold[i_job]
    
    setwd(root_dir)
    suppressPackageStartupMessages({ library(fda) })
    source("functions.R")
    load("truth_active_idx.RData")
    
    set.seed(10^6 * hh + 10^3 * J + fold)
    
    # Load the replicate (constant N)
    rep_data <- load_replicate(hh)
    Y_full <- rep_data$Y_full
    Xlist_full <- rep_data$Xlist_full
    stopifnot(length(Xlist_full) == p)
    
    N_total <- ncol(Y_full$coefs)
    # CV folds on base 1:N_total
    fold_size <- floor(N_total / folds_per_worker)
    idx_base  <- 1:N_total
    test_base  <- idx_base[((fold - 1) * fold_size + 1) : (fold * fold_size)]
    train_base <- setdiff(idx_base, test_base)
    
    ds <- build_dataset_split(Xlist_full, Y_full, train_base, test_base)
    Xlist_train <- ds$Xlist_train
    Xlist_test  <- ds$Xlist_test
    Y_train     <- ds$Y_train
    Y_test      <- ds$Y_test
    
    # Parties: split predictors vertically
    parties <- assign_to_parties(p, J, mode = "roundrobin")
    
    # DP: per-predictor clipping radii and noise stds
    Sx_vec <- adapt_Sx(Xlist_train)
    sx_vec <- scale_sx_by_parties(sx_default, parties, keep_total_privacy)
    
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
    
    # Communication cost (per party, single release)
    Ntrain <- length(train_base)
    comm_mb <- comm_cost_mb_parties(Ntrain, t_basis, parties)
    
    list(
      hh = hh, fold = fold, J = J, Ji = which(parties_seq == J),
      WMAPE = metrics$wmape, SMAPE = metrics$smape, NRMSE = metrics$nrmse,
      MAPE  = metrics$mape, RMSE  = metrics$rmse,
      Sensitivity = ss["sensitivity"], Specificity = ss["specificity"],
      Time_sec = time_sec, Comm_MB = comm_mb
    )
  },
  future.seed = TRUE
)

# ---------------------- Assemble results ----------------------
res_df <- do.call(rbind, lapply(results_list, function(x) as.data.frame(x, check.names = FALSE)))

K <- length(parties_seq)
MAPE_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
SMAPE_arr <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
WMAPE_arr <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
RMSE_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
NRMSE_arr <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
Sens_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
Spec_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
Time_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))
Comm_arr  <- array(NA_real_, dim = c(K, num_duplicate, folds_per_worker))

for (r in seq_len(nrow(res_df))) {
  i  <- res_df$Ji[r]
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

res <- data.frame(
  parties              = parties_seq,
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

future::plan(sequential)

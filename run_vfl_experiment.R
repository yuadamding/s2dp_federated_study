# ============================================================
# Accuracy vs Privacy Budget Sweep for DP VFL Functional Boosting
# - VFL semantics: same individuals across workers; each worker owns
#   a disjoint subset of predictors; 'yfdobj' identical across workers.
# - Sweeps Gaussian DP noise sx; computes epsilon via zCDP -> (ε, δ).
# - Fully robust to empty job sets and partial failures.
# ============================================================

root_dir <- "/Users/yuding/Dropbox/VFL_code"
setwd(root_dir)

suppressPackageStartupMessages({
  library(fda)
  library(future.apply)
})

source("functions.R")        # contains vfl_dp_foboost(), predict_vfl_dp_foboost()

# ---------------------- Settings ----------------------------
numworkersseq     <- c(2, 4, 6, 8, 10)
num_duplicate     <- 10
folds_per_worker  <- 4
N_global          <- 500      # must match your generator_vfl.R
p                 <- 20
rangeval          <- c(0, 100)
t_basis           <- 20
basisobj          <- create.bspline.basis(rangeval, nbasis = t_basis)
tgrid             <- seq(rangeval[1], rangeval[2], by = 1)

# DP sweep & accounting
sx_grid           <- c(0.0, 0.1, 0.2, 0.3)  # noise std in whitened coords
Sx_mode           <- "fixed"                # "fixed" => DP-correct ε, "empirical" for diagnostics
Sx_fixed          <- 3.0                    # per-feature clipping radius in whitened norm
delta_total       <- 1e-5
compose_over_cv   <- FALSE                  # one release -> no extra composition

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
owner_of_feature <- function(j, numworkers) ((j - 1) %% numworkers) + 1L

subset_fd <- function(fdobj, idx) {
  stopifnot(is.fd(fdobj))
  co <- fdobj$coefs
  if (is.matrix(co)) {
    fd(coef = co[, idx, drop = FALSE], basisobj = fdobj$basis, fdnames = fdobj$fdnames)
  } else stop("3D fd$coefs not supported.")
}

# VFL-aware per-worker loader: allow NULL for non-owned features
load_worker_data_nullable <- function(l, numworkers, hh, p_expected) {
  fy <- sprintf("yfdobj_%d_%d_%d.RData", l, numworkers, hh)
  fx <- sprintf("predictorLst_%d_%d_%d.RData", l, numworkers, hh)
  if (!file.exists(fy)) stop(sprintf("[load_worker_data] Missing file: %s", fy))
  if (!file.exists(fx)) stop(sprintf("[load_worker_data] Missing file: %s", fx))
  
  env <- new.env(parent = emptyenv())
  load(fy, envir = env); load(fx, envir = env)
  
  if (!exists("yfdobj", envir = env)) stop(sprintf("[load_worker_data] 'yfdobj' not in %s", fy))
  if (!exists("predictorLst", envir = env)) stop(sprintf("[load_worker_data] 'predictorLst' not in %s", fx))
  
  yfdobj       <- get("yfdobj", envir = env)
  predictorLst <- get("predictorLst", envir = env)
  
  if (!inherits(yfdobj, "fd")) stop(sprintf("[load_worker_data] yfdobj not 'fd' (%s)", fy))
  if (!is.list(predictorLst)) stop(sprintf("[load_worker_data] predictorLst not list (%s)", fx))
  if (length(predictorLst) != p_expected)
    stop(sprintf("[load_worker_data] length(predictorLst)=%d != p=%d in %s", length(predictorLst), p_expected, fx))
  
  for (j in seq_len(p_expected)) {
    owns <- (owner_of_feature(j, numworkers) == l)
    xj <- predictorLst[[j]]
    if (owns && is.null(xj))
      stop(sprintf("[load_worker_data] worker %d should own feature %d but is NULL", l, j))
    if (!is.null(xj)) {
      if (!inherits(xj, "fd"))
        stop(sprintf("[load_worker_data] predictorLst[[%d]] at worker %d not 'fd'", j, l))
      if (xj$basis$nbasis != yfdobj$basis$nbasis)
        stop(sprintf("[load_worker_data] nbasis mismatch j=%d at worker %d", j, l))
      if (ncol(xj$coefs) != ncol(yfdobj$coefs))
        stop(sprintf("[load_worker_data] N mismatch j=%d at worker %d", j, l))
    }
  }
  list(yfdobj = yfdobj, predictorLst = predictorLst)
}

# Build X list by choosing the owning worker's feature; Y from worker 1
build_global_dataset_vfl <- function(numworkers, hh, train_idx, test_idx) {
  p_expected <- p
  wrk <- lapply(seq_len(numworkers), function(l) {
    load_worker_data_nullable(l, numworkers, hh, p_expected)
  })
  Y_full <- wrk[[1]]$yfdobj
  X_full <- vector("list", p_expected)
  for (j in seq_len(p_expected)) {
    l_owner <- owner_of_feature(j, numworkers)
    X_full[[j]] <- wrk[[l_owner]]$predictorLst[[j]]
    if (is.null(X_full[[j]]))
      stop(sprintf("[build_global_dataset_vfl] Feature %d missing at owner %d", j, l_owner))
  }
  list(
    Xlist_train = lapply(X_full, subset_fd, idx = train_idx),
    Xlist_test  = lapply(X_full, subset_fd, idx = test_idx),
    Y_train     = subset_fd(Y_full,        train_idx),
    Y_test      = subset_fd(Y_full,        test_idx)
  )
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
  
  My <- inprod(ytrue_fd$basis, ytrue_fd$basis)
  C_hat  <- yhat_fd$coefs
  C_true <- ytrue_fd$coefs
  C_diff <- C_hat - C_true
  il2  <- sum(colSums(C_diff * (My %*% C_diff))) / ncol(C_diff)
  denom <- sum(colSums(C_true * (My %*% C_true))) / ncol(C_true)
  ril2 <- il2 / (denom + eps)
  
  list(wmape = wmape, smape = smape, nrmse = nrmse, mape = mape, rmse = rmse, il2 = il2, ril2 = ril2)
}

comm_cost_mb <- function(Ntrain, Qx, p) {
  bytes <- p * (Ntrain * Qx + Qx * Qx) * 8
  bytes / (1024^2)
}

adapt_Sx <- function(Xlist, mode = c("fixed", "empirical"), Sx_fixed = 3.0) {
  mode <- match.arg(mode)
  if (mode == "fixed") return(rep(Sx_fixed, length(Xlist)))
  Sx_vec <- numeric(length(Xlist))
  for (j in seq_along(Xlist)) {
    fdj <- Xlist[[j]]
    Mx <- inprod(fdj$basis, fdj$basis)
    C  <- fdj$coefs
    norms <- sqrt(colSums(C * (Mx %*% C)))
    Sx_vec[j] <- as.numeric(quantile(norms, 0.95, na.rm = TRUE))
    if (!is.finite(Sx_vec[j]) || Sx_vec[j] <= 0) Sx_vec[j] <- 3.0
  }
  Sx_vec
}

# zCDP accounting: Δ = 2Sx (replace-one), ρ = Δ^2/(2σ^2); compose across features
eps_from_sx_zcdp <- function(Sx_vec, sx, delta, folds_compose = 1L) {
  if (sx <= 0) return(Inf)
  rho_j     <- 2 * (Sx_vec^2) / (sx^2)
  rho_total <- sum(rho_j) * folds_compose
  rho_total + 2 * sqrt(rho_total * log(1 / delta))
}

# ---------------------- Job table ----------------------------
set <- 1:N_global
fold_size <- N_global / folds_per_worker
stopifnot(fold_size == floor(fold_size))

tasks <- expand.grid(
  sx         = sx_grid,
  hh         = seq_len(num_duplicate),
  numworkers = numworkersseq,
  fold       = seq_len(folds_per_worker),
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
)
tasks$nw_i <- match(tasks$numworkers, numworkersseq)

# ---------------------- Parallel plan -----------------------
n_cores <- parallel::detectCores(logical = TRUE)
future::plan(multisession, workers = max(1, min(n_cores - 1, nrow(tasks))))

# ---------------------- Run sweep in parallel ---------------
job_rows <- future_lapply(
  X = seq_len(nrow(tasks)),
  FUN = function(i_job) {
    # Keep each job self-contained
    library(fda)
    source("functions.R")
    setwd(root_dir)
    
    sx         <- tasks$sx[i_job]
    hh         <- tasks$hh[i_job]
    numworkers <- tasks$numworkers[i_job]
    fold       <- tasks$fold[i_job]
    
    set.seed(1e6 * hh + 1e3 * numworkers + 10L * fold + round(1e3 * sx))
    
    # Train/test indices (global; same individuals across workers)
    test_idx  <- set[((fold - 1) * fold_size + 1):(fold * fold_size)]
    train_idx <- setdiff(set, test_idx)
    
    # Build dataset
    ds <- build_global_dataset_vfl(numworkers, hh, train_idx, test_idx)
    X_train <- ds$Xlist_train; X_test <- ds$Xlist_test
    Y_train <- ds$Y_train;     Y_test <- ds$Y_test
    
    # DP params per feature and epsilon per job
    Sx_vec <- adapt_Sx(X_train, mode = Sx_mode, Sx_fixed = Sx_fixed)
    sx_vec <- rep(sx, p)
    folds_compose <- if (isTRUE(compose_over_cv)) 2L else 1L
    eps_job <- eps_from_sx_zcdp(Sx_vec, sx, delta_total, folds_compose = folds_compose)
    
    # Fit
    t0 <- Sys.time()
    fit <- vfl_dp_foboost(
      xfd_list = X_train,
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
    yhat_test <- predict_vfl_dp_foboost(fit, X_test)
    mets <- metrics_fd(yhat_test, Y_test, grid = tgrid)
    
    # Communication cost for this fold
    comm_mb <- comm_cost_mb(Ntrain = length(train_idx), Qx = t_basis, p = p)
    
    # Return one-row data.frame
    data.frame(
      sx = sx,
      numworkers = numworkers,
      hh = hh,
      fold = fold,
      WMAPE = mets$wmape,
      SMAPE = mets$smape,
      NRMSE = mets$nrmse,
      MAPE  = mets$mape,
      RMSE  = mets$rmse,
      IL2   = mets$il2,
      RIL2  = mets$ril2,
      Eps_job = eps_job,
      Time_sec = time_sec,
      Comm_MB = comm_mb,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  },
  future.seed = TRUE
)

# Keep only valid rows (defensive)
valid_idx <- vapply(job_rows, function(x) is.data.frame(x) && nrow(x) == 1, logical(1))
if (!any(valid_idx)) {
  cat("\n[WARN] No successful jobs. Check data files or run with future::plan(sequential) to debug.\n")
  sweep_df <- data.frame()  # empty
} else {
  sweep_df <- do.call(rbind, job_rows[valid_idx])
}

# ---------------------- Summaries (guarded) -----------------
if (nrow(sweep_df) == 0) {
  # Write empty shells; avoid assigning single scalars to 0-row frames
  write.csv(sweep_df, file = "privacy_sweep_perjob.csv", row.names = FALSE)
  cat("\n[INFO] Wrote empty per-job CSV. No summary due to 0 rows.\n")
} else {
  # Mean and sd by (workers, sx) – includes epsilon
  agg_mean <- aggregate(
    cbind(WMAPE, SMAPE, NRMSE, MAPE, RMSE, IL2, RIL2, Time_sec, Comm_MB, Eps_job) ~
      numworkers + sx,
    data = sweep_df,
    FUN = function(x) mean(x, na.rm = TRUE)
  )
  agg_sd <- aggregate(
    cbind(WMAPE, SMAPE, NRMSE, MAPE, RMSE, IL2, RIL2, Time_sec, Comm_MB, Eps_job) ~
      numworkers + sx,
    data = sweep_df,
    FUN = function(x) sd(x, na.rm = TRUE)
  )
  
  # Friendly column names
  names(agg_mean)[-(1:2)] <- paste0(names(agg_mean)[-(1:2)], "_mean")
  names(agg_sd)[-(1:2)]   <- paste0(names(agg_sd)[-(1:2)],   "_sd")
  
  # Join mean & sd
  summary_df <- merge(agg_mean, agg_sd, by = c("numworkers", "sx"), all = TRUE)
  
  # Save results
  write.csv(sweep_df,  file = "privacy_sweep_perjob.csv",    row.names = FALSE)
  write.csv(summary_df, file = "privacy_sweep_summary.csv",  row.names = FALSE)
  save(sweep_df, summary_df, file = "privacy_sweep_results.RData")
  
  cat("\n=== Accuracy vs Privacy Budget (means ± sd) ===\n")
  print(summary_df[order(summary_df$numworkers, summary_df$sx), ])
}

# Optional: reset to sequential to avoid leaving a multisession pool
future::plan(sequential)

# ============================================================
# Black-box Membership Inference Attack (MIA) for DP VFL boosting
# - VFL semantics: same individuals across workers; each worker owns
#   a disjoint subset of predictors; 'yfdobj' identical across workers.
# - Train on an 80% "member" split; evaluate attack on members vs non-members.
# - Attacks:
#     LA   = Loss-threshold attack (functional L2 residual)
#     CONF = Confidence/fit score (functional norm of fitted output)
# - Sweeps DP noise levels sx and reports accuracy vs privacy budget ε.
# ============================================================

root_dir <- "/Users/yuding/Dropbox/VFL_code"
setwd(root_dir)

suppressPackageStartupMessages({
  library(fda)
  library(future.apply)
})

source("functions.R")  # vfl_dp_foboost(), predict_vfl_dp_foboost()

# ---------------------- Settings ----------------------------
numworkersseq     <- c(2, 4, 6, 8, 10)
num_duplicate     <- 10
N_global          <- 500     # must match generator_vfl.R
p                 <- 20
rangeval          <- c(0, 100)
t_basis           <- 20
basisobj          <- create.bspline.basis(rangeval, nbasis = t_basis)

# Privacy sweep
sx_grid           <- c(0.0, 0.1, 0.2, 0.3)  # per-feature Gaussian std (whitened)
Sx_mode           <- "fixed"                 # "fixed" gives DP-correct ε; "empirical" is diagnostic
Sx_fixed          <- 3.0                     # per-feature clipping radius (whitened norm)
delta_total       <- 1e-5
compose_over_cv   <- FALSE                   # boosting uses the same DP release -> no extra composition

# Boosting/penalties
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

# Train/test split for membership
train_frac <- 0.8

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
  
  # basic checks (owned features must be fd; non-owned can be NULL)
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

build_global_dataset_vfl <- function(numworkers, hh, idx_keep) {
  p_expected <- p
  wrk <- lapply(seq_len(numworkers), function(l) {
    load_worker_data_nullable(l, numworkers, hh, p_expected)
  })
  Y_full <- wrk[[1]]$yfdobj
  Xlist_full <- vector("list", p_expected)
  for (j in seq_len(p_expected)) {
    l_owner <- owner_of_feature(j, numworkers)
    Xlist_full[[j]] <- wrk[[l_owner]]$predictorLst[[j]]
    if (is.null(Xlist_full[[j]]))
      stop(sprintf("[build_global_dataset_vfl] Feature %d missing at owner %d", j, l_owner))
  }
  # Subset to idx_keep
  Xlist <- lapply(Xlist_full, subset_fd, idx = idx_keep)
  Y     <- subset_fd(Y_full,        idx = idx_keep)
  list(Xlist = Xlist, Y = Y)
}

# Functional L2 metrics from coefficients (Gram My)
residual_L2_per_sample <- function(y_fd, yhat_fd) {
  stopifnot(is.fd(y_fd), is.fd(yhat_fd))
  My <- inprod(y_fd$basis, y_fd$basis)
  C_true <- y_fd$coefs
  C_hat  <- yhat_fd$coefs
  C_diff <- C_true - C_hat               # Qy x N
  # quadratic form per sample: diag(C_diff^T My C_diff)
  colSums(C_diff * (My %*% C_diff))
}

fit_norm_L2_per_sample <- function(yhat_fd) {
  My <- inprod(yhat_fd$basis, yhat_fd$basis)
  C_hat <- yhat_fd$coefs
  colSums(C_hat * (My %*% C_hat))
}

# ROC/AUC helpers
roc_from_scores <- function(scores, labels) {
  # labels: 1=member, 0=non-member; higher score => more likely member
  ord <- order(scores, decreasing = TRUE)
  s <- scores[ord]; y <- labels[ord]
  P <- sum(y == 1); N <- sum(y == 0)
  if (P == 0 || N == 0) return(list(fpr = c(0,1), tpr = c(0,1)))
  tp <- cumsum(y == 1); fp <- cumsum(y == 0)
  tpr <- tp / P; fpr <- fp / N
  # Prepend (0,0)
  fpr <- c(0, fpr); tpr <- c(0, tpr)
  list(fpr = fpr, tpr = tpr)
}

auc_trapz <- function(fpr, tpr) {
  # Trapezoid AUC; assumes fpr sorted ascending
  sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1)) / 2)
}

attack_advantage <- function(fpr, tpr) {
  max(tpr - fpr, na.rm = TRUE)
}

tpr_at_fpr <- function(fpr, tpr, target = 0.1) {
  if (all(is.na(fpr)) || all(is.na(tpr))) return(NA_real_)
  # find first fpr >= target
  idx <- which(fpr >= target)
  if (length(idx) == 0) return(tail(tpr, 1))
  tpr[idx[1]]
}

# Privacy accounting: zCDP -> (ε, δ)
eps_from_sx_zcdp <- function(Sx_vec, sx, delta, folds_compose = 1L) {
  if (sx <= 0) return(Inf)
  rho_j     <- 2 * (Sx_vec^2) / (sx^2)   # Δ=2Sx, ρ=Δ^2/(2σ^2)
  rho_total <- sum(rho_j) * folds_compose
  eps <- rho_total + 2 * sqrt(rho_total * log(1 / delta))
  eps
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

# ---------------------- Tasks -------------------------------
tasks <- expand.grid(
  sx         = sx_grid,
  hh         = seq_len(num_duplicate),
  numworkers = numworkersseq,
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
)

n_cores <- parallel::detectCores(logical = TRUE)
future::plan(multisession, workers = max(1, min(n_cores - 1, nrow(tasks))))

# ---------------------- Run MIA in parallel -----------------
attack_list <- future_lapply(
  X = seq_len(nrow(tasks)),
  FUN = function(i_job) {
    library(fda)
    source("functions.R")
    load("truth_active_idx.RData")  # active_idx (not strictly needed for MIA)
    setwd(root_dir)
    
    sx         <- tasks$sx[i_job]
    hh         <- tasks$hh[i_job]
    numworkers <- tasks$numworkers[i_job]
    
    set.seed(1e6 * hh + 1e3 * numworkers + round(1e3 * sx))
    
    # Membership split: 80% train (members), 20% test (non-members)
    all_idx <- 1:N_global
    n_train <- floor(train_frac * N_global)
    train_idx <- sort(sample(all_idx, n_train, replace = FALSE))
    test_idx  <- setdiff(all_idx, train_idx)
    
    # Build training and test datasets (VFL semantics)
    # Training set
    ds_train <- build_global_dataset_vfl(numworkers, hh, train_idx)
    X_train <- ds_train$Xlist
    Y_train <- ds_train$Y
    # Test set (for scoring)
    ds_test <- build_global_dataset_vfl(numworkers, hh, test_idx)
    X_test <- ds_test$Xlist
    Y_test <- ds_test$Y
    
    # DP params per feature
    Sx_vec_train <- adapt_Sx(X_train, mode = Sx_mode, Sx_fixed = Sx_fixed)
    sx_vec_train <- rep(sx, p)
    
    # Privacy budget for this training release
    folds_compose <- if (isTRUE(compose_over_cv)) 2L else 1L
    eps_train <- eps_from_sx_zcdp(Sx_vec_train, sx, delta_total, folds_compose = folds_compose)
    
    # Fit on training (members)
    t0 <- Sys.time()
    fit <- vfl_dp_foboost(
      xfd_list = X_train,
      yfd      = Y_train,
      Sx_vec   = Sx_vec_train,
      sx_vec   = sx_vec_train,
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
    
    # Predict on both train (members) and test (non-members)
    yhat_train <- predict_vfl_dp_foboost(fit, X_train)
    yhat_test  <- predict_vfl_dp_foboost(fit, X_test)
    
    # Attack scores
    # LA: higher score => more likely member (negative residual L2)
    r2_train <- residual_L2_per_sample(Y_train, yhat_train)  # length n_train
    r2_test  <- residual_L2_per_sample(Y_test,  yhat_test)   # length n_test
    score_LA_train <- -r2_train
    score_LA_test  <- -r2_test
    
    # CONF: confidence/fit magnitude, higher => more confident -> more likely member
    fit2_train <- fit_norm_L2_per_sample(yhat_train)
    fit2_test  <- fit_norm_L2_per_sample(yhat_test)
    score_CONF_train <- fit2_train
    score_CONF_test  <- fit2_test
    
    # Labels: 1=member, 0=non-member
    scores_LA   <- c(score_LA_train, score_LA_test)
    labels_LA   <- c(rep(1L, length(score_LA_train)), rep(0L, length(score_LA_test)))
    scores_CONF <- c(score_CONF_train, score_CONF_test)
    labels_CONF <- labels_LA
    
    # ROC/AUC etc.
    roc_LA   <- roc_from_scores(scores_LA, labels_LA)
    roc_CONF <- roc_from_scores(scores_CONF, labels_CONF)
    AUC_LA   <- auc_trapz(roc_LA$fpr,   roc_LA$tpr)
    ADV_LA   <- attack_advantage(roc_LA$fpr,   roc_LA$tpr)
    TPR10_LA <- tpr_at_fpr(roc_LA$fpr,  roc_LA$tpr, target = 0.1)
    AUC_CF   <- auc_trapz(roc_CONF$fpr, roc_CONF$tpr)
    ADV_CF   <- attack_advantage(roc_CONF$fpr, roc_CONF$tpr)
    TPR10_CF <- tpr_at_fpr(roc_CONF$fpr, roc_CONF$tpr, target = 0.1)
    
    time_sec <- as.numeric(difftime(t1, t0, units = "secs"))
    
    data.frame(
      numworkers = numworkers,
      workers    = numworkers,   # convenience alias for grouping
      hh         = hh,
      sx         = sx,
      AUC_LA     = AUC_LA,
      ADV_LA     = ADV_LA,
      TPR10_LA   = TPR10_LA,
      AUC_CONF   = AUC_CF,
      ADV_CONF   = ADV_CF,
      TPR10_CONF = TPR10_CF,
      Eps_train  = eps_train,
      Time_sec   = time_sec,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  },
  future.seed = TRUE
)

# ---------------------- Assemble & Summarize -----------------
attack_df <- do.call(rbind, lapply(attack_list, function(x) {
  # keep rows that are proper data.frames; drop NULL/NA safely
  if (is.data.frame(x) && nrow(x) > 0) x else NULL
}))

if (is.null(attack_df) || nrow(attack_df) == 0) {
  cat("\n[WARN] No attack rows produced; check upstream errors in futures.\n",
      "Tip: run with future::plan(sequential) to see first error.\n")
} else {
  # Means by (workers, sx). Use numworkers+sx (and provide workers alias).
  # (Guard: aggregate() fails on zero-row input; we already checked.)
  attack_summary <- aggregate(
    cbind(AUC_LA, ADV_LA, TPR10_LA, AUC_CONF, ADV_CONF, TPR10_CONF, Eps_train, Time_sec) ~
      numworkers + sx,
    data = attack_df,
    FUN = function(x) mean(x, na.rm = TRUE)
  )
  
  attack_summary_sd <- aggregate(
    cbind(AUC_LA, ADV_LA, TPR10_LA, AUC_CONF, ADV_CONF, TPR10_CONF, Eps_train, Time_sec) ~
      numworkers + sx,
    data = attack_df,
    FUN = function(x) sd(x, na.rm = TRUE)
  )
  
  cat("\n=== Black-box MIA (means) by workers × sx ===\n")
  print(attack_summary)
  
  # Save
  write.csv(attack_df,       file = "mia_blackbox_perjob.csv", row.names = FALSE)
  write.csv(attack_summary,  file = "mia_blackbox_summary_mean.csv", row.names = FALSE)
  write.csv(attack_summary_sd, file = "mia_blackbox_summary_sd.csv", row.names = FALSE)
  
  save(attack_df, attack_summary, attack_summary_sd, file = "mia_blackbox_results.RData")
}

# Optional: reset plan
future::plan(sequential)

# ======= case_study_s2dp_fgb.R  =======
suppressPackageStartupMessages({
  library(fda)
  library(Matrix)         # for Diagonal()
  library(future.apply)
})

# Paper-aligned S2DP implementation (must have):
#  - dp_release_whitened(), form_dp_moments()
#  - predict_vfl_dp_foboost() that whitens new X (Mx^{-1/2})
#  - stabilized solve_penalized_fof()
source("functions.R")

# -------------------- load split arrays --------------------
load("predictors_passive.RData")  # -> X_passive_arr : T x N x P_passive
load("predictors_active.RData")   # -> X_active_arr  : T x N x P_active
load("response.RData")            # -> Y_arr         : T x N

stopifnot(exists("X_passive_arr"), exists("X_active_arr"), exists("Y_arr"))
T_obs  <- dim(Y_arr)[1]
N_pat  <- dim(Y_arr)[2]
P_pass <- if (length(dim(X_passive_arr)) == 3) dim(X_passive_arr)[3] else 0L
P_act  <- if (length(dim(X_active_arr))  == 3) dim(X_active_arr)[3]  else 0L
cat(sprintf("[INFO] Y: %d x %d | Passive: P=%d | Active: P=%d\n", T_obs, N_pat, P_pass, P_act))

# -------------------- passive feature names (align with X_passive_arr order) --------------------
# If produced by preprocessing, this file lists passive_keep in order.
passive_all_names <- if (file.exists("passive_kept_names.txt")) {
  readLines("passive_kept_names.txt")
} else {
  if (P_pass > 0) rep(NA_character_, P_pass) else character(0)
}

# -------------------- settings --------------------
folds        <- 5
eps_grid     <- c(Inf, 100, 80, 60, 40, 20, 10)  # total budget grid (global ε); Inf = no DP
delta_total  <- 1e-5

rangeval <- c(0, 100)
Q_basis  <- 10                          # use Q=10
tgrid    <- seq(rangeval[1], rangeval[2], length.out = T_obs)
basisobj <- create.bspline.basis(rangeval, nbasis = Q_basis)

lambda_s <- 1e-1; lambda_t <- 1e-1; lambda_st <- 0
nu_hi <- 0.25; nu_mid <- 0.20; nu_lo <- 0.15
max_steps <- 30; patience <- 6; min_steps <- 8

# Passive-only DP screening
TOPK <- if (P_pass > 0) min(20L, floor(P_pass/3)) else 0L
SCREEN_Q <- 0.70
CLIP_CAP <- c(0.5, 1.5)

# -------------------- helpers --------------------
zscore_over_time <- function(M_TxN) {
  m <- colMeans(M_TxN); s <- apply(M_TxN, 2, sd); s[s < 1e-8] <- 1
  sweep(sweep(M_TxN, 2, m, "-"), 2, s, "/")
}
project_to_basis <- function(mat_NxT, basisobj, tgrid) {
  Phi <- eval.basis(tgrid, basisobj)
  Xt  <- t(mat_NxT)
  G   <- crossprod(Phi) + 1e-10 * diag(ncol(Phi))
  A   <- crossprod(Phi, Xt)
  solve(G, A)
}
make_fd_from_array <- function(M_TxN, basisobj, tgrid, value_name) {
  C_QxN <- project_to_basis(t(M_TxN), basisobj, tgrid)
  fd(coef = C_QxN, basisobj = basisobj,
     fdnames = list("arg"="t","rep"=paste0("id", seq_len(ncol(M_TxN))), "values"=value_name))
}
metrics_fd <- function(yhat_fd, ytrue_fd, grid) {
  Yhat <- eval.fd(grid, yhat_fd); Ytru <- eval.fd(grid, ytrue_fd)
  eps <- 1e-8; err <- Yhat - Ytru
  rmse <- sqrt(mean(err^2)); rmse_subj <- sqrt(colMeans(err^2))
  denom_sd <- sd(as.numeric(Ytru)) + eps
  nrmse <- rmse / denom_sd; nrmse_worst <- max(rmse_subj) / denom_sd
  smape <- mean(2*abs(err)/(abs(Yhat)+abs(Ytru)+eps))*100
  wmape <- sum(abs(err))/(sum(abs(Ytru))+eps)*100
  mape  <- mean(abs(err)/(abs(Ytru)+eps))*100
  My <- inprod(ytrue_fd$basis, ytrue_fd$basis)
  C_hat <- yhat_fd$coefs; C_true <- ytrue_fd$coefs; C_diff <- C_hat - C_true
  il2 <- sum(colSums(C_diff * (My %*% C_diff))) / ncol(C_diff)
  denom <- sum(colSums(C_true * (My %*% C_true))) / ncol(C_true)
  ril2 <- il2 / (denom + eps)
  list(wmape=wmape, smape=smape, nrmse=nrmse, nrmse_worst=nrmse_worst,
       mape=mape, rmse=rmse, il2=il2, ril2=ril2)
}
sym_eigen_sqrt_local <- function(M, ridge=1e-8){
  ee <- eigen((M+t(M))/2, symmetric=TRUE)
  lam <- pmax(ee$values, ridge); U <- ee$vectors
  list(half = U %>% diag(sqrt(lam), length(lam)) %*% t(U),
       invhalf = U %*% diag(1/sqrt(lam), length(lam)) %*% t(U))
}
sym_eigen_sqrt_local <- function(M, ridge=1e-8){ # (safe, no pipe)
  ee <- eigen((M+t(M))/2, symmetric=TRUE)
  lam <- pmax(ee$values, ridge); U <- ee$vectors
  list(half = U %*% diag(sqrt(lam), length(lam)) %*% t(U),
       invhalf = U %*% diag(1/sqrt(lam), length(lam)) %*% t(U))
}
adapt_Sx_empirical_capped <- function(Xlist, q = SCREEN_Q, cap_low = CLIP_CAP[1], cap_high = CLIP_CAP[2]) {
  sapply(Xlist, function(fdj) {
    Mx  <- inprod(fdj$basis, fdj$basis)
    Z   <- sym_eigen_sqrt_local(Mx)$invhalf %*% fdj$coefs
    norms <- sqrt(colSums(Z * Z))
    sx <- as.numeric(quantile(norms, q, na.rm = TRUE))
    sx <- max(cap_low, min(cap_high, sx))
    if (!is.finite(sx)) 1.0 else sx
  })
}
comm_cost_mb <- function(Ntrain, Qx, p) (p * (Ntrain * Qx) * 8) / (1024^2)

# ε ↔ ρ
if (!exists("eps_from_rho", mode="function")) {
  eps_from_rho <- function(rho, delta) rho + 2 * sqrt(pmax(rho, 0) * log(1/delta))
}
if (!exists("rho_from_eps", mode="function")) {
  rho_from_eps <- function(eps, delta) uniroot(function(r) eps_from_rho(r,delta)-eps, c(0, 1e8))$root
}

# -------------------- preprocess -> fd lists --------------------
Y_z <- zscore_over_time(Y_arr)
yfdobj <- make_fd_from_array(Y_z, basisobj, tgrid, value_name="Y")

X_pass_list <- list()
if (P_pass > 0) {
  Xp_z <- X_passive_arr
  for (j in seq_len(P_pass)) Xp_z[,,j] <- zscore_over_time(X_passive_arr[,,j])
  for (j in seq_len(P_pass)) X_pass_list[[j]] <- make_fd_from_array(Xp_z[,,j], basisobj, tgrid, value_name=paste0("Xp", j))
}
X_act_list <- list()
if (P_act > 0) {
  Xa_z <- X_active_arr
  for (j in seq_len(P_act)) Xa_z[,,j] <- zscore_over_time(X_active_arr[,,j])
  for (j in seq_len(P_act)) X_act_list[[j]] <- make_fd_from_array(Xa_z[,,j], basisobj, tgrid, value_name=paste0("Xa", j))
}
cat(sprintf("[INFO] fd built: Y %d x %d; P_pass=%d, P_act=%d, Q=%d\n",
            nrow(yfdobj$coefs), ncol(yfdobj$coefs), length(X_pass_list), length(X_act_list), Q_basis))

# -------------------- RANDOM CV splits --------------------
set.seed(20251011)  # reproducible random partition
perm <- sample.int(N_pat)                         # shuffle subjects
fold_assign <- rep(1:folds, length.out = N_pat)  # balanced fold labels
fold_assign <- sample(fold_assign)                # random ordering
fold_splits <- lapply(1:folds, function(f) {
  test_idx  <- which(fold_assign == f)
  train_idx <- setdiff(seq_len(N_pat), test_idx)
  list(train = train_idx, test = test_idx)
})
# sanity: print fold sizes
cat("[INFO] Fold sizes (test): ", paste(sapply(fold_splits, function(l) length(l$test)), collapse=", "), "\n")

# -------------------- Passive-only DP screening --------------------
dp_screen_topk_passive <- function(Xp_train, Y_train, Sx_vec_pass, eps_t, delta_total, K) {
  if (length(Xp_train) == 0 || K <= 0) return(integer(0))
  Ycoef <- t(Y_train$coefs); Yc <- scale(Ycoef, center=TRUE, scale=FALSE)
  scores <- numeric(length(Xp_train))
  if (is.finite(eps_t) && eps_t > 0) {
    rho_target <- rho_from_eps(eps_t, delta_total)
    sx <- sqrt( sum(Sx_vec_pass^2) / (2 * pmax(rho_target, 1e-12)) )
    for (j in seq_along(Xp_train)) {
      Mx  <- inprod(Xp_train[[j]]$basis, Xp_train[[j]]$basis)
      Cx  <- Xp_train[[j]]$coefs
      rel <- dp_release_whitened(C = Cx, M = Mx, S = Sx_vec_pass[j], s = sx)
      ZdpN <- t(rel$Zdp)                         # N x Qx
      Zc   <- scale(ZdpN, center=TRUE, scale=FALSE)
      Gxy  <- crossprod(Zc, Yc) / nrow(Zc)
      scores[j] <- sqrt(sum(Gxy*Gxy))
    }
  } else {
    for (j in seq_along(Xp_train)) {
      Mx <- inprod(Xp_train[[j]]$basis, Xp_train[[j]]$basis)
      Z  <- t(sym_eigen_sqrt_local(Mx)$invhalf %*% Xp_train[[j]]$coefs)
      G  <- crossprod(Z, Yc) / nrow(Z)
      scores[j] <- sqrt(sum(G*G))
    }
  }
  head(order(scores, decreasing=TRUE), K)
}

# -------------------- Run ε-matched S2DP (passive DP only) --------------------
tasks <- expand.grid(eps_target = eps_grid, fold = seq_len(folds), KEEP.OUT.ATTRS = FALSE)
n_cores <- parallel::detectCores(TRUE)
future::plan(multisession, workers = max(1, min(n_cores - 1, nrow(tasks))))

job_rows <- future_lapply(seq_len(nrow(tasks)), function(i_job) {
  library(fda); library(Matrix); source("functions.R")
  eps_t <- tasks$eps_target[i_job]; fold <- tasks$fold[i_job]
  
  subset_fd <- function(fdobj, idx) fd(coef = fdobj$coefs[, idx, drop=FALSE],
                                       basisobj = fdobj$basis, fdnames = fdobj$fdnames)
  tr <- fold_splits[[fold]]$train; te <- fold_splits[[fold]]$test
  
  # Build TRAIN/TEST lists
  Xp_tr <- lapply(X_pass_list, subset_fd, idx = tr)
  Xp_te <- lapply(X_pass_list, subset_fd, idx = te)
  Xa_tr <- lapply(X_act_list,  subset_fd, idx = tr)
  Xa_te <- lapply(X_act_list,  subset_fd, idx = te)
  Y_tr  <- subset_fd(yfdobj, tr)
  Y_te  <- subset_fd(yfdobj, te)
  
  # Passive empirical clipping (TRAIN)
  Sx_pass <- if (length(Xp_tr)) adapt_Sx_empirical_capped(Xp_tr, q = SCREEN_Q, cap_low = CLIP_CAP[1], cap_high = CLIP_CAP[2]) else numeric(0)
  
  # Passive TOPK screening; active always kept
  keep_pass_idx <- if (length(Xp_tr) && TOPK > 0)
    dp_screen_topk_passive(Xp_tr, Y_tr, Sx_pass, eps_t, delta_total, TOPK)
  else seq_along(Xp_tr)
  
  Xp_tr_kept   <- if (length(keep_pass_idx)) Xp_tr[keep_pass_idx] else list()
  Xp_te_kept   <- if (length(keep_pass_idx)) Xp_te[keep_pass_idx] else list()
  Sx_pass_kept <- if (length(keep_pass_idx)) Sx_pass[keep_pass_idx] else numeric(0)
  
  # IDs/names of passive features kept for THIS job (align with passive_all_names / X_passive_arr)
  p_pass_used_ids   <- if (length(keep_pass_idx)) paste(keep_pass_idx, collapse = ";") else ""
  p_pass_used_names <- if (length(keep_pass_idx)) {
    # guard length mismatch
    names_use <- passive_all_names
    if (length(names_use) < max(keep_pass_idx)) names_use <- c(names_use, rep(NA_character_, max(keep_pass_idx) - length(names_use)))
    paste(names_use[keep_pass_idx], collapse = ";")
  } else ""
  
  # Calibrate sx for ε on passive only
  if (is.finite(eps_t) && eps_t > 0 && length(Sx_pass_kept)) {
    rho_target <- rho_from_eps(eps_t, delta_total)
    sx <- sqrt( sum(Sx_pass_kept^2) / (2 * pmax(rho_target, 1e-12)) )
  } else {
    sx <- 0
  }
  
  # Assemble combined X: [passive_kept | active_all]
  X_tr_all <- c(Xp_tr_kept, Xa_tr)
  X_te_all <- c(Xp_te_kept, Xa_te)
  
  P_pass_kept <- length(Xp_tr_kept)
  P_act_all   <- length(Xa_tr)
  P_used_all  <- P_pass_kept + P_act_all
  
  # DP vectors aligned with X_tr_all:
  #  - Passive: (Sx_pass_kept, s = sx)
  #  - Active : huge S (no clip), s = 0 (no DP)
  Sx_vec <- c(if (P_pass_kept) Sx_pass_kept else numeric(0),
              if (P_act_all)   rep(1e9, P_act_all) else numeric(0))
  sx_vec <- c(if (P_pass_kept) rep(sx, P_pass_kept) else numeric(0),
              if (P_act_all)   rep(0,  P_act_all)   else numeric(0))
  
  # Shrinkage schedule
  nu_eff <- if (!is.finite(eps_t)) nu_hi else if (eps_t <= 20) nu_lo else if (eps_t <= 60) nu_mid else nu_hi
  
  t0 <- Sys.time()
  fit <- vfl_dp_foboost(
    xfd_list = X_tr_all, yfd = Y_tr,
    Sx_vec = Sx_vec, sx_vec = sx_vec,
    Omega_x_list = NULL, Omega_y = NULL,
    lambda_s = lambda_s, lambda_t = lambda_t, lambda_st = lambda_st,
    nu = nu_eff, max_steps = max_steps, crossfit = TRUE,
    stop_mode = "aic_train_dp", min_steps = min_steps,
    aic = "spherical", aic_c = TRUE, df_K = 5, patience = patience
  )
  time_sec <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  
  # Predict & metrics
  yhat_te <- predict_vfl_dp_foboost(fit, X_te_all)
  mets <- metrics_fd(yhat_te, Y_te, grid = tgrid)
  
  # Communication: only passive kept parties
  comm_mb <- comm_cost_mb(Ntrain = length(tr), Qx = Q_basis, p = P_pass_kept)
  
  data.frame(
    eps_target = eps_t, fold = fold,
    p_pass_used = P_pass_kept, p_act_used = P_act_all,
    p_pass_used_ids = p_pass_used_ids,                 # NEW: indices of passive features used (screened-in)
    p_pass_used_names = p_pass_used_names,             # NEW: names of passive features used
    WMAPE = mets$wmape, SMAPE = mets$smape,
    NRMSE = mets$nrmse, NRMSE_worst = mets$nrmse_worst,
    MAPE  = mets$mape,  RMSE = mets$rmse,
    IL2   = mets$il2,   RIL2 = mets$ril2,
    Time_sec = time_sec, Comm_MB = comm_mb,
    stringsAsFactors = FALSE, check.names = FALSE
  )
}, future.seed = TRUE)

# -------------------- Summaries --------------------
ok <- vapply(job_rows, function(x) is.data.frame(x) && nrow(x) == 1, logical(1))
sweep_df <- if (any(ok)) do.call(rbind, job_rows[ok]) else data.frame()

out_perjob   <- "case_arrays_privacy_sweep_perjob.csv"
out_summary  <- "case_arrays_privacy_sweep_summary.csv"

if (!nrow(sweep_df)) {
  write.csv(sweep_df, out_perjob, row.names = FALSE)
  cat("\n[INFO] Wrote empty per-job CSV (no rows). No summary.\n")
} else {
  base_cols <- c("p_pass_used","p_act_used","WMAPE","SMAPE","NRMSE","NRMSE_worst","MAPE","RMSE","IL2","RIL2",
                 "Time_sec","Comm_MB")
  agg_mean <- aggregate(sweep_df[, base_cols],
                        by = list(eps_target = sweep_df$eps_target),
                        FUN = function(x) mean(x, na.rm = TRUE))
  agg_sd   <- aggregate(sweep_df[, base_cols],
                        by = list(eps_target = sweep_df$eps_target),
                        FUN = function(x) sd(x, na.rm = TRUE))
  names(agg_mean)[-1] <- paste0(names(agg_mean)[-1], "_mean")
  names(agg_sd)[-1]   <- paste0(names(agg_sd)[-1],   "_sd")
  summary_df <- merge(agg_mean, agg_sd, by = "eps_target", all = TRUE)
  
  write.csv(sweep_df,  out_perjob,  row.names = FALSE)   # includes *ids* and *names*
  write.csv(summary_df, out_summary, row.names = FALSE)
  
  cat("\n=== CASE ARRAYS — (Passive DP, Active non-DP) Accuracy vs Privacy ===\n")
  print(summary_df[order(summary_df$eps_target, decreasing = TRUE), ], row.names = FALSE)
}

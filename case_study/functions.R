# =============================== functions.R ===============================
suppressPackageStartupMessages({ library(fda); library(Matrix) })

# ---- helpers --------------------------------------------------------------
.symmetrize <- function(M) (M + t(M))/2
fro_sq <- function(A) sum(A * A)
`%||%` <- function(a,b) if(!is.null(a)) a else b

sym_eigen_sqrt <- function(M, ridge = 1e-8) {
  M <- .symmetrize(M)
  ee <- eigen(M, symmetric = TRUE)
  lam <- pmax(ee$values, ridge); U <- ee$vectors
  list(
    half    = U %*% diag(sqrt(lam),  length(lam)) %*% t(U),
    invhalf = U %*% diag(1/sqrt(lam),length(lam)) %*% t(U)
  )
}
.psd_project <- function(M, floor = 1e-6) {
  E <- eigen(.symmetrize(M), symmetric = TRUE)
  lam <- pmax(E$values, floor)
  E$vectors %*% diag(lam, length(lam)) %*% t(E$vectors)
}
center_rows <- function(X){
  mu <- colMeans(X); list(Xc = sweep(X, 2, mu, "-"), mean = mu)
}
penalty_matrix <- function(basis, Lfdobj = int2Lfd(2)) eval.penalty(basis, Lfdobj)

# ---- zCDP accounting ------------------------------------------------------
eps_from_rho <- function(rho, delta) rho + 2 * sqrt(pmax(rho,0) * log(1/delta))
rho_from_eps <- function(eps_target, delta){
  if (!is.finite(eps_target)) return(0)
  f <- function(rho) eps_from_rho(rho, delta) - eps_target
  lo <- 0; hi <- max(1, eps_target^2)
  while (f(hi) < 0) hi <- hi*2
  uniroot(f, c(lo,hi))$root
}
eps_from_sx_zcdp_vec <- function(Sx_vec, sx, delta){
  rho_j <- 2 * (Sx_vec^2) / (sx^2)
  list(rho_j = rho_j, eps_j = eps_from_rho(rho_j, delta),
       rho_total = sum(rho_j),
       eps_global = eps_from_rho(sum(rho_j), delta))
}
# ---- additive composition ------------------------------------------------------
sigma_from_gaussian <- function(S, s) 2 * (S^2) / (s^2)

prop3_sigma_one_shot <- function(Sx_vec, s_x) {
  sum(sigma_from_gaussian(Sx_vec, s_x))
}

prop3_sigma_roundwise_fixed <- function(S_res, s_res, S_est, s_est, R_fixed) {
  R_fixed * ( sigma_from_gaussian(S_res, s_res) + sigma_from_gaussian(S_est, s_est) )
}

eps_from_sigma <- function(sigma, delta) sigma + 2 * sqrt(pmax(sigma,0) * log(1/delta))

# ----- Fold-aware noise calibration: split the target across releases -----
# If CV makes each subject participate in (folds-1) training sets, then the
# number of subject-level releases is m_cv = folds - 1. We calibrate each
# *release* to spend sigma_target / m_cv so the total equals sigma_target.
s_from_eps_one_shot_split <- function(Sx_vec, eps_target, delta, num_releases = 1L) {
  if (!is.finite(eps_target) || eps_target <= 0) return(0)
  sigma_target <- rho_from_eps(eps_target, delta)      # "σ" in Prop 3 (zCDP ρ)
  sigma_each   <- sigma_target / max(1L, as.integer(num_releases))
  K <- sum(2 * (Sx_vec^2))                             # sum_j 2 S_j^2
  sqrt(K / pmax(sigma_each, 1e-12))                    # s_x for *each* release
}

# ---- DP release (whitened) -----------------------------------------------
# Input coefficients C (Q x N), Gram M (Q x Q); clip radius S; noise s
dp_release_whitened <- function(C, M, S, s, ridge = 1e-8) {
  Ms <- sym_eigen_sqrt(M, ridge)
  Z   <- Ms$invhalf %*% C
  nz  <- sqrt(colSums(Z * Z) + 1e-16)
  scl <- pmin(1, S / nz)
  Zc  <- Z %*% diag(scl, nrow = ncol(Z))
  Zdp <- Zc + s * matrix(rnorm(length(Zc)), nrow(Zc), ncol(Zc))
  list(Zdp = Zdp, Sigma = (s^2) * Diagonal(nrow(Z)))
}

# ---- DP-adjusted moments --------------------------------------------------
# Zdp: N x Q (centered inside), Ycoef: N x Qy
form_dp_moments <- function(Zdp, Ycoef, Sigma_x) {
  N <- nrow(Zdp)
  cz <- center_rows(Zdp); Zc <- cz$Xc; zbar <- cz$mean
  cy <- center_rows(Ycoef); Yc <- cy$Xc; ybar <- cy$mean
  Gxx_c  <- crossprod(Zc) / N
  Gxy_c  <- crossprod(Zc, Yc) / N
  Gxx_bar_raw <- (N/(N-1)) * Gxx_c - as.matrix(Sigma_x)
  Gxx_bar <- .psd_project(Gxx_bar_raw, floor = 1e-6)     # <-- stable
  Gxy_bar <- (N/(N-1)) * Gxy_c
  list(Zc = Zc, zbar = zbar, Yc = Yc, ybar = ybar,
       Gxx_bar = .symmetrize(Gxx_bar), Gxy_bar = as.matrix(Gxy_bar))
}

# ---- Penalized FoF solve (stable PD) -------------------------------------
solve_penalized_fof <- function(Gxx_bar, Gxy_bar, Omega_x, Omega_y,
                                lambda_s = 0, lambda_t = 0, lambda_st = 0,
                                stabilize = list(alpha = 0.1, tau = NULL, ridge = 1e-6)) {
  Qx <- nrow(Gxx_bar); Qy <- ncol(Gxy_bar)
  A <- .symmetrize(Gxx_bar + lambda_s * Omega_x)
  
  tau <- stabilize$tau
  if (is.null(tau)) {
    d <- diag(A); tau <- max(mean(abs(d)), 1e-3)
  }
  alpha <- stabilize$alpha %||% 0.1
  ridge <- stabilize$ridge %||% 1e-6
  if (alpha > 0) A <- (1 - alpha) * A + alpha * tau * diag(Qx)
  
  # ensure PD
  ev <- eigen(A, symmetric = TRUE, only.values = TRUE)$values
  if (!all(is.finite(ev)) || min(ev) < ridge) {
    A <- A + (abs(min(ev, na.rm=TRUE)) + ridge) * diag(Qx)
  }
  A <- .symmetrize(A)
  
  H <- kronecker(Diagonal(Qy), A)
  if (lambda_t != 0)  H <- H + lambda_t  * kronecker(Omega_y, Diagonal(Qx))
  if (lambda_st != 0) H <- H + lambda_st * kronecker(Omega_y, Omega_x)
  
  b <- solve(H, as.vector(Gxy_bar), sparse = FALSE)
  matrix(b, nrow = Qx, ncol = Qy)
}

# ---- Selection score (noise-corrected) -----------------------------------
selection_score_corr <- function(Zj, Rm_j, dB, Sigma_xj, N) {
  fit_inc <- Zj %*% dB
  sse_bar  <- fro_sq(fit_inc - Rm_j) / N
  corr_bar <- ((N - 1) / N) * sum(diag(t(dB) %*% Sigma_xj %*% dB))
  sse_bar - corr_bar
}

# ---- Hutchinson DF --------------------------------------------------------
estimate_df_hutchinson <- function(path, Z_list, Sigma_list, Omega_x_list, Omega_y,
                                   Gxx_bar_list, lambda_s, lambda_t, lambda_st,
                                   N, Qy, K = 3) {
  if (length(path) == 0) return(0)
  df_vals <- numeric(K)
  for (k in seq_len(K)) {
    V <- matrix(rnorm(N * Qy), nrow = N, ncol = Qy)
    F <- matrix(0, N, Qy)
    Blist <- lapply(Z_list, function(Z) matrix(0, ncol(Z), Qy))
    for (jj in path) {
      Zj <- Z_list[[jj]]; Bj <- Blist[[jj]]
      R  <- V - F
      Rm_j <- R + Zj %*% Bj
      GxRj <- (N/(N-1)) * (crossprod(Zj, Rm_j) / N)
      dB <- solve_penalized_fof(Gxx_bar_list[[jj]], GxRj,
                                Omega_x_list[[jj]], Omega_y,
                                lambda_s, lambda_t, lambda_st,
                                stabilize = list(alpha = 0.05, ridge = 1e-6))
      Bj <- Bj + dB; F <- F + Zj %*% dB; Blist[[jj]] <- Bj
    }
    df_vals[k] <- sum(V * F)
  }
  mean(df_vals)
}

# ---- S2DP-FGB (no shadowed helpers) --------------------------------------
vfl_dp_foboost <- function(xfd_list, yfd,
                           Sx_vec, sx_vec,
                           Omega_x_list = NULL, Omega_y = NULL,
                           Lfd_x = int2Lfd(2), Lfd_y = int2Lfd(2),
                           lambda_s = 0, lambda_t = 0, lambda_st = 0,
                           nu = 1.0, max_steps = 50,
                           crossfit = TRUE,
                           stop_mode = c("cv","aic_train","aic_train_dp"),
                           min_steps = 5,
                           aic = c("spherical","mv"),
                           aic_c = TRUE,
                           df_K = 3,
                           patience = 3,
                           dB_cap = 10.0) {
  
  stop_mode <- match.arg(stop_mode); aic <- match.arg(aic)
  P <- length(xfd_list); stopifnot(P >= 1, inherits(yfd,"fd"),
                                   length(Sx_vec) == P, length(sx_vec) == P)
  
  # response coeffs (centered)
  ybasis <- yfd$basis
  Ycoef  <- t(yfd$coefs)          # N x Qy
  N <- nrow(Ycoef); Qy <- ncol(Ycoef); stopifnot(N >= 2)
  cY <- center_rows(Ycoef); Yc <- cY$Xc; ybar <- cY$mean
  
  # timing: DP release
  t_rel0 <- Sys.time()
  
  Z_list <- vector("list", P)
  zbar_list <- vector("list", P)
  Sigma_list <- vector("list", P)
  Gxx_bar_list <- vector("list", P)
  Omega_x_used <- vector("list", P)
  
  for (j in seq_len(P)) {
    xfd <- xfd_list[[j]]
    Cx  <- xfd$coefs                  # Qx x N
    Mx  <- inprod(xfd$basis, xfd$basis)
    
    # one-shot DP in whitened coords (TOP-LEVEL helper)
    rel  <- dp_release_whitened(Cx, Mx, S = Sx_vec[j], s = sx_vec[j])
    ZdpN <- t(rel$Zdp)               # N x Qx
    
    # de-biased centered moments with PSD clamp (TOP-LEVEL helper)
    mom  <- form_dp_moments(ZdpN, Yc, Sigma_x = rel$Sigma)
    Z_list[[j]]       <- mom$Zc
    zbar_list[[j]]    <- mom$zbar
    Sigma_list[[j]]   <- rel$Sigma
    Gxx_bar_list[[j]] <- mom$Gxx_bar
    
    # penalty in whitened coords
    OX <- if (is.null(Omega_x_list) || is.null(Omega_x_list[[j]]))
      penalty_matrix(xfd$basis, Lfd_x) else .symmetrize(Omega_x_list[[j]])
    Mxs <- sym_eigen_sqrt(Mx)
    Omega_x_used[[j]] <- .symmetrize(t(Mxs$invhalf) %*% OX %*% Mxs$invhalf)
  }
  release_time_s <- as.numeric(difftime(Sys.time(), t_rel0, units="secs"))
  
  # boosting phase
  t_tr0 <- Sys.time()
  if (is.null(Omega_y)) Omega_y <- penalty_matrix(ybasis, Lfd_y)
  Omega_y <- .symmetrize(Omega_y)
  
  B_list <- lapply(Z_list, function(Z) matrix(0, ncol(Z), Qy))
  Fhat <- matrix(0, N, Qy)
  selected <- integer(0)
  
  idx <- sample.int(N); A <- idx[seq_len(floor(N/2))]; B <- setdiff(seq_len(N), A)
  Fhat_val_B <- matrix(0, length(B), Qy)
  Fhat_val_A <- matrix(0, length(A), Qy)
  best_val <- Inf; best_state <- NULL; no_improve <- 0
  
  # AIC helper (uses TOP-LEVEL Hutchinson DF + stable solver)
  compute_AIC_train <- function(SSE) {
    if (aic == "spherical") {
      df_hat <- estimate_df_hutchinson(selected, Z_list, Sigma_list,
                                       Omega_x_used, Omega_y, Gxx_bar_list,
                                       lambda_s, lambda_t, lambda_st, N, Qy, K = df_K)
      val <- N * Qy * log(SSE / (N * Qy)) +
        2 * df_hat * if (aic_c) (N*Qy / max(1, N*Qy - df_hat - 1)) else 1
    } else {
      Sres <- crossprod(Yc - Fhat) / N
      eps <- 1e-10
      logdet <- as.numeric(determinant(.symmetrize(Sres) + eps * diag(Qy), TRUE)$modulus)
      df_hat <- estimate_df_hutchinson(selected, Z_list, Sigma_list,
                                       Omega_x_used, Omega_y, Gxx_bar_list,
                                       lambda_s, lambda_t, lambda_st, N, Qy, K = df_K)
      pen <- 2 * df_hat * if (aic_c) (N*Qy / max(1, N*Qy - df_hat - 1)) else 1
      val <- N * logdet + pen
    }
    val
  }
  
  for (m in seq_len(max_steps)) {
    R <- Yc - Fhat
    scores <- rep(NA_real_, P)
    dB_list <- vector("list", P); dBA_list <- vector("list", P); dBB_list <- vector("list", P)
    
    for (j in seq_len(P)) {
      Zj <- Z_list[[j]]; Bj <- B_list[[j]]
      Ox <- Omega_x_used[[j]]; Gxxj <- Gxx_bar_list[[j]]
      
      # full increment (TOP-LEVEL stable solver)
      Rm_j <- R + Zj %*% Bj
      GxRj <- (N/(N-1)) * (crossprod(Zj, Rm_j) / N)
      dB   <- solve_penalized_fof(Gxxj, GxRj, Ox, Omega_y,
                                  lambda_s, lambda_t, lambda_st,
                                  stabilize = list(alpha = 0.05, ridge = 1e-6))
      # guard-rail cap
      fn <- sqrt(sum(dB * dB)); if (is.finite(fn) && fn > dB_cap) dB <- dB * (dB_cap / fn)
      dB_list[[j]] <- dB
      
      # cross-fit increments (same solver)
      RmA_j <- R[A, , drop=FALSE] + Zj[A, , drop=FALSE] %*% Bj
      GxRjA <- (length(A)/(length(A)-1)) * (crossprod(Zj[A, , drop=FALSE], RmA_j) / length(A))
      dBA   <- solve_penalized_fof(Gxxj, GxRjA, Ox, Omega_y,
                                   lambda_s, lambda_t, lambda_st,
                                   stabilize = list(alpha = 0.05, ridge = 1e-6))
      fnA <- sqrt(sum(dBA * dBA)); if (is.finite(fnA) && fnA > dB_cap) dBA <- dBA * (dB_cap / fnA)
      dBA_list[[j]] <- dBA
      
      RmB_j <- R[B, , drop=FALSE] + Zj[B, , drop=FALSE] %*% Bj
      GxRjB <- (length(B)/(length(B)-1)) * (crossprod(Zj[B, , drop=FALSE], RmB_j) / length(B))
      dBB   <- solve_penalized_fof(Gxxj, GxRjB, Ox, Omega_y,
                                   lambda_s, lambda_t, lambda_st,
                                   stabilize = list(alpha = 0.05, ridge = 1e-6))
      fnB <- sqrt(sum(dBB * dBB)); if (is.finite(fnB) && fnB > dB_cap) dBB <- dBB * (dB_cap / fnB)
      dBB_list[[j]] <- dBB
      
      # noise-aware selection score
      scores[j] <- selection_score_corr(Zj[B, , drop=FALSE],
                                        R[B, , drop=FALSE] + Zj[B, , drop=FALSE] %*% Bj,
                                        dBA, Sigma_list[[j]], length(B)) +
        selection_score_corr(Zj[A, , drop=FALSE],
                             R[A, , drop=FALSE] + Zj[A, , drop=FALSE] %*% Bj,
                             dBB, Sigma_list[[j]], length(A))
    }
    
    j_star <- which.min(scores); selected <- c(selected, j_star)
    dB <- dB_list[[j_star]]
    B_list[[j_star]] <- B_list[[j_star]] + nu * dB
    Fhat <- Fhat + nu * (Z_list[[j_star]] %*% dB)
    
    SSE_train <- fro_sq(Yc - Fhat)
    if (stop_mode == "cv") {
      Fhat_val_B <- Fhat_val_B + nu * (Z_list[[j_star]][B, , drop=FALSE] %*% dBA_list[[j_star]])
      Fhat_val_A <- Fhat_val_A + nu * (Z_list[[j_star]][A, , drop=FALSE] %*% dBB_list[[j_star]])
      SSE_val <- fro_sq(Yc[B, , drop=FALSE] - Fhat_val_B) +
        fro_sq(Yc[A, , drop=FALSE] - Fhat_val_A)
      val_metric <- SSE_val
    } else if (stop_mode == "aic_train_dp") {
      infl <- 0
      for (jj in seq_len(P)) {
        Bj <- B_list[[jj]]
        infl <- infl + sum(diag(t(Bj) %*% Sigma_list[[jj]] %*% Bj))
      }
      SSE_use <- max(1e-12, SSE_train - (N - 1) * infl)
      val_metric <- compute_AIC_train(SSE_use)
    } else {
      val_metric <- compute_AIC_train(SSE_train)
    }
    
    if (m >= min_steps && (val_metric + 1e-8 < best_val)) {
      best_val <- val_metric
      best_state <- list(B_list = B_list, Fhat = Fhat, selected = selected)
      no_improve <- 0
    } else if (m >= min_steps) {
      no_improve <- no_improve + 1
      if (no_improve >= patience) break
    }
  }
  
  if (!is.null(best_state)) {
    B_list <- best_state$B_list; Fhat <- best_state$Fhat; selected <- best_state$selected
  }
  train_loop_time_s <- as.numeric(difftime(Sys.time(), t_tr0, units="secs"))
  
  yhat_fd <- fd(coef = t(sweep(Fhat, 2, ybar, "+")), basisobj = yfd$basis, fdnames = yfd$fdnames)
  
  list(
    B_list = B_list, selected = selected, yhat_fd = yhat_fd,
    centers = list(ybar = ybar, zbar_list = zbar_list),
    Sigma_list = Sigma_list, Omega_x_list = Omega_x_used, Omega_y = Omega_y,
    Z_list = Z_list, Yc = Yc, stop_mode = stop_mode,
    timing = list(release_time_s = release_time_s,
                  train_loop_time_s = train_loop_time_s,
                  total_time_s = release_time_s + train_loop_time_s)
  )
}

# --- REPLACE predict() with this (unchanged logic, no inner helpers) -------
predict_vfl_dp_foboost <- function(fit, xfd_list_new) {
  P <- length(fit$B_list); stopifnot(length(xfd_list_new) == P)
  Nnew <- ncol(xfd_list_new[[1]]$coefs); Qy <- nrow(fit$yhat_fd$coefs)
  Yhat <- matrix(0, Nnew, Qy)
  for (j in seq_len(P)) {
    Cx_new <- xfd_list_new[[j]]$coefs
    Mx  <- inprod(xfd_list_new[[j]]$basis, xfd_list_new[[j]]$basis)
    Mxs <- sym_eigen_sqrt(Mx)
    Znew <- t(Mxs$invhalf %*% Cx_new)
    Znew <- sweep(Znew, 2, fit$centers$zbar_list[[j]], "-")
    Yhat <- Yhat + Znew %*% fit$B_list[[j]]
  }
  Yhat <- sweep(Yhat, 2, fit$centers$ybar, "+")
  fd(coef = t(Yhat), basisobj = fit$yhat_fd$basis, fdnames = fit$yhat_fd$fdnames)
}
# ======================= functions_roundwise_dp.R (fixed rounds) =======================
suppressPackageStartupMessages({ library(fda); library(Matrix) })
# expects source("functions.R") first (for shared helpers and stable solver)

.sym_sqrt <- function(M, ridge = 1e-8) sym_eigen_sqrt(M, ridge)

# Residual broadcast (active -> passives), whitened Y
dp_release_residual_round <- function(Rcoef_NxQy, My, S_res, s_res) {
  rel <- dp_release_whitened(C = t(Rcoef_NxQy), M = My, S = S_res, s = s_res)
  t(rel$Zdp)
}

# Operator increment DP (passive -> active): clip Frobenius, add isotropic noise
dp_release_operator_round <- function(dB, S_est, s_est) {
  fn <- sqrt(fro_sq(dB) + 1e-16)
  scale <- min(1, S_est / fn)
  dB_clip <- dB * scale
  dB_clip + matrix(rnorm(length(dB_clip), sd = s_est), nrow(dB_clip), ncol(dB_clip))
}

roundwise_dp_vfl_boost <- function(
    xfd_list, yfd,
    # fixed-round privacy knobs (already calibrated outside to meet eps_target)
    S_res, s_res,              # residual clip/noise per round (whitened Y)
    S_est, s_est,              # estimator clip/noise per round (ΔB)
    # penalties & boosting
    Omega_x_list = NULL, Omega_y = NULL,
    Lfd_x = int2Lfd(2), Lfd_y = int2Lfd(2),
    lambda_s = 0, lambda_t = 0, lambda_st = 0,
    nu = 0.3,
    R_fixed = 30,              # <-- EXACT number of rounds to run
    use_crossfit = TRUE,
    delta_total = 1e-5,        # kept for interface symmetry
    dB_cap = 10.0              # guard-rail for pathological steps
) {
  P <- length(xfd_list); stopifnot(P >= 1, inherits(yfd,"fd"))
  ybasis <- yfd$basis; My <- inprod(ybasis, ybasis)
  Ycoef <- t(yfd$coefs)               # N x Qy
  N <- nrow(Ycoef); Qy <- ncol(Ycoef)
  ybar <- colMeans(Ycoef); Yc <- sweep(Ycoef, 2, ybar, "-")
  
  # whiten features and penalties
  Z_list <- vector("list", P)
  Gxx_list <- vector("list", P)
  Ox_list  <- vector("list", P)
  Qx <- nrow(xfd_list[[1]]$coefs)
  
  for (j in seq_len(P)) {
    xfd <- xfd_list[[j]]
    Mx  <- inprod(xfd$basis, xfd$basis)
    Msx <- .sym_sqrt(Mx)
    Zj <- t(Msx$invhalf %*% xfd$coefs)   # N x Qx
    Z_list[[j]] <- Zj
    Gxx_list[[j]] <- crossprod(Zj) / N
    OX <- if (is.null(Omega_x_list) || is.null(Omega_x_list[[j]]))
      eval.penalty(xfd$basis, Lfd_x) else .symmetrize(Omega_x_list[[j]])
    Ox_list[[j]] <- .symmetrize(t(Msx$invhalf) %*% OX %*% Msx$invhalf)
  }
  if (is.null(Omega_y)) Omega_y <- eval.penalty(ybasis, Lfd_y)
  Omega_y <- .symmetrize(Omega_y)
  
  # state
  Fhat <- matrix(0, N, Qy)
  B_list <- lapply(seq_len(P), function(j) matrix(0, Qx, Qy))
  selected <- integer(0)
  
  # crossfit split (for selection only — no early stopping)
  idx <- sample.int(N); A <- idx[seq_len(floor(N/2))]; B <- setdiff(seq_len(N), A)
  
  # run EXACTLY R_fixed rounds
  for (r in seq_len(R_fixed)) {
    Rcoef <- Yc - Fhat
    
    # (1) DP residual broadcast for this round
    Udp <- dp_release_residual_round(Rcoef_NxQy = Rcoef, My = My, S_res = S_res, s_res = s_res)
    
    # (2) parties compute candidate increments
    rss_vec <- rep(NA_real_, P)
    dB_list <- vector("list", P); dBA_list <- vector("list", P); dBB_list <- vector("list", P)
    
    for (j in seq_len(P)) {
      Zj <- Z_list[[j]]; Gxx <- Gxx_list[[j]]; Ox <- Ox_list[[j]]
      
      GxR <- crossprod(Zj, Udp) / N
      dB  <- solve_penalized_fof(Gxx, GxR, Ox, Omega_y,
                                 lambda_s, lambda_t, lambda_st,
                                 stabilize = list(alpha = 0.05, ridge = 1e-6))
      # cap step for stability
      fn <- sqrt(sum(dB * dB)); if (is.finite(fn) && fn > dB_cap) dB <- dB * (dB_cap / fn)
      dB_list[[j]] <- dB
      rss_vec[j] <- fro_sq(Udp - Zj %*% dB)
      
      if (use_crossfit) {
        GA <- length(A); GB <- length(B)
        dBA <- solve_penalized_fof(Gxx, crossprod(Zj[A, , drop=FALSE], Udp[A, , drop=FALSE]) / GA,
                                   Ox, Omega_y, lambda_s, lambda_t, lambda_st,
                                   stabilize = list(alpha = 0.05, ridge = 1e-6))
        dBB <- solve_penalized_fof(Gxx, crossprod(Zj[B, , drop=FALSE], Udp[B, , drop=FALSE]) / GB,
                                   Ox, Omega_y, lambda_s, lambda_t, lambda_st,
                                   stabilize = list(alpha = 0.05, ridge = 1e-6))
        # cap
        fnA <- sqrt(sum(dBA*dBA)); if (is.finite(fnA) && fnA > dB_cap) dBA <- dBA * (dB_cap / fnA)
        fnB <- sqrt(sum(dBB*dBB)); if (is.finite(fnB) && fnB > dB_cap) dBB <- dBB * (dB_cap / fnB)
        dBA_list[[j]] <- dBA; dBB_list[[j]] <- dBB
      }
    }
    
    # (3) select party this round
    j_star <- if (use_crossfit) {
      rss_cf <- sapply(seq_len(P), function(j)
        fro_sq(Udp[B, , drop=FALSE] - Z_list[[j]][B, , drop=FALSE] %*% dBA_list[[j]]) +
          fro_sq(Udp[A, , drop=FALSE] - Z_list[[j]][A, , drop=FALSE] %*% dBB_list[[j]]))
      which.min(rss_cf)
    } else which.min(rss_vec)
    
    selected <- c(selected, j_star)
    
    # (4) DP on estimator increment, then update
    dB_dp <- dp_release_operator_round(dB_list[[j_star]], S_est = S_est, s_est = s_est)
    B_list[[j_star]] <- B_list[[j_star]] + nu * dB_dp
    Fhat <- Fhat + nu * (Z_list[[j_star]] %*% dB_dp)
  }
  
  # final fd (train-space prediction)
  yhat_train_fd <- fd(coef = t(sweep(Fhat, 2, ybar, "+")), basisobj = ybasis, fdnames = yfd$fdnames)
  
  list(
    ybar = ybar, B_list = B_list, ybasis = ybasis,
    selected = selected, rounds = R_fixed,
    # IMPORTANT: fixed total privacy (reported as the target by caller)
    eps = list(eps_global = NA_real_),   # placeholder; pipeline fills with eps_target
    train_yhat_fd = yhat_train_fd,
    dims = list(Qx = nrow(xfd_list[[1]]$coefs), Qy = Qy, P = P)
  )
}

predict_roundwise_dp <- function(fit, xfd_list_new) {
  P <- fit$dims$P; stopifnot(length(xfd_list_new) == P)
  Qy <- fit$dims$Qy; Nnew <- ncol(xfd_list_new[[1]]$coefs)
  Yhat <- matrix(0, Nnew, Qy)
  for (j in seq_len(P)) {
    xfd <- xfd_list_new[[j]]
    Mx  <- inprod(xfd$basis, xfd$basis)
    Msx <- .sym_sqrt(Mx)
    Znew <- t(Msx$invhalf %*% xfd$coefs)
    Yhat <- Yhat + Znew %*% fit$B_list[[j]]
  }
  Yhat <- sweep(Yhat, 2, fit$ybar, "+")
  fd(coef = t(Yhat), basisobj = fit$ybasis, fdnames = xfd_list_new[[1]]$fdnames)
}

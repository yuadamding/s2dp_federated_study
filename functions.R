############################################################################
# DP FoF + VFL Functional Boosting (paper-aligned, whitened one-shot DP)
# - One predictor per passive party (P = number of parties = p features)
# - Whitened release: z = M^{-1/2} α (clip in ||·||_2), noise N(0, s^2 I)
# - AIC stopping with Hutchinson DF (default)
############################################################################

suppressPackageStartupMessages({
  library(fda)
  library(Matrix)
})

# ---------- Numeric helpers ------------------------------------------------
.symmetrize <- function(M) (M + t(M)) / 2
fro_sq <- function(A) sum(A * A)
`%||%` <- function(a, b) if (!is.null(a)) a else b

sym_eigen_sqrt <- function(M, ridge = 1e-8) {
  M <- .symmetrize(M)
  ee <- eigen(M, symmetric = TRUE)
  lam <- pmax(ee$values, ridge)
  U <- ee$vectors
  list(
    half    = U %*% diag(sqrt(lam),  nrow = length(lam)) %*% t(U),
    invhalf = U %*% diag(1/sqrt(lam), nrow = length(lam)) %*% t(U)
  )
}

center_rows <- function(X) {
  mu <- colMeans(X)
  list(Xc = sweep(X, 2, mu, FUN = "-"), mean = mu)
}

penalty_matrix <- function(basis, Lfdobj = int2Lfd(2)) eval.penalty(basis, Lfdobj)

# ---------- zCDP accounting ------------------------------------------------
# For whitened release with replace-one sensitivity Δ = 2 Sx and noise σ = s:
#   rho_j = Δ^2 / (2σ^2) = 2 Sx^2 / s^2
#   eps(ρ,δ) = ρ + 2 sqrt(ρ log(1/δ))
eps_from_rho <- function(rho, delta) rho + 2 * sqrt(pmax(rho, 0) * log(1/delta))

eps_from_sx_zcdp_vec <- function(Sx_vec, sx, delta) {
  rho_j <- 2 * (Sx_vec^2) / (sx^2)
  list(
    rho_j = rho_j,
    eps_j = eps_from_rho(rho_j, delta),
    rho_total = sum(rho_j),
    eps_global = eps_from_rho(sum(rho_j), delta)
  )
}

# ---------- DP release (paper-aligned: whitened) ---------------------------
# Input coefficients C (Q x N), Gram M (Q x Q)
# Output: Zdp (Q x N) in whitened coordinates, Sigma = s^2 I_Q
dp_release_whitened <- function(C, M, S, s, ridge = 1e-8) {
  Ms <- sym_eigen_sqrt(M, ridge)
  # RKHS norm == Euclidean norm in whitened space
  Z   <- Ms$invhalf %*% C               # Q x N
  nz  <- sqrt(colSums(Z * Z) + 1e-16)
  scl <- pmin(1, S / nz)
  Zc  <- Z %*% diag(scl, nrow = ncol(Z))
  Zdp <- Zc + s * matrix(rnorm(length(Zc)), nrow(Zc), ncol(Zc))
  list(Zdp = Zdp, Sigma = (s^2) * Diagonal(nrow(Z)))
}

# ---------- DP-adjusted moments -------------------------------------------
# Zdp: N x Q (centered inside), Ycoef: N x Qy
form_dp_moments <- function(Zdp, Ycoef, Sigma_x) {
  N <- nrow(Zdp)
  cz <- center_rows(Zdp); Zc <- cz$Xc; zbar <- cz$mean
  cy <- center_rows(Ycoef); Yc <- cy$Xc; ybar <- cy$mean
  Gxx_c  <- crossprod(Zc) / N
  Gxy_c  <- crossprod(Zc, Yc) / N
  Gxx_bar <- (N/(N-1)) * Gxx_c - as.matrix(Sigma_x)
  Gxy_bar <- (N/(N-1)) * Gxy_c
  list(Zc = Zc, zbar = zbar, Yc = Yc, ybar = ybar,
       Gxx_bar = .symmetrize(Gxx_bar), Gxy_bar = as.matrix(Gxy_bar))
}

# ---------- Penalized FoF solve (Kronecker-sum system) ---------------------
solve_penalized_fof <- function(Gxx_bar, Gxy_bar, Omega_x, Omega_y,
                                lambda_s = 0, lambda_t = 0, lambda_st = 0,
                                stabilize = list(alpha = 0, tau = NULL, ridge = 1e-8)) {
  Qx <- nrow(Gxx_bar); Qy <- ncol(Gxy_bar)
  A <- .symmetrize(Gxx_bar + lambda_s * Omega_x)
  alpha <- stabilize$alpha %||% 0
  tau   <- stabilize$tau
  ridge <- stabilize$ridge %||% 1e-8
  if (is.null(tau)) tau <- sum(diag(Gxx_bar)) / max(1, Qx)
  if (alpha > 0) A <- (1 - alpha) * A + alpha * tau * diag(Qx)
  A <- .symmetrize(A + ridge * diag(Qx))
  
  H <- kronecker(Diagonal(Qy), A)
  if (lambda_t != 0)  H <- H + lambda_t  * kronecker(Omega_y, Diagonal(Qx))
  if (lambda_st != 0) H <- H + lambda_st * kronecker(Omega_y, Omega_x)
  
  b <- solve(H, as.vector(Gxy_bar), sparse = FALSE)
  matrix(b, nrow = Qx, ncol = Qy)
}

# ---------- Selection score (FIXED scaling) --------------------------------
# Per-sample SSE minus ((N-1)/N) * tr(dBᵀ Σ_x dB)
selection_score_corr <- function(Zj, Rm_j, dB, Sigma_xj, N) {
  fit_inc <- Zj %*% dB
  sse_bar  <- fro_sq(fit_inc - Rm_j) / N
  corr_bar <- ((N - 1) / N) * sum(diag(t(dB) %*% Sigma_xj %*% dB))
  sse_bar - corr_bar
}

# ---------- Hutchinson DF --------------------------------------------------
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
      Bj <- Bj + dB
      F  <- F + Zj %*% dB
      Blist[[jj]] <- Bj
    }
    df_vals[k] <- sum(V * F)
  }
  mean(df_vals)
}

# ---------- VFL DP boosting (one predictor per party) ----------------------
vfl_dp_foboost <- function(xfd_list, yfd,
                           Sx_vec, sx_vec,
                           Omega_x_list = NULL, Omega_y = NULL,
                           Lfd_x = int2Lfd(2), Lfd_y = int2Lfd(2),
                           lambda_s = 0, lambda_t = 0, lambda_st = 0,
                           nu = 1.0, max_steps = 50,
                           crossfit = TRUE,
                           stop_mode = c("aic_train", "aic_train_dp", "cv"),
                           min_steps = 5,
                           aic = c("spherical", "mv"),
                           aic_c = TRUE,
                           df_K = 3,
                           patience = 3) {
  
  stop_mode <- match.arg(stop_mode)
  aic <- match.arg(aic)
  
  P <- length(xfd_list)
  stopifnot(P >= 1, inherits(yfd, "fd"),
            length(Sx_vec) == P, length(sx_vec) == P)
  
  # Y coefficients and center once
  ybasis <- yfd$basis
  Ycoef  <- t(yfd$coefs)                  # N x Qy
  N <- nrow(Ycoef); Qy <- ncol(Ycoef)
  stopifnot(N >= 2)
  cY <- center_rows(Ycoef); Yc <- cY$Xc; ybar <- cY$mean
  
  # Per-party release in whitened space; build Ω̃_x = M^{-1/2} Ω_x M^{-1/2}
  Z_list <- vector("list", P)
  zbar_list <- vector("list", P)
  Sigma_list <- vector("list", P)
  Gxx_bar_list <- vector("list", P)
  Omega_x_used <- vector("list", P)
  
  for (j in seq_len(P)) {
    xfd <- xfd_list[[j]]
    Cx  <- xfd$coefs                 # Qx x N
    Mx  <- inprod(xfd$basis, xfd$basis)
    
    # DP release (whitened)
    rel  <- dp_release_whitened(Cx, Mx, S = Sx_vec[j], s = sx_vec[j])
    ZdpN <- t(rel$Zdp)               # N x Qx
    mom  <- form_dp_moments(ZdpN, Yc, Sigma_x = rel$Sigma)
    Z_list[[j]]       <- mom$Zc
    zbar_list[[j]]    <- mom$zbar
    Sigma_list[[j]]   <- rel$Sigma
    Gxx_bar_list[[j]] <- mom$Gxx_bar
    
    # Penalty in whitened coordinates
    OX <- if (is.null(Omega_x_list) || is.null(Omega_x_list[[j]])) {
      penalty_matrix(xfd$basis, Lfd_x)
    } else .symmetrize(Omega_x_list[[j]])
    Mxs <- sym_eigen_sqrt(Mx)
    Omega_x_used[[j]] <- .symmetrize(t(Mxs$invhalf) %*% OX %*% Mxs$invhalf)
  }
  
  if (is.null(Omega_y)) Omega_y <- penalty_matrix(ybasis, Lfd_y)
  Omega_y <- .symmetrize(Omega_y)
  
  # Init
  B_list <- lapply(Z_list, function(Z) matrix(0, ncol(Z), Qy))
  Fhat <- matrix(0, N, Qy)
  selected <- integer(0)
  sse_trace <- numeric(0); aic_trace <- numeric(0); df_trace <- numeric(0)
  
  # Cross-fit split (for selection or CV if requested)
  idx <- sample.int(N); A <- idx[seq_len(floor(N/2))]; B <- setdiff(seq_len(N), A)
  Fhat_val_B <- matrix(0, length(B), Qy)
  Fhat_val_A <- matrix(0, length(A), Qy)
  best_val <- Inf; best_state <- NULL; no_improve <- 0
  
  compute_AIC_train <- function(SSE) {
    if (aic == "spherical") {
      df_hat <- estimate_df_hutchinson(selected, Z_list, Sigma_list,
                                       Omega_x_used, Omega_y, Gxx_bar_list,
                                       lambda_s, lambda_t, lambda_st,
                                       N, Qy, K = df_K)
      val <- N * Qy * log(SSE / (N * Qy)) + 2 * df_hat
      if (aic_c) {
        denom <- max(1, N * Qy - df_hat - 1)
        val <- N * Qy * log(SSE / (N * Qy)) + 2 * df_hat * (N * Qy / denom)
      }
      df_trace <<- c(df_trace, df_hat); val
    } else {
      Sres <- crossprod(Yc - Fhat) / N
      eps <- 1e-10
      val <- N * as.numeric(determinant(.symmetrize(Sres) + eps * diag(Qy), logarithm = TRUE)$modulus)
      df_hat <- estimate_df_hutchinson(selected, Z_list, Sigma_list,
                                       Omega_x_used, Omega_y, Gxx_bar_list,
                                       lambda_s, lambda_t, lambda_st,
                                       N, Qy, K = df_K)
      if (aic_c) {
        denom <- max(1, N * Qy - df_hat - 1)
        val <- val + 2 * df_hat * (N * Qy / denom)
      } else {
        val <- val + 2 * df_hat
      }
      df_trace <<- c(df_trace, df_hat); val
    }
  }
  
  for (m in seq_len(max_steps)) {
    R <- Yc - Fhat
    
    scores <- rep(NA_real_, P)
    dB_list  <- vector("list", P)
    dBA_list <- vector("list", P)
    dBB_list <- vector("list", P)
    
    for (j in seq_len(P)) {
      Zj <- Z_list[[j]]; Bj <- B_list[[j]]
      Ox <- Omega_x_used[[j]]; Gxxj <- Gxx_bar_list[[j]]
      
      # Full increment
      Rm_j <- R + Zj %*% Bj
      GxRj <- (N/(N-1)) * (crossprod(Zj, Rm_j) / N)
      dB   <- solve_penalized_fof(Gxxj, GxRj, Ox, Omega_y,
                                  lambda_s, lambda_t, lambda_st,
                                  stabilize = list(alpha = 0.05, ridge = 1e-6))
      dB_list[[j]] <- dB
      
      # Cross-fit increments
      RmA_j <- R[A, , drop=FALSE] + Zj[A, , drop=FALSE] %*% Bj
      GxRjA <- ((length(A))/(length(A)-1)) * (crossprod(Zj[A, , drop=FALSE], RmA_j) / length(A))
      dBA   <- solve_penalized_fof(Gxxj, GxRjA, Ox, Omega_y,
                                   lambda_s, lambda_t, lambda_st,
                                   stabilize = list(alpha = 0.05, ridge = 1e-6))
      dBA_list[[j]] <- dBA
      
      RmB_j <- R[B, , drop=FALSE] + Zj[B, , drop=FALSE] %*% Bj
      GxRjB <- ((length(B))/(length(B)-1)) * (crossprod(Zj[B, , drop=FALSE], RmB_j) / length(B))
      dBB   <- solve_penalized_fof(Gxxj, GxRjB, Ox, Omega_y,
                                   lambda_s, lambda_t, lambda_st,
                                   stabilize = list(alpha = 0.05, ridge = 1e-6))
      dBB_list[[j]] <- dBB
      
      # Noise-aware selection (scaled)
      scoreAB <- selection_score_corr(Zj[B, , drop=FALSE],
                                      R[B, , drop=FALSE] + Zj[B, , drop=FALSE] %*% Bj,
                                      dBA, Sigma_list[[j]], length(B))
      scoreBA <- selection_score_corr(Zj[A, , drop=FALSE],
                                      R[A, , drop=FALSE] + Zj[A, , drop=FALSE] %*% Bj,
                                      dBB, Sigma_list[[j]], length(A))
      scores[j] <- scoreAB + scoreBA
    }
    
    j_star <- which.min(scores)
    selected <- c(selected, j_star)
    
    # Update train fit
    dB <- dB_list[[j_star]]
    B_list[[j_star]] <- B_list[[j_star]] + nu * dB
    Fhat <- Fhat + nu * (Z_list[[j_star]] %*% dB)
    
    # Update cross-fit predictions
    Fhat_val_B <- Fhat_val_B + nu * (Z_list[[j_star]][B, , drop=FALSE] %*% dBA_list[[j_star]])
    Fhat_val_A <- Fhat_val_A + nu * (Z_list[[j_star]][A, , drop=FALSE] %*% dBB_list[[j_star]])
    
    SSE_train <- fro_sq(Yc - Fhat)
    sse_trace <- c(sse_trace, SSE_train)
    
    if (stop_mode == "cv") {
      SSE_val <- fro_sq(Yc[B, , drop=FALSE] - Fhat_val_B) +
        fro_sq(Yc[A, , drop=FALSE] - Fhat_val_A)
      val_metric <- SSE_val
      is_better <- (SSE_val + 1e-8 < best_val)
    } else if (stop_mode == "aic_train_dp") {
      infl <- 0
      for (jj in seq_len(P)) {
        Bj <- B_list[[jj]]
        infl <- infl + sum(diag(t(Bj) %*% Sigma_list[[jj]] %*% Bj))
      }
      SSE_use <- max(1e-12, SSE_train - (N - 1) * infl)
      AICm <- compute_AIC_train(SSE_use)
      aic_trace <- c(aic_trace, AICm)
      val_metric <- AICm
      is_better <- (AICm + 1e-8 < best_val)
    } else {  # "aic_train" (default)
      AICm <- compute_AIC_train(SSE_train)
      aic_trace <- c(aic_trace, AICm)
      val_metric <- AICm
      is_better <- (AICm + 1e-8 < best_val)
    }
    
    if (m >= min_steps && is_better) {
      best_val <- val_metric
      best_state <- list(B_list = B_list, Fhat = Fhat, selected = selected,
                         sse_trace = sse_trace, aic_trace = aic_trace)
      no_improve <- 0
    } else if (m >= min_steps) {
      no_improve <- no_improve + 1
      if (no_improve >= patience) break
    }
  }
  
  if (!is.null(best_state)) {
    B_list   <- best_state$B_list
    Fhat     <- best_state$Fhat
    selected <- best_state$selected
    sse_trace <- best_state$sse_trace
    aic_trace <- best_state$aic_trace
  }
  
  yhat_fd <- fd(coef = t(sweep(Fhat, 2, ybar, FUN = "+")),
                basisobj = yfd$basis, fdnames = yfd$fdnames)
  
  list(
    B_list = B_list, selected = selected,
    yhat_fd = yhat_fd,
    centers = list(ybar = ybar, zbar_list = zbar_list),
    Sigma_list = Sigma_list,
    Omega_x_list = Omega_x_used, Omega_y = Omega_y,
    Z_list = Z_list, Yc = sweep(Yc, 2, 0, FUN = "+"),
    traces = list(sse = sse_trace, aic = aic_trace),
    stop_mode = stop_mode
  )
}

predict_vfl_dp_foboost <- function(fit, xfd_list_new) {
  P <- length(fit$B_list); stopifnot(length(xfd_list_new) == P)
  Nnew <- ncol(xfd_list_new[[1]]$coefs)
  Qy <- nrow(fit$yhat_fd$coefs)
  Yhat <- matrix(0, Nnew, Qy)
  for (j in seq_len(P)) {
    Cx_new <- xfd_list_new[[j]]$coefs
    # NOTE: prediction uses *centered* whitened coordinates would require M^{-1/2},
    # but we trained in whitened Z space already; at prediction time we use the
    # same centering stored for Z (zbar_list) applied to unwhitened Cx via M^{-1/2}.
    Mx  <- inprod(xfd_list_new[[j]]$basis, xfd_list_new[[j]]$basis)
    Mxs <- sym_eigen_sqrt(Mx)
    Znew <- t(Mxs$invhalf %*% Cx_new)                     # N x Q
    Znew <- sweep(Znew, 2, fit$centers$zbar_list[[j]], FUN = "-")
    Yhat <- Yhat + Znew %*% fit$B_list[[j]]
  }
  Yhat <- sweep(Yhat, 2, fit$centers$ybar, FUN = "+")
  fd(coef = t(Yhat), basisobj = fit$yhat_fd$basis, fdnames = fit$yhat_fd$fdnames)
}

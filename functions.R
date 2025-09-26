############################################################################
# Differentially Private Penalized Function-on-Function Regression
# and VFL Functional Boosting with CV-stopped, DP-aware selection
#
# Dependencies: fda, Matrix
############################################################################

suppressPackageStartupMessages({
  library(fda)
  library(Matrix)
})

# ---------- Numeric helpers ------------------------------------------------

.symmetrize <- function(M) (M + t(M)) / 2

sym_eigen_sqrt <- function(M, ridge = 1e-8) {
  M <- .symmetrize(M)
  ee <- eigen(M, symmetric = TRUE)
  lam <- pmax(ee$values, ridge)
  U <- ee$vectors
  Mhalf    <- U %*% diag(sqrt(lam), nrow = length(lam)) %*% t(U)
  Minvhalf <- U %*% diag(1/sqrt(lam), nrow = length(lam)) %*% t(U)
  list(half = Mhalf, invhalf = Minvhalf)
}

center_rows <- function(X) {
  mu <- colMeans(X)
  list(Xc = sweep(X, 2, mu, FUN = "-"), mean = mu)
}

fro_sq <- function(A) sum(A * A)

`%||%` <- function(a, b) if (!is.null(a)) a else b

# ---------- DP on coefficient space ---------------------------------------

# Per-record Gaussian mechanism in coefficient space with RKHS-aware clipping.
dp_release_coefficients <- function(C, M, S, s, ridge = 1e-8) {
  # C: Q x N,  M: Q x Q (Gram),  clip radius S, noise std s
  Q <- nrow(C); N <- ncol(C)
  M <- .symmetrize(M)
  Mhalf <- sym_eigen_sqrt(M, ridge)$half
  # RKHS norms ||c||_M for each column
  denom <- sqrt(colSums(C * (M %*% C)) + 1e-16)
  scales <- pmin(1, S / denom)
  Cclip <- C %*% diag(scales, nrow = N)
  # Gaussian mechanism: e ~ N(0, s^2 M)
  G <- matrix(rnorm(Q * N), Q, N)
  E <- s * (Mhalf %*% G)
  Cdp <- Cclip + E
  list(Cdp = Cdp, Cclip = Cclip, M = M, Sigma = (s^2) * M, Mhalf = Mhalf)
}

# ---------- Penalty matrices ----------------------------------------------

penalty_matrix <- function(basis, Lfdobj = int2Lfd(2)) {
  eval.penalty(basis, Lfdobj)
}

# ---------- DP-corrected moments (finite-sample correction) ----------------

form_dp_moments <- function(Zdp, Ycoef, Sigma_x, s_y = 0, Sigma_y = NULL) {
  # Zdp: N x Qx (post-DP, unwhitened); Ycoef: N x Qy (response coefficients)
  N <- nrow(Zdp)
  stopifnot(N >= 2, nrow(Ycoef) == N)
  cz <- center_rows(Zdp); Zc <- cz$Xc; zbar <- cz$mean
  cy <- center_rows(Ycoef); Yc <- cy$Xc; ybar <- cy$mean
  Gxx_c <- crossprod(Zc) / N
  Gxy_c <- crossprod(Zc, Yc) / N
  Gxx_bar <- (N/(N-1)) * Gxx_c - Sigma_x
  Gxy_bar <- (N/(N-1)) * Gxy_c
  list(Zc = Zc, zbar = zbar, Yc = Yc, ybar = ybar,
       Gxx_bar = as.matrix(.symmetrize(Gxx_bar)),
       Gxy_bar = as.matrix(Gxy_bar))
}

# ---------- Generalized Sylvester/Kronecker solver -------------------------

solve_penalized_fof <- function(Gxx_bar, Gxy_bar, Omega_x, Omega_y,
                                lambda_s = 0, lambda_t = 0, lambda_st = 0,
                                stabilize = list(alpha = 0, tau = NULL, ridge = 1e-8)) {
  Qx <- nrow(Gxx_bar); Qy <- ncol(Gxy_bar)
  stopifnot(ncol(Gxx_bar) == Qx, nrow(Gxy_bar) == Qx,
            nrow(Omega_x) == Qx, ncol(Omega_x) == Qx,
            nrow(Omega_y) == Qy, ncol(Omega_y) == Qy)
  
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
  
  d <- as.vector(Gxy_bar)
  b <- solve(H, d, sparse = FALSE)
  matrix(b, nrow = Qx, ncol = Qy)
}

# ---------- Centralized DP FoF (optional) ----------------------------------

dp_fof_fit <- function(xfd, yfd,
                       Sx, sx, Sy = NULL, sy = 0,
                       Omega_x = NULL, Omega_y = NULL,
                       Lfd_x = int2Lfd(2), Lfd_y = int2Lfd(2),
                       lambda_s = 0, lambda_t = 0, lambda_st = 0,
                       stabilize = list(alpha = 0.0, ridge = 1e-8)) {
  stopifnot(inherits(xfd, "fd"), inherits(yfd, "fd"))
  xbasis <- xfd$basis; ybasis <- yfd$basis
  Cx <- xfd$coefs; Cy <- yfd$coefs
  Qx <- nrow(Cx); Qy <- nrow(Cy); N <- ncol(Cx); stopifnot(ncol(Cy) == N)
  Mx <- inprod(xbasis, xbasis); My <- inprod(ybasis, ybasis)
  
  dpX <- dp_release_coefficients(Cx, Mx, S = Sx, s = sx)
  Zdp <- t(dpX$Cdp)                 # N x Qx
  
  # Work in coefficient space for Y (alpha==coefficients if from smooth.basis)
  Ycoef <- t(Cy)                    # N x Qy
  if (!is.null(Sy) && sy > 0) {
    dpY <- dp_release_coefficients(Cy, My, S = Sy, s = sy)
    Ycoef <- t(dpY$Cdp)
  }
  
  mom <- form_dp_moments(Zdp, Ycoef, Sigma_x = dpX$Sigma, s_y = sy)
  if (is.null(Omega_x)) Omega_x <- penalty_matrix(xbasis, Lfd_x)
  if (is.null(Omega_y)) Omega_y <- penalty_matrix(ybasis, Lfd_y)
  Omega_x <- .symmetrize(Omega_x); Omega_y <- .symmetrize(Omega_y)
  
  B <- solve_penalized_fof(mom$Gxx_bar, mom$Gxy_bar, Omega_x, Omega_y,
                           lambda_s, lambda_t, lambda_st,
                           stabilize = list(alpha = stabilize$alpha %||% 0, ridge = stabilize$ridge %||% 1e-8))
  
  Yhat_center <- mom$Zc %*% B
  Yhat_orig   <- sweep(Yhat_center, 2, mom$ybar, FUN = "+")
  yhat_fd <- fd(coef = t(Yhat_orig), basisobj = ybasis, fdnames = yfd$fdnames)
  
  list(
    B = B, yhat_fd = yhat_fd,
    centers = list(ybar = mom$ybar, zbar = mom$zbar),
    Sigma_x = dpX$Sigma,
    Mx = Mx, My = My,
    Omega_x = Omega_x, Omega_y = Omega_y,
    Zc = mom$Zc, Yc = mom$Yc
  )
}

predict_dp_fof <- function(fit, xfd_new, center = TRUE) {
  stopifnot(inherits(xfd_new, "fd"))
  Cx_new <- xfd_new$coefs
  Z_new  <- t(Cx_new)
  if (center) Z_new <- sweep(Z_new, 2, fit$centers$zbar, FUN = "-")
  Yhat <- Z_new %*% fit$B
  if (center) Yhat <- sweep(Yhat, 2, fit$centers$ybar, FUN = "+")
  fd(coef = t(Yhat), basisobj = fit$yhat_fd$basis, fdnames = xfd_new$fdnames)
}

# ---------- VFL DP boosting (Algorithm 2) -----------------------------------

selection_score_corr <- function(Zj, Rm_j, dB, Sigma_xj, N) {
  fit_inc <- Zj %*% dB
  sse <- fro_sq(fit_inc - Rm_j)
  corr <- (N - 1) * sum(diag(t(dB) %*% Sigma_xj %*% dB))
  sse - corr
}

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
      Zj <- Z_list[[jj]]
      Bj <- Blist[[jj]]
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

vfl_dp_foboost <- function(xfd_list, yfd,
                           Sx_vec, sx_vec,
                           Omega_x_list = NULL,
                           Omega_y = NULL, Lfd_x = int2Lfd(2), Lfd_y = int2Lfd(2),
                           lambda_s = 0, lambda_t = 0, lambda_st = 0,
                           nu = 1.0,
                           max_steps = 50,
                           crossfit = TRUE,                 # for selection
                           stop_mode = c("cv", "aic_train", "aic_train_dp"),
                           min_steps = 5,
                           aic = c("spherical", "mv"),
                           aic_c = TRUE,
                           df_K = 3,
                           patience = 3,
                           sse_correct_dp = FALSE) {       # used only for "aic_train_dp"
  stop_mode <- match.arg(stop_mode)
  aic <- match.arg(aic)
  P <- length(xfd_list)
  stopifnot(P >= 1, inherits(yfd, "fd"), length(Sx_vec) == P, length(sx_vec) == P)
  
  # Prepare Y (coefficient space; centered once)
  ybasis <- yfd$basis
  Cy <- yfd$coefs
  Ycoef <- t(Cy)                       # N x Qy
  N <- nrow(Ycoef); Qy <- ncol(Ycoef)
  stopifnot(N >= 2)
  cY <- center_rows(Ycoef); Yc <- cY$Xc; ybar <- cY$mean
  
  # Per-party DP designs & unbiased Gxx_bar
  Z_list <- vector("list", P)
  zbar_list <- vector("list", P)
  Sigma_list <- vector("list", P)
  Gxx_bar_list <- vector("list", P)
  Omega_x_used <- vector("list", P)
  
  for (j in seq_len(P)) {
    xfd <- xfd_list[[j]]
    Cx <- xfd$coefs
    Mx <- inprod(xfd$basis, xfd$basis)
    dpX <- dp_release_coefficients(Cx, Mx, S = Sx_vec[j], s = sx_vec[j])
    Zdp <- t(dpX$Cdp)
    cz  <- center_rows(Zdp)
    Zj  <- cz$Xc
    Gxx_c <- crossprod(Zj) / N
    Gxx_bar <- (N/(N-1)) * Gxx_c - dpX$Sigma
    
    Z_list[[j]]       <- Zj
    zbar_list[[j]]    <- cz$mean
    Sigma_list[[j]]   <- dpX$Sigma
    Gxx_bar_list[[j]] <- as.matrix(.symmetrize(Gxx_bar))
    Omega_x_used[[j]] <- if (is.null(Omega_x_list) || is.null(Omega_x_list[[j]])) {
      penalty_matrix(xfd$basis, Lfd_x)
    } else .symmetrize(Omega_x_list[[j]])
  }
  
  if (is.null(Omega_y)) Omega_y <- penalty_matrix(ybasis, Lfd_y)
  Omega_y <- .symmetrize(Omega_y)
  
  # Init
  B_list <- lapply(Z_list, function(Z) matrix(0, ncol(Z), Qy))
  Fhat <- matrix(0, N, Qy)              # centered fit
  selected <- integer(0)
  sse_trace <- numeric(0); aic_trace <- numeric(0); df_trace <- numeric(0)
  
  # A/B split used for selection and CV stopping
  idx <- sample.int(N)
  A <- idx[seq_len(floor(N/2))]
  B <- setdiff(seq_len(N), A)
  
  Fhat_val_B <- matrix(0, length(B), Qy)   # predict B using increments fitted on A
  Fhat_val_A <- matrix(0, length(A), Qy)   # predict A using increments fitted on B
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
      df_trace <<- c(df_trace, df_hat)
      val
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
      df_trace <<- c(df_trace, df_hat)
      val
    }
  }
  
  for (m in seq_len(max_steps)) {
    R <- Yc - Fhat
    
    # Per-party local solves (full, and per-fold A and B)
    scores <- rep(NA_real_, P)
    dB_list  <- vector("list", P)
    dBA_list <- vector("list", P)
    dBB_list <- vector("list", P)
    
    for (j in seq_len(P)) {
      Zj <- Z_list[[j]]; Bj <- B_list[[j]]
      Omega_xj <- Omega_x_used[[j]]; Gxx_bar_j <- Gxx_bar_list[[j]]
      
      # Full fit increment (for training update)
      Rm_j <- R + Zj %*% Bj
      GxRj <- (N/(N-1)) * (crossprod(Zj, Rm_j) / N)
      dB <- solve_penalized_fof(Gxx_bar_j, GxRj, Omega_xj, Omega_y,
                                lambda_s, lambda_t, lambda_st,
                                stabilize = list(alpha = 0.05, ridge = 1e-6))
      dB_list[[j]] <- dB
      
      # Fold A → B
      RmA_j <- R[A, , drop = FALSE] + Zj[A, , drop = FALSE] %*% Bj
      GxRjA <- ((length(A))/(length(A)-1)) * (crossprod(Zj[A, , drop = FALSE], RmA_j) / length(A))
      dBA <- solve_penalized_fof(Gxx_bar_j, GxRjA, Omega_xj, Omega_y,
                                 lambda_s, lambda_t, lambda_st,
                                 stabilize = list(alpha = 0.05, ridge = 1e-6))
      dBA_list[[j]] <- dBA
      
      # Fold B → A
      RmB_j <- R[B, , drop = FALSE] + Zj[B, , drop = FALSE] %*% Bj
      GxRjB <- ((length(B))/(length(B)-1)) * (crossprod(Zj[B, , drop = FALSE], RmB_j) / length(B))
      dBB <- solve_penalized_fof(Gxx_bar_j, GxRjB, Omega_xj, Omega_y,
                                 lambda_s, lambda_t, lambda_st,
                                 stabilize = list(alpha = 0.05, ridge = 1e-6))
      dBB_list[[j]] <- dBB
      
      # DP-aware selection score (sum of crossfold scores)
      scoreAB <- selection_score_corr(Zj[B, , drop = FALSE],
                                      R[B, , drop = FALSE] + Zj[B, , drop = FALSE] %*% Bj,
                                      dBA, Sigma_list[[j]], length(B))
      scoreBA <- selection_score_corr(Zj[A, , drop = FALSE],
                                      R[A, , drop = FALSE] + Zj[A, , drop = FALSE] %*% Bj,
                                      dBB, Sigma_list[[j]], length(A))
      scores[j] <- scoreAB + scoreBA
    }
    
    j_star <- which.min(scores)
    selected <- c(selected, j_star)
    
    # Training update
    dB <- dB_list[[j_star]]
    B_list[[j_star]] <- B_list[[j_star]] + nu * dB
    Fhat <- Fhat + nu * (Z_list[[j_star]] %*% dB)
    
    # CV accumulators
    Fhat_val_B <- Fhat_val_B + nu * (Z_list[[j_star]][B, , drop=FALSE] %*% dBA_list[[j_star]])
    Fhat_val_A <- Fhat_val_A + nu * (Z_list[[j_star]][A, , drop=FALSE] %*% dBB_list[[j_star]])
    
    # Stopping
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
    } else { # "aic_train"
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
  
  # Revert to best state
  if (!is.null(best_state)) {
    B_list   <- best_state$B_list
    Fhat     <- best_state$Fhat
    selected <- best_state$selected
    sse_trace <- best_state$sse_trace
    aic_trace <- best_state$aic_trace
  }
  
  # Back to original (uncentered) coefficient space
  Yhat_orig <- sweep(Fhat, 2, ybar, FUN = "+")
  yhat_fd <- fd(coef = t(Yhat_orig), basisobj = ybasis, fdnames = yfd$fdnames)
  
  list(
    B_list = B_list, selected = selected,
    yhat_fd = yhat_fd,
    centers = list(ybar = ybar, zbar_list = zbar_list),
    Sigma_list = Sigma_list,
    Omega_x_list = Omega_x_used, Omega_y = Omega_y,
    Z_list = Z_list, Yc = Yc,
    traces = list(sse = sse_trace, aic = aic_trace),
    stop_mode = stop_mode
  )
}

predict_vfl_dp_foboost <- function(fit, xfd_list_new) {
  P <- length(fit$B_list)
  stopifnot(length(xfd_list_new) == P)
  Znew_list <- vector("list", P)
  for (j in seq_len(P)) {
    Cx_new <- xfd_list_new[[j]]$coefs
    Znew <- t(Cx_new)
    Znew_list[[j]] <- sweep(Znew, 2, fit$centers$zbar_list[[j]], FUN = "-")
  }
  Nnew <- nrow(Znew_list[[1]])
  Qy <- nrow(fit$yhat_fd$coefs)
  Yhat <- matrix(0, Nnew, Qy)
  for (j in seq_len(P)) {
    Yhat <- Yhat + Znew_list[[j]] %*% fit$B_list[[j]]
  }
  Yhat <- sweep(Yhat, 2, fit$centers$ybar, FUN = "+")
  fd(coef = t(Yhat), basisobj = fit$yhat_fd$basis, fdnames = fit$yhat_fd$fdnames)
}

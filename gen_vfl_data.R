# ============================================================
# Stable VFL data generator (orthonormalized coefficients, controlled SNR)
# VFL semantics: same individuals (N_global) for all #workers k.
# Each worker owns a disjoint subset of predictors; non-owned = NULL.
#
# Writes (for (l, k, hh)):
#   yfdobj_<l>_<k>_<hh>.RData
#   predictorLst_<l>_<k>_<hh>.RData   (length p; NULL for non-owned)
# Also writes:
#   truth_active_idx.RData
# ============================================================

root_dir <- "/Users/yuding/Dropbox/VFL_code"
setwd(root_dir)

suppressPackageStartupMessages({
  library(fda)
})

# ---- helpers ----
sym_eigen_sqrt <- function(M, ridge = 1e-10) {
  ee <- eigen((M + t(M))/2, symmetric = TRUE)
  lam <- pmax(ee$values, ridge); U <- ee$vectors
  list(half = U %*% diag(sqrt(lam)) %*% t(U),
       invhalf = U %*% diag(1/sqrt(lam)) %*% t(U))
}

owner_of_feature <- function(j, numworkers) {
  ((j - 1) %% numworkers) + 1L
}

# ---- settings ----
set.seed(2025)
rangeval <- c(0, 100)
Qx <- Qy <- 20
basisobj <- create.bspline.basis(rangeval, nbasis = Qx)

N_global         <- 500
numworkersseq    <- c(2, 4, 6, 8, 10)
num_duplicate    <- 20

p <- 20
active_idx <- 1:5
rank_r <- 3
a_active <- 1.0
SNR_target <- 10

# Gram matrices and (inverse) square roots
Mx <- inprod(basisobj, basisobj)
My <- inprod(basisobj, basisobj)
sym_eigen_sqrt_mat <- function(M) {
  ee <- eigen((M + t(M))/2, symmetric = TRUE)
  lam <- pmax(ee$values, 1e-12); U <- ee$vectors
  list(half = U %*% diag(sqrt(lam)) %*% t(U),
       invhalf = U %*% diag(1/sqrt(lam)) %*% t(U))
}
MxS <- sym_eigen_sqrt_mat(Mx); MyS <- sym_eigen_sqrt_mat(My)

# True whitened operators B_w_j
set.seed(1001)
make_Bw <- function(j) {
  if (!(j %in% active_idx)) return(matrix(0, Qx, Qy))
  U <- qr.Q(qr(matrix(rnorm(Qx * rank_r), Qx, rank_r)))
  V <- qr.Q(qr(matrix(rnorm(Qy * rank_r), Qy, rank_r)))
  s <- a_active * c(1.0, 0.7, 0.4)[seq_len(rank_r)]
  Bw <- U %*% diag(s, nrow = rank_r) %*% t(V)
  Bw / max(1e-12, norm(Bw, "F")) * a_active * sqrt(rank_r)
}
B_w_list <- lapply(seq_len(p), make_Bw)

# ---- main loop ----
for (hh in seq_len(num_duplicate)) {
  message(sprintf("==== Replicate %d ====", hh))
  for (k in numworkersseq) {
    
    # Predictor coefficients (whitened): Z_w_j ~ N(0, I)
    Z_w_list <- lapply(seq_len(p), function(j) matrix(rnorm(Qx * N_global), Qx, N_global))
    
    # Whitened signal in Y
    Yw_signal <- matrix(0, Qy, N_global)
    for (j in seq_len(p)) {
      Yw_signal <- Yw_signal + t(B_w_list[[j]]) %*% Z_w_list[[j]]
    }
    
    # Set noise variance to hit target SNR
    SigY <- cov(t(Yw_signal)); trSignal <- sum(diag(SigY))
    sigma_w <- sqrt(trSignal / (Qy * SNR_target + 1e-12))
    Ew <- matrix(rnorm(Qy * N_global, sd = sigma_w), Qy, N_global)
    Yw <- Yw_signal + Ew
    
    # Back to coefficient space
    Cx_list <- lapply(seq_len(p), function(j) MxS$invhalf %*% Z_w_list[[j]])
    Cy <- MyS$invhalf %*% Yw
    yfdobj_global <- fd(coef = Cy, basisobj = basisobj)
    
    # Per-worker files
    for (l in seq_len(k)) {
      
      # Initialize with NULL placeholders (length p).
      pred_list <- vector("list", p)
      
      for (j in seq_len(p)) {
        if (owner_of_feature(j, k) == l) {
          pred_list[[j]] <- fd(coef = Cx_list[[j]], basisobj = basisobj)
        }
        # else: leave as NULL; DO NOT do pred_list[[j]] <- NULL
      }
      
      # Optional sanity: owned features must be non-NULL; non-owned must be NULL
      for (j in seq_len(p)) {
        owns <- (owner_of_feature(j, k) == l)
        if (owns && is.null(pred_list[[j]]))
          stop(sprintf("[generator] worker %d should own feature %d but it's NULL (k=%d, hh=%d)",
                       l, j, k, hh))
        if (!owns && !is.null(pred_list[[j]]))
          stop(sprintf("[generator] worker %d should NOT own feature %d but it's non-NULL (k=%d, hh=%d)",
                       l, j, k, hh))
      }
      
      yfdobj <- yfdobj_global
      save(yfdobj,      file = sprintf("yfdobj_%d_%d_%d.RData", l, k, hh))
      predictorLst <- pred_list
      # FIX: use '==' instead of 'identical(...)'
      stopifnot(length(predictorLst) == p)
      save(predictorLst, file = sprintf("predictorLst_%d_%d_%d.RData", l, k, hh))
    }
  }
}
save(active_idx, file = "truth_active_idx.RData")
message("Data generation complete.")

# ============================================================
# Stable VFL data generator (orthonormalized coefficients, controlled SNR)
# Writes, for each replicate hh:
#   yfdobj_<hh>.RData
#   predictorLst_<hh>.RData
# Also writes:
#   truth_active_idx.RData
# ============================================================

setwd("/Users/yuding/Dropbox/VFL_code")

suppressPackageStartupMessages({
  library(fda)
})

# ---- helpers ----
sym_eigen_sqrt <- function(M, ridge = 1e-10) {
  M <- (M + t(M)) / 2
  ee <- eigen(M, symmetric = TRUE)
  lam <- pmax(ee$values, ridge)
  U <- ee$vectors
  list(
    half    = U %*% diag(sqrt(lam), nrow = length(lam)) %*% t(U),
    invhalf = U %*% diag(1 / sqrt(lam), nrow = length(lam)) %*% t(U)
  )
}

# ---- settings ----
set.seed(2025)
rangeval <- c(0, 100)
Qx <- Qy <- 20
basisobj <- create.bspline.basis(rangeval, nbasis = Qx)

N_total        <- 1000   # <-- constant N across all party counts
num_duplicate  <- 20

p <- 20
active_idx <- 1:5        # true sparse set
rank_r <- 3
a_active <- 1.0          # operator magnitude (whitened)
SNR_target <- 10         # signal-to-noise ratio in whitened Y

# Gram matrices and square roots
Mx <- inprod(basisobj, basisobj)
My <- inprod(basisobj, basisobj)
MxS <- sym_eigen_sqrt(Mx)
MyS <- sym_eigen_sqrt(My)

# Build true B^(w)_j (whitened operators)
set.seed(1001)
make_Bw <- function(j) {
  if (!(j %in% active_idx)) return(matrix(0, Qx, Qy))
  U <- qr.Q(qr(matrix(rnorm(Qx * rank_r), Qx, rank_r)))
  V <- qr.Q(qr(matrix(rnorm(Qy * rank_r), Qy, rank_r)))
  s <- a_active * c(1.0, 0.7, 0.4)
  Bw <- U %*% diag(s, nrow = rank_r) %*% t(V)
  Bw / max(1e-12, norm(Bw, "F")) * a_active * sqrt(rank_r)
}
B_w_list <- lapply(1:p, make_Bw)

# ---- main loop ----
for (hh in 1:num_duplicate) {
  message(sprintf("==== Replicate %d ====", hh))
  
  # Draw whitened predictor coefficients Z_w_j ~ N(0, I)
  Z_w_list <- lapply(1:p, function(j) matrix(rnorm(Qx * N_total), Qx, N_total))
  
  # Signal in whitened Y: Yw_signal = sum_j t(Bw_j) %*% Z_w_j
  Yw_signal <- matrix(0, Qy, N_total)
  for (j in 1:p) {
    Yw_signal <- Yw_signal + t(B_w_list[[j]]) %*% Z_w_list[[j]]
  }
  
  # Choose noise level to match SNR_target
  SigY <- stats::cov(t(Yw_signal))    # Qy x Qy
  trSignal <- sum(diag(SigY))
  sigma_w <- sqrt(trSignal / (Qy * SNR_target + 1e-12))
  
  # Add Gaussian noise in whitened space
  Ew <- matrix(rnorm(Qy * N_total, sd = sigma_w), Qy, N_total)
  Yw <- Yw_signal + Ew
  
  # Map whitened coefficients back to fd coefficient space:
  #   X: Cx = Mx^{-1/2} Z_w ;  Y: Cy = My^{-1/2} Yw
  predictorLst <- vector("list", p)
  for (j in 1:p) {
    Cx <- MxS$invhalf %*% Z_w_list[[j]]
    predictorLst[[j]] <- fd(coef = Cx, basisobj = basisobj)
  }
  Cy <- MyS$invhalf %*% Yw
  yfdobj <- fd(coef = Cy, basisobj = basisobj)
  
  save(yfdobj,       file = sprintf("yfdobj_%d.RData", hh))
  save(predictorLst, file = sprintf("predictorLst_%d.RData", hh))
}

save(active_idx, file = "truth_active_idx.RData")
message("Data generation complete.")

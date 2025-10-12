# Stable generator (one predictor per passive party; no worker dimension)
suppressPackageStartupMessages(library(fda))

set.seed(2025)

# ----- settings -----
rangeval <- c(0, 100)
Qx <- Qy <- 20
basisobj <- create.bspline.basis(rangeval, nbasis = Qx)

N_global      <- 500
num_duplicate <- 20
p             <- 20              # number of parties = number of predictors
active_idx    <- 1:5
rank_r        <- 3
a_active      <- 1.0
SNR_target    <- 10

# Grams and roots
Mx <- inprod(basisobj, basisobj)
My <- inprod(basisobj, basisobj)
sym_sqrt <- function(M) {
  ee <- eigen((M + t(M))/2, symmetric = TRUE)
  lam <- pmax(ee$values, 1e-12); U <- ee$vectors
  list(half = U %*% diag(sqrt(lam)) %*% t(U),
       invhalf = U %*% diag(1/sqrt(lam)) %*% t(U))
}
MxS <- sym_sqrt(Mx); MyS <- sym_sqrt(My)

# True operators in whitened coords
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

for (hh in seq_len(num_duplicate)) {
  message(sprintf("==== Replicate %d ====", hh))
  
  # Predictors in whitened space (per party)
  Z_w_list <- lapply(seq_len(p), function(j) matrix(rnorm(Qx * N_global), Qx, N_global))
  
  # Whitened signal in Y
  Yw_signal <- matrix(0, Qy, N_global)
  for (j in seq_len(p)) Yw_signal <- Yw_signal + t(B_w_list[[j]]) %*% Z_w_list[[j]]
  
  # Add noise at target SNR
  SigY <- cov(t(Yw_signal)); trSignal <- sum(diag(SigY))
  sigma_w <- sqrt(trSignal / (Qy * SNR_target + 1e-12))
  Ew <- matrix(rnorm(Qy * N_global, sd = sigma_w), Qy, N_global)
  Yw <- Yw_signal + Ew
  
  # Back to coefficient space
  Cx_list <- lapply(seq_len(p), function(j) MxS$half %*% Z_w_list[[j]]  ) # α = M^{1/2} z
  Cy      <- MyS$half %*% Yw
  yfdobj  <- fd(coef = Cy, basisobj = basisobj)
  
  # Save one y and one list of x (length p, each an fd)
  predictorList <- lapply(seq_len(p), function(j) fd(coef = Cx_list[[j]], basisobj = basisobj))
  save(yfdobj,        file = sprintf("yfdobj_%d.RData", hh))
  save(predictorList, file = sprintf("predictorList_%d.RData", hh))
}
save(active_idx, file = "truth_active_idx.RData")
message("Data generation complete.")

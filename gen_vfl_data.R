suppressPackageStartupMessages(library(fda))
source("functions.R")  # for predict_linmod()

# ---------------- Settings (fixed) ----------------
set.seed(2025)
rangeval <- c(0, 100)
t <- 20                          # #basis
basisobj <- create.bspline.basis(rangeval, nbasis = t)

n <- 6250                         # subjects
p <- 4                          # predictors / parties
active_idx <- 1:2                # which predictors truly drive Y
num_duplicate <- 20              # replicates to write

# noise & ease controls (very easy)
sigma_pred <- 0.005              # tiny predictor noise on the grid
sigma_y    <- 0.005              # tiny response noise in coef space
gain_active <- 1.0               # strength of each active predictor contribution

# ---------------- Helpers ----------------
############################################################################
predict_linmod =function(linmodres, newdata = NULL){
  if(is.null(newdata)){
    return (linmodres$yhatfdobj)
  }
  xbasis = newdata$basis
  xnbasis = xbasis$nbasis
  ranget = xbasis$rangeval
  coefx = newdata$coefs
  coefdx = dim(coefx)
  ncurves = coefdx[2]
  
  nfine = max(201, 10 * xnbasis  + 1)
  tfine = seq(ranget[1], ranget[2], len = nfine)
  
  alphafd = linmodres$beta0estfd
  betasbasis = linmodres$beta1estbifd$sbasis
  Hinprod = inprod(xbasis, betasbasis)
  xcoef = coefx
  Hmat = t(xcoef) %*% Hinprod
  betacoef = t(linmodres$beta1estbifd$coefs)
  xbetacoef = betacoef %*% t(Hmat)
  xbetafd = fd(xbetacoef, linmodres$beta1estbifd$tbasis)
  yhatmat = eval.fd(tfine, alphafd) %*% matrix(1, 1, ncurves) + 
    eval.fd(tfine, xbetafd)
  res = smooth.basis(tfine, yhatmat, xbasis)$fd
  
  return(res)
}
############################################################################
# Hann window on [a,b] (smooth, compact support)
hann <- function(tgrid, a, b) {
  x <- (tgrid - a) / (b - a)
  y <- numeric(length(tgrid))
  idx <- which(x >= 0 & x <= 1)
  y[idx] <- 0.5 * (1 - cos(2*pi*x[idx]))
  y
}

make_grid <- function(rangeval, n = 101) {
  tvec <- seq(rangeval[1], rangeval[2], length.out = n)
  list(t = tvec)
}

G <- make_grid(rangeval, n = 101)
time_vec <- G$t  # 0:100 (101 pts)

# Build p non-overlapping windows across [0,100] with a small gap for near-orthogonality
edges  <- seq(rangeval[1], rangeval[2], length.out = p + 1)
shrink <- 0.85  # shrink each bin to leave gaps
phi_mat <- matrix(0, nrow = length(time_vec), ncol = p)  # |grid| x p
for (j in seq_len(p)) {
  a <- edges[j]; b <- edges[j + 1]
  len <- b - a
  mid <- (a + b)/2
  a2  <- mid - (shrink * len)/2
  b2  <- mid + (shrink * len)/2
  phi_mat[, j] <- hann(time_vec, a2, b2)
}

# ---------------- Fixed bi-basis linear operator (identity; very easy) ----------------
# We’ll use predict_linmod(lin, x_fd) to map each predictor to Y,
# with an identity B-matrix so Y basically sums active predictors.
Bmat <- diag(t)                                  # t x t (identity)
bbspl_s <- create.bspline.basis(rangeval, t)
bbspl_t <- create.bspline.basis(rangeval, t)
betafdnames <- list("Time_s", "Time_t", "Reg. Coefficient")
lin <- list(
  beta0estfd   = fd(matrix(0, t, 1), basisobj),  # zero intercept
  beta1estbifd = bifd(Bmat, bbspl_s, bbspl_t, betafdnames),
  yhatfdobj    = 0
)

# ============================ Generate & Save ============================
for (hh in seq_len(num_duplicate)) {
  message(sprintf("==== Replicate %d ====", hh))
  
  # (1) Create predictors on the grid: subject-specific amplitude × pattern + tiny noise
  # predictors[grid, i, m]
  predictors <- array(0, dim = c(length(time_vec), p, n))
  amplitudes <- matrix(rnorm(p * n, sd = 1), nrow = p, ncol = n)  # N(0,1) amps
  for (i in seq_len(p)) {
    for (m in seq_len(n)) {
      predictors[, i, m] <- amplitudes[i, m] * phi_mat[, i] +
        rnorm(length(time_vec), sd = sigma_pred)
    }
  }
  
  # (2) Smooth each predictor into an fd with the shared B-spline basis
  predictorList <- vector("list", p)
  for (i in seq_len(p)) {
    # smooth.basis: x = time_vec, y = matrix of length(time_vec) x n
    predictorList[[i]] <- smooth.basis(time_vec, predictors[, i, ], basisobj)$fd
  }
  
  # (3) Build Y as sum of active predictors passed through the (easy) operator
  yfdobj <- fd(matrix(0, t, n), basisobj)
  for (i in active_idx) {
    # scale active contribution by gain_active (can increase above 1 for even easier data)
    yfdobj <- yfdobj + gain_active * predict_linmod(lin, predictorList[[i]])
  }
  # tiny coefficient-space noise for Y
  yfdobj <- yfdobj + fd(matrix(rnorm(t * n, sd = sigma_y), t, n), basisobj)
  
  # (4) Save in the current format expected by your new pipelines
  save(yfdobj,        file = sprintf("yfdobj_%d.RData", hh))
  save(predictorList, file = sprintf("predictorList_%d.RData", hh))
}

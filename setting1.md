# Simulation Setting 1 — Detailed Data-Generation Guide

This document explains exactly how **Setting 1** builds the synthetic vertical federated functional dataset used in your benchmarks. It covers the signal model, notation and shapes, whitening and Gram geometry, operator construction, SNR calibration, per-party file outputs, and recommended sanity checks. Everything here matches the generator you’ve been running (B-spline basis, low-rank function-on-function operator, controlled SNR, one functional predictor per party or features split across parties).

---

## 1) Problem geometry and notation

**Subjects and parties.**

* Number of subjects: $N=N_{\text{global}}$ (e.g., 500).
* Number of functional predictors (also “parties” in VFL): (p) (e.g., 20).
* Option A (multi-worker): there are (k) workers; each worker $l\in{1,\dots,k}$ “owns” a disjoint subset of the (p) predictors; non-owned predictors at that worker are `NULL`.
* Option B (one predictor per party): $p=K$, i.e., one passive per predictor.

**Time domain and bases.**

* Domain $\mathcal T = [0,100]$.
* B-spline basis for predictors and response: $Q_x=Q_y=t_{{\rm basis}}$ (e.g., 20).
* Gram matrices:
  $
  M_x=\langle \phi_x,\phi_x\rangle \in\mathbb R^{Q_x\times Q_x},\quad
  M_y=\langle \phi_y,\phi_y\rangle \in\mathbb R^{Q_y\times Q_y}.
  $

**fd objects.**

* For any functional signal (f), its `fd` representation is `coef` $\in\mathbb R^{Q\times N}$ plus the B-spline basis.
* Predictors: $X_j$ has coefficients $C_{xj}\in\mathbb R^{Q_x\times N}$.
* Response: $Y$ has coefficients $C_y\in\mathbb R^{Q_y\times N}$.

**Whitened coordinates (crucial).**

* Let $M_x^{1/2},M_x^{-1/2}$ and $M_y^{1/2},M_y^{-1/2}$ be symmetric (inverse) square roots from an eigen-decomposition with a small ridge (e.g., (10^{-12})).
* **Whitened coefficients:** $Z_j = M_x^{-1/2}C_{xj}\in\mathbb R^{Q_x\times N}$, $Y_w=M_y^{-1/2}C_y\in\mathbb R^{Q_y\times N}$.
* In these coordinates, RKHS norms reduce to Euclidean norms and the Gaussian noise you add is **isotropic**.

---

## 2) Data-generation model

### 2.1 Predictors in whitened space

For each predictor $j=1,\dots,p$ you draw i.i.d. whitened coefficients:
$
Z_j \sim \mathcal N(0, I_{Q_x})\ \text{columnwise},\quad Z_j\in\mathbb R^{Q_x\times N}.
$
You then **map back to coefficient space**:
$
C_{xj} = M_x^{1/2} Z_j\ \in\ \mathbb R^{Q_x\times N}.
$
These become `fd(coef=C_{xj}, basis=basis_x)`.

### 2.2 Low-rank function-on-function (FoF) operators

You create a set of ground-truth linear operators ({B_j}_{j=1}^p) acting from $X_j$ to $Y$ **in whitened coordinates**:
$
B_j \in \mathbb R^{Q_x\times Q_y},\qquad B_j = U_j,{\rm diag}(s),V^\top,
$
with rank (r) (e.g., 3). Construction details:

* Draw $U_j\in\mathbb R^{Q_x\times r}$ and $V\in\mathbb R^{Q_y\times r}$ with columns orthonormalized (e.g., via QR).
* Choose singular values $s=(s_1,\dots,s_r)$ (e.g., geometric decay (1.0, 0.7, 0.4)) and scale by a global gain $a_{\rm active}$ (e.g., 1.0).
* Optionally **sparsify** contributions by setting $B_j=0$ for $j\notin active_idx$ (e.g., only 5 predictors are truly predictive); store `active_idx` on disk for verification.

> Tip: normalize $B_j$ to a target Frobenius norm (or to $\sqrt r$) to standardize signal strength across replicates.

### 2.3 Response in whitened space and SNR control

Compute the **signal component** of the response in whitened coordinates:
$
Y_w^{\rm signal} = \sum_{j=1}^p B_j^\top Z_j\ \in\ \mathbb R^{Q_y\times N}.
$
To hit a target SNR, add isotropic Gaussian noise $E_w\sim \mathcal N(0,\sigma_w^2 I)$:

1. Estimate signal power via the empirical covariance of $Y_w^{\rm signal}$:
  $\ \Sigma_{\rm sig}=\mathrm{cov}(Y_w^{\rm signal}{}^\top)$, use (\mathrm{tr}(\Sigma_{\rm sig})).
2. Given target $\mathrm{SNR}=\frac{\text{signal power}}{\text{noise power}}$, choose
   $
   \sigma_w ;\leftarrow;\sqrt{\frac{\mathrm{tr}(\Sigma_{\rm sig})}{Q_y\cdot \mathrm{SNR}}}.
   $
3. Set $Y_w = Y_w^{\rm signal} + E_w$.

Finally, map back to coefficient space for the response:
$
C_y = M_y^{1/2} Y_w\ \in\ \mathbb R^{Q_y\times N},
$
and store `yfdobj = fd(coef=C_y, basis=basis_y)`.

---

## 3) File layout and VFL semantics

### 3.1 Multi-worker layout (original Setting 1)

For each **replicate** $h=1,\dots,H$ and each **worker** $l=1,\dots,k$:

* `yfdobj_l_k_hh.RData` — the **same** response `fd` at all workers (active holds (Y)).
* `predictorLst_l_k_hh.RData` — a list of length (p):

  * slot $j$ is `fd(coef=C_{xj}, basis=basis_x)` if worker $l$ **owns** predictor $j$; otherwise `NULL`.
* `truth_active_idx.RData` — indices of genuinely predictive parties (where $B_j\neq 0$).

Ownership is typically round-robin:
$
\text{owner}(j,k)= ((j-1)\bmod k)+1.
$

### 3.2 One-party-per-predictor layout (simplified variant)

For each **replicate** $h$:

* `yfdobj_hh.RData` — the response `fd`.
* `predictorList_hh.RData` — list of length (p), **no NULLs** (each party has exactly one predictor).
* `truth_active_idx.RData` — same as above.

This variant matches “one predictor per passive party” used in your strict paper-aligned S2DP-FGB experiments.

---

## 4) Shapes at a glance

* $Z_j \in \mathbb R^{Q_x\times N}$ (whitened predictor coefficients)
* $B_j \in \mathbb R^{Q_x\times Q_y}$ (whitened operator)
* $Y_w^{\rm signal} \in \mathbb R^{Q_y\times N}), (E_w \in \mathbb R^{Q_y\times N}$
* $C_{xj}=M_x^{1/2}Z_j\in \mathbb R^{Q_x\times N}), (C_y=M_y^{1/2}Y_w\in \mathbb R^{Q_y\times N}$

---

## 5) Key configuration knobs

* **Basis and grid:** `rangeval=c(0,100)`, `t_basis=Qx=Qy=20`, grid for evaluation: `seq(0,100,by=1)`.
* **Sample sizes:** `N_global` (e.g., 500), `num_duplicate` (replicates), number of parties `p` (e.g., 20), number of workers `k` (if using multi-worker).
* **Signal complexity:** operator rank `r` (e.g., 3), singular values `s`, active set `active_idx`.
* **SNR:** `SNR_target` (e.g., 10).
* **Randomness:** set seeds **per replicate** to ensure reproducibility of (Z_j), ({B_j}), and noise.

---

## 6) Why whitening matters (and how it’s used)

* Clipping, sensitivity, and noise accounting in later DP steps (S2DP-FGB or Roundwise-DP) all become **isotropic** and easy in whitened space.
* Numerically, $M^{-1/2}$ also stabilizes the Sylvester/Kronecker solves used later in fitting.
* In Setting 1, whitened space is **only a construction device**: the saved `fd` objects are in **coefficient space**; your training pipelines re-enter whitened coordinates as needed.

---

## 7) Sanity checks (recommended)

Run these after each replicate to ensure correctness:

1. **Basis compatibility**

   * Check `nbasis(x_j) == nbasis(y)` for all (j), and `ncol(coef)` equals (N).

2. **Power and SNR**

   * Compute $\widehat{\mathrm{SNR}} = \frac{|!Y_w^{\rm signal}!|_F^2/Q_y}{|E_w|_F^2/Q_y}$ from the generator to confirm it matches the target (allow small sampling variance).

3. **Active set**

   * Verify that predictive features in `active_idx` contribute non-zero operator $B_j$ and others are (near) zero.

4. **Ownership (multi-worker)**

   * For each worker $l$, assert `predictorLst[[j]]` is `NULL` iff `owner(j)!=l`.

5. **Reproducibility**

   * Fix seed(s) and ensure identical outputs across runs (hash saved `.RData` files).

---

## 8) How Setting 1 aligns with the two training methods

* **S2DP-FGB (one-shot DP):**
  Setting 1 is ideal: you release **one** DP object per party $Z_j^{dp}$ (whitened, clipped, isotropic noise), subtract $\Sigma_x$ in moments, and proceed with penalized FoF and AIC stopping.

* **Roundwise-DP:**
  Same dataset. At each boosting round, the active broadcasts a **DP residual** in **whitened-(Y)**, passives fit local increments, and the selected party returns a **DP increment** (either a prediction vector or an operator update). Setting 1’s low-rank operators ensure a meaningful residual descent path.

---

## 9) Practical tips and edge cases

* **Eigen decompositions:** use a tiny ridge (e.g., $10^{-12}$) before taking $M^{\pm 1/2}$ to avoid negative or near-zero eigenvalues from numeric noise.
* **Operator scaling:** if you change (p,r) or `active_idx`, re-tune the singular values or the global gain so that the pre-noise response amplitude stays in a stable range.
* **SNR extremes:** very high SNR can make cross-validation overly optimistic; very low SNR can flatten AIC differences—pick SNRs that span your target (\varepsilon) range without saturating the metrics.

---

## 10) What the generator writes (filenames)

**Multi-worker (original Setting 1)**

* `yfdobj_<l>_<k>_<hh>.RData` — response `fd` (same across $l$).
* `predictorLst_<l>_<k>_<hh>.RData` — length-(p) list of `fd` or `NULL`.
* `truth_active_idx.RData`.

**One-party-per-predictor (simplified variant)**

* `yfdobj_<hh>.RData` — response `fd`.
* `predictorList_<hh>.RData` — length-(p) list of `fd`.
* `truth_active_idx.RData`.

---

## 11) Minimal pseudo-code (centralized view)

```r
# Inputs: N, p, Qx, Qy, rank r, SNR_target, active_idx
Mx <- inprod(basis_x, basis_x); My <- inprod(basis_y, basis_y)
MxS <- sym_eigen_sqrt(Mx); MyS <- sym_eigen_sqrt(My)

# 1) Predictors (whitened -> coeff)
Z_w_list <- lapply(1:p, function(j) matrix(rnorm(Qx * N), Qx, N))
Cx_list  <- lapply(1:p, function(j) MxS$half %*% Z_w_list[[j]])

# 2) Operators in whitened coords
B_w_list <- lapply(1:p, function(j) {
  if (j %in% active_idx) low_rank_U_diagS_Vt(Qx, Qy, r) else matrix(0, Qx, Qy)
})

# 3) Response in whitened coords + SNR noise
Yw_signal <- Reduce(`+`, lapply(1:p, function(j) t(B_w_list[[j]]) %*% Z_w_list[[j]]))
sigma_w <- sqrt( trace(cov(t(Yw_signal))) / (Qy * SNR_target) )
Ew <- matrix(rnorm(Qy * N, sd = sigma_w), Qy, N)
Yw <- Yw_signal + Ew

# 4) Map response back to coefficients and save fd objects
Cy <- MyS$half %*% Yw
yfdobj <- fd(coef = Cy, basisobj = basis_y)
predictorList <- lapply(1:p, function(j) fd(coef = Cx_list[[j]], basisobj = basis_x))
```

---

## 12) Summary

Setting 1 produces **clean, controllable** functional data for VFL:

* Predictors: i.i.d. **whitened Gaussian** columns mapped to B-spline coefficients.
* Response: sum of **low-rank FoF** contributions plus isotropic **Gaussian noise** tuned to an explicit **SNR**.
* Files: organized for VFL experiments (multi-worker or one-party-per-predictor).
* Geometry: whitening makes your later DP and penalized solves straightforward and stable.

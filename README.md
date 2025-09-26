# VFL–DP–FoF: Differentially‑Private Function‑on‑Function Regression in Vertical FL

This repository contains an end‑to‑end R implementation of:

1. **Differentially private penalized function‑on‑function (FoF) regression** in coefficient space, and
2. A **Vertical Federated Learning** (VFL) wrapper that performs **functional boosting** across parties with a **DP‑aware selection score** and **principled early stopping** (cross‑validated or AIC$_c$).

It includes a **stable data generator** (orthonormalized coefficients; controlled SNR), a **parallel runner** for experiments, and the **core FoF/boosting routines**.

---

## Contents

```
VFL_code/
├─ generator_vfl.R                 # data generator (VFL semantics; fixed N across workers)
├─ run_vfl_experiment_parallel.R   # parallel experiment runner
├─ functions.R                     # DP FoF + VFL functional boosting implementation
├─ vfl_dp_foboost_results.RData    # written by runner (results & arrays)
├─ truth_active_idx.RData          # ground truth active predictors for evaluation
├─ yfdobj_<l>_<k>_<hh>.RData       # response fd (identical across l for a fixed k,hh)
└─ predictorLst_<l>_<k>_<hh>.RData # per-worker predictor list (length p; NULL for non‑owners)
```

---

## Quick start

> **R version**: 4.1+ recommended
> **Packages**: `fda`, `Matrix`, `future.apply`

```r
install.packages(c("fda", "Matrix", "future.apply"))
```

1. **Generate data** (clears/overwrites matching files in the working directory):

```r
# in R
setwd("/path/to/your/dir")
source("generator_vfl.R")
```

This writes `yfdobj_*`, `predictorLst_*`, and `truth_active_idx.RData` for each replicate `hh` and worker count `k`.

2. **Run experiments in parallel**:

```r
setwd("/path/to/your/dir")
source("run_vfl_experiment_parallel.R")
```

This produces `vfl_dp_foboost_results.RData` with arrays and a per‑job data.frame (`res_df`), and prints a summary table.

---

## What the code does

### 1) VFL data generator (`generator_vfl.R`)

* **VFL semantics**: The number of **individuals** (samples) is **fixed** (`N_global`) for all worker counts $k$. Increasing $k$ only repartitions **features** (predictors) across workers; all workers observe the same individuals.
* **Orthonormalized coefficients**: Predictors $X^{(j)}$ and responses $Y$ are simulated in whitened coefficient space, then mapped back to `fd` coefficients using basis Gram matrices $M_x, M_y$ (Bsplines by default).
* **Signal & noise**: A low‑rank operator $B_j$ is created for a small set of **active** predictors (default: `active_idx <- 1:5`); whitened noise is added to reach target SNR (`SNR_target`).
* **Files written per $(l,k,hh)$**:

  * `yfdobj_<l>_<k>_<hh>.RData`: an `fd` object with coefficients $Q_y \times N$. **Identical across $l$** for a fixed $(k,hh)$.
  * `predictorLst_<l>_<k>_<hh>.RData`: a **length‑`p` list**. Entry `j` is:

    * an `fd` with the coefficients of feature $j$ **if worker $l$ owns feature $j$**,
    * `NULL` otherwise.
* **Ownership**: `owner_of_feature(j, k) = ((j-1) %% k) + 1`. Each feature belongs to exactly one worker.

> **Important invariant**: For **every** `predictorLst_<l>_<k>_<hh>.RData`,
> `length(predictorLst) == p` and each `predictorLst[[j]]` is either `fd` (if owned by worker `l`) or `NULL` (if not owned).

### 2) Parallel runner (`run_vfl_experiment_parallel.R`)

* Builds a **job table** over `(duplicate = hh) × (workers = k) × (fold)` and runs each job in parallel via `future.apply` (multisession).
* **VFL‑aware loading**:

  * Picks **one** `yfdobj` (from worker 1) — identical across workers by construction.
  * For each feature $j$, loads the **owning** worker’s `fd` object into the global `Xlist`.
* Performs **DP‑aware functional boosting** via `vfl_dp_foboost()`:

  * Per‑party **clipping** in RKHS (coefficient space) and **Gaussian noise** (covariance $s_x^2 M_x$).
  * **DP moment corrections** for centered covariance and cross‑moments.
  * At each boosting step, solves a **penalized FoF Sylvester** update for every candidate party, computes a **DP‑corrected selection score**, and updates the best party with shrinkage $\nu$.
  * **Early stopping** by **two‑fold CV** on the train split (default). AIC/AIC$_c$ option is available.
* **Metrics**:

  * pointwise: RMSE, NRMSE, sMAPE, WMAPE, MAPE
  * functional (basis‑aware): $\| \hat Y - Y \|^2_{L^2}$ averaged per curve (IL2) and its ratio to $\| Y \|^2_{L^2}$ (RIL2)
  * selection: sensitivity (TPR on `active_idx`), specificity (TNR on inactives)
  * runtime & approximate **communication cost** (MB)
* Saves **arrays** over `(k, hh, fold)` and a **summary table** aggregated over all replicates & folds.

### 3) Core DP FoF & boosting (`functions.R`)

Public API (all used by the runner):

* `dp_release_coefficients(C, M, S, s)`: per‑record clipping in metric induced by $M$ and addition of $\mathcal{N}(0, s^2 M)$ noise.
* `form_dp_moments(Zdp, Ycoef, Sigma_x, ...)`: centered sample covariance/cross‑covariance with the finite‑sample DP corrections $\frac{N}{N-1}$ and subtraction of $ \Sigma_x$.
* `penalty_matrix(basis, Lfd)`: roughness penalty $ \int (D^m \phi_i)(D^m \phi_j)$.
* `solve_penalized_fof(Gxx, Gxy, Omega_x, Omega_y, lambda_s, lambda_t, lambda_st, ...)`: generalized Sylvester/Kronecker solve for $B$.
* `vfl_dp_foboost(...)`: VFL boosting with per‑party DP designs, DP‑corrected selection, CV stopping (or AIC/AIC$_c$), and Hutchinson df estimator (for AIC paths).
* `predict_vfl_dp_foboost(fit, Xlist_new)`: prediction on new `fd` lists.

---

## Configuration knobs (where to tune)

### Data generator (`generator_vfl.R`)

* `N_global` — number of individuals (fixed across workers).
* `p` — number of features/predictors.
* `active_idx` — indices of truly active predictors (default: `1:5`).
* `rank_r`, `a_active` — rank/strength of true operators $B_j$ (in whitened space).
* `SNR_target` — response SNR in whitened coordinates.
* `basisobj` — basis family and size (default: B‑splines, `nbasis=20`).

### Runner (`run_vfl_experiment_parallel.R`)

* `numworkersseq` — vector of worker counts to evaluate.
* `num_duplicate`, `folds_per_worker` — repetitions and CV folds (on the **fixed** `N_global`).
* **DP**:

  * `Sx_mode` — `"empirical"` (simulation only; **not** DP‑accounted) or `"fixed"` (use `Sx_fixed`).
  * `sx_default` — per‑record noise std in whitened space; for DP, calibrate to $(\varepsilon,\delta)$ or zCDP $\rho$.
* **FoF penalties & boosting**:

  * `lambda_s`, `lambda_t`, `lambda_st` — penalties on predictor/response roughness and interaction.
  * `nu` — shrinkage (e.g., `0.3`).
  * `max_steps`, `min_steps`, `patience` — boosting path length and early‑stopping controls.
  * `use_crossfit=TRUE`, `stop_mode="cv"` — two‑fold CV by default (AIC/AIC$_c$ also supported).

---

## File formats (RData)

* **`yfdobj_<l>_<k>_<hh>.RData`**
  Contains `yfdobj` (`fd`): coefficient matrix `Qy × N_global`.

* **`predictorLst_<l>_<k>_<hh>.RData`**
  Contains `predictorLst` (list length `p`). Entry `j` is:

  * `fd` (`Qx × N_global`) *if* worker `l` owns feature `j`,
  * `NULL` otherwise.

* **`truth_active_idx.RData`**
  Contains `active_idx` (integer vector), used for sensitivity/specificity.

* **`vfl_dp_foboost_results.RData`**
  Arrays of metrics over `(k, hh, fold)` and `res` summary table; also `res_df` (long data.frame per job).

---

## Reproducibility

* The generator sets seeds internally; the runner sets a **job‑specific seed** so parallel jobs are reproducible (`future.seed = TRUE`).
* **Do not mix old and new files**: if you change `N_global`, `p`, basis size, or worker counts, **delete old `yfdobj_*` and `predictorLst_*` files** before regenerating.

Example:

```r
setwd("/path/to/your/dir")
file.remove(list.files(pattern = "^(yfdobj|predictorLst)_.*\\.RData$"))
source("generator_vfl.R")
source("run_vfl_experiment_parallel.R")
```

---

## How VFL is enforced here (and how it differs from HFL)

* **VFL**: Each worker owns a **subset of features** for the **same N individuals**.
  In files: each `predictorLst_<l>_*` has length `p`, with **NULL** entries for all features not owned by worker `l`.
  The runner builds the global design by **selecting** the owning worker’s `fd` for each feature $j$. **No column concatenation across workers**.

* **HFL** (not used here): Each worker owns **different individuals** (rows). You would `cbind` across workers. This is explicitly **not** what the runner does.

---

## Differential Privacy notes (what is DP vs simulation‑only)

* **DP mechanism** (inside `functions.R`):

  * Per‑record clipping in RKHS (coefficient metric $M_x$): $c \mapsto c \cdot \min(1, S_x/\|c\|_{M_x})$.
  * Add **Gaussian noise** with covariance $s_x^2 M_x$ to the **coefficients** (equivalent to isotropic noise in whitened coordinates).
  * **Moment corrections** subtract the known DP covariance and apply the finite‑sample $\frac{N}{N-1}$ centering factor.
* **Simulation‑only shortcut**: `Sx_mode = "empirical"` uses the empirical 95% quantile of per‑record norms as the clip radius (convenient for stability, **not** a DP guarantee).
* **To claim DP**: set `Sx_mode = "fixed"` and pick `Sx_fixed`, then calibrate `sx_default` to your $(\varepsilon,\delta)$ (or zCDP $\rho$) using the **analytic Gaussian mechanism**, accounting for your adjacency notion and composition across parties.

---

## Interpreting results

The runner prints a one‑row summary per `k`:

* **WMAPE_mean / sMAPE_mean / MAPE_mean** — smaller is better.
* **NRMSE_mean** — normalized RMSE; smaller is better.
* **IL2_mean / RIL2_mean** — functional L2 errors in the basis metric; RIL2 is scale‑free.
* **Sensitivity_mean / Specificity_mean** — feature selection quality against `active_idx`.
* **Time_hours_mean** — average runtime per job.
* **Comm_MB_mean** — approximate one‑time communication (MB) for DP coefficient releases.

If you need **better WMAPE**, consider:

* Reducing DP noise (smaller `sx_default`) or increasing clip `Sx_fixed` (if within privacy budget).
* Larger `N_global` or higher `SNR_target` in the generator.
* Tuning penalties (`lambda_s`, `lambda_t`) and shrinkage `nu`.
* Allowing a longer path (`max_steps`) with modest `patience`.
* Ensuring `Sx_mode="fixed"` for reproducible DP behavior (avoid dynamic clipping on heavy‑tailed draws).

---

## Minimal API reference (selected)

```r
# Fit centralized DP FoF (optional path)
fit <- dp_fof_fit(xfd, yfd, Sx, sx,
                  Omega_x=NULL, Omega_y=NULL,
                  lambda_s=0, lambda_t=0, lambda_st=0)

# Fit VFL boosting (used by runner)
fit <- vfl_dp_foboost(xfd_list, yfd,
                      Sx_vec, sx_vec,
                      lambda_s, lambda_t, lambda_st,
                      nu=0.3, max_steps=30,
                      crossfit=TRUE, stop_mode="cv",
                      min_steps=10, patience=6)

# Predict (new X's must be fd lists with the same basis)
yhat <- predict_vfl_dp_foboost(fit, xfd_list_new)
```

**`fit`** contains:

* `B_list`: per‑party FoF operator matrices in coefficient space
* `selected`: selected party index at each step
* `yhat_fd`: fitted responses on the training set (centered back to original mean)
* `centers`: training means used for centering
* `Sigma_list`, `Omega_x_list`, `Omega_y`
* `traces`: SSE and AIC trajectories (if applicable)

---

## Implementation details (method ↔ code alignment)

* Clipping & Gaussian mechanism applied in **coefficient space** with covariance $s_x^2 M_x$.
* **Centered moments** use noisy means (unbiased after subtracting $\Sigma_x$ and multiplying by $N/(N-1)$).
* Penalized FoF: generalized **Sylvester** / **Kronecker** solve with roughness penalties.
* Boosting: per‑party **local FoF update** on partial residuals; **noise‑corrected selection score** subtracts the expected DP inflation; **two‑fold cross‑fit** removes adaptive optimism and restores independence for scoring; early stopping via **CV** (default) or **AIC/AIC$_c$** with a Hutchinson df estimator.

---

## Tips for extending / real‑data VFL

* Replace the generator with your own `fd` objects per party; keep **length‑`p` lists** and `NULL` for non‑owners.
* Ensure **all parties use the exact same basis** (range and `nbasis`) and share the same `N` individuals in the same order.
* For DP deployments, set `Sx_mode="fixed"` and calibrate `sx_default` to your privacy budget; consider **fold‑wise** moment computation (cross‑fit) to strengthen independence assumptions behind selection scoring.

---

## License

Research/academic use. If you need a formal license, add one here.

---

## Acknowledgments

Built on top of the `fda` package and standard linear‑algebra routines (`Matrix`). The method and notation follow the accompanying write‑up on DP‑corrected FoF regression and VFL functional boosting with principled stopping.

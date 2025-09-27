# VFL–DP–FoF: Differentially‑Private Function‑on‑Function Regression in Vertical FL

This repository provides an end‑to‑end R implementation of:

1. **Differentially private (DP) penalized function‑on‑function (FoF) regression** in coefficient space, and
2. A **Vertical Federated Learning (VFL)** wrapper that performs **functional boosting** across parties with a **DP‑aware selection score** and **principled early stopping** (cross‑validated or AIC_c).

It includes a stable VFL **data generator** (orthonormalized coefficients; controlled SNR), a **parallel runner** for model evaluation, a **privacy sweep + black‑box membership‑inference attack (MIA)** to assess privacy risk vs. DP noise, and the **core FoF/boosting routines**.

---

## Repository layout

```
VFL_code/
├─ generator_vfl.R                   # VFL data generator (fixed N across workers; feature ownership)
├─ run_vfl_experiment_parallel.R     # Parallel experiment runner (utility metrics, selection metrics)
├─ run_vfl_privacy_sweep.R           # Grid sweep over sx (DP noise) & workers; launches MIA; aggregates results
├─ mia_blackbox.R                    # Black-box MIA (loss- and confidence-based) + zCDP ε accounting
├─ functions.R                       # DP FoF + VFL functional boosting (DP moments + Sylvester updates)
├─ vfl_dp_foboost_results.RData      # Written by the runner (arrays + summary; created after experiments)
├─ truth_active_idx.RData            # Ground-truth active predictors (from generator)
├─ yfdobj_<l>_<k>_<hh>.RData         # Response fd (identical across l for fixed k,hh)
└─ predictorLst_<l>_<k>_<hh>.RData   # Per-worker predictor list (length p; NULL for non-owners)
```

---

## Requirements

* **R** ≥ 4.1
* Packages: `fda`, `Matrix`, `future.apply`, `pROC` (for AUC), `stats` (base), `parallel` (base)

Install from CRAN:

```r
install.packages(c("fda", "Matrix", "future.apply", "pROC"))
```

---

## Quick start

### 1) Generate VFL data

```r
setwd("/path/to/VFL_code")
source("generator_vfl.R")
```

**What it writes**

* `truth_active_idx.RData`
* For each replicate `hh` and worker count `k`, and each worker `l ∈ {1,…,k}`:

  * `yfdobj_<l>_<k>_<hh>.RData` — an `fd` object of Y **(identical across l)**
  * `predictorLst_<l>_<k>_<hh>.RData` — **list of length p**; entry `j` is:

    * an `fd` if worker `l` owns feature `j`,
    * `NULL` otherwise.
      Ownership is deterministic: `owner_of_feature(j, k) = ((j - 1) %% k) + 1`.

> **Invariant:** Every `predictorLst_*` has `length(predictorLst) == p` and each slot is either `fd` (owned) or `NULL` (not owned).

### 2) Run the parallel VFL experiment (utility & selection)

```r
setwd("/path/to/VFL_code")
source("run_vfl_experiment_parallel.R")
```

This launches parallel jobs over `(duplicate = hh) × (workers = k) × (fold)`, fits DP‑aware VFL boosting, and prints a summary table. It also writes:

* `vfl_dp_foboost_results.RData` — arrays of metrics (WMAPE, sMAPE, MAPE, RMSE/NRMSE, functional IL2/RIL2, sensitivity/specificity, time, comm cost) and a long `res_df` with per‑job results.

### 3) Sweep privacy noise and run black‑box MIA (privacy–utility analysis)

```r
setwd("/path/to/VFL_code")
source("run_vfl_privacy_sweep.R")   # calls mia_blackbox.R internally
```

* Sweeps `sx` (Gaussian noise std in **whitened coefficient space**) and `numworkers`.
* Trains models, runs **loss‑based** and **confidence‑based** MIAs, and computes **zCDP → (ε, δ)** accounting for the one‑shot per‑record releases.
* Prints an aggregated table like:

```
=== Black-box MIA (means) by workers × sx ===
   numworkers   sx  AUC_LA ADV_LA TPR10_LA  AUC_CONF ADV_CONF TPR10_CONF  Eps_train Time_sec
1           2 0.00  0.603  0.19     0.192     0.601    0.186     0.182        Inf     7.17
...
11          2 0.20  0.584  0.17     0.148     0.528    0.089     0.115      9644     5.24
16          2 0.30  0.516  0.08     0.091     0.484    0.050     0.088      4429     5.01
```

**Interpretation (at a glance):**
Higher `sx` ⇒ more noise ⇒ **lower** MIA AUC/ADV/TPR10 (better privacy), but **formal ε** from local per‑record releases can still be large unless clipping is tight or you switch to central‑DP on moments.

---

## Method ↔ code alignment (key points)

* **DP at the feature parties:** Per‑record **clipping** in the RKHS metric induced by the basis Gram matrix (M_x), then add Gaussian noise with covariance (s_x^2 M_x) to coefficient columns (equiv. to isotropic noise in whitened coords).
* **Finite‑sample DP moment correction:** after centering by **noisy means**, correct covariance by (N/(N-1)) and subtract the known DP covariance (\Sigma_x = s_x^2 M_x).
* **FoF update:** Solve a penalized generalized **Sylvester** system for each candidate party (j):
  ((\overline\Gamma_{xx}^{dp,j} + \lambda_s \Omega_{xj})\Delta B_j + \Delta B_j(\lambda_t \Omega_y) + \lambda_{st},\Omega_{xj}\Delta B_j\Omega_y = \overline\Gamma_{xR_{-j}}^{dp,j}).
* **Selection score (DP‑aware):** ( S_{\text{corr}}(j) = |Z_j \Delta B_j - R_{-j}|*F^2 - (N-1),\mathrm{tr}(\Delta B_j^\top \Sigma*{xj} \Delta B_j)).
* **Stopping (default):** two‑fold **CV** on the train split; **AIC/AIC_c** also available with a Hutchinson df estimator.
* **VFL semantics:** Same individuals across workers; each worker owns a subset of **features**. The runner **selects the owner’s `fd`** for each feature; it does **not** concatenate samples across workers (that would be HFL).

---

## Configuration (where to tune)

### In `generator_vfl.R`

* `N_global`: number of individuals (fixed across all `k`).
* `p`: total number of predictors.
* `active_idx`: indices of truly active features (default `1:5`).
* `rank_r`, `a_active`: rank/strength of the true operators in whitened space.
* `SNR_target`: response SNR in whitened space.
* `basisobj`: basis family and size (Bsplines with `nbasis = 20` by default).

### In `run_vfl_experiment_parallel.R`

* `numworkersseq`, `num_duplicate`, `folds_per_worker`.
* **DP knobs**

  * `Sx_mode`: `"fixed"` (**DP‑accounted**) or `"empirical"` (**simulation‑only**; convenience).
  * `Sx_fixed`: fixed clipping radius (whitened‑norm bound mapped to coefficient metric).
  * `sx_default`: per‑record Gaussian noise std in whitened space (calibrate to your ((\varepsilon,\delta)) or zCDP (\rho)).
* **FoF / boosting knobs**

  * `lambda_s`, `lambda_t`, `lambda_st`: penalties.
  * `nu`: shrinkage (e.g., `0.3`).
  * `max_steps`, `min_steps`, `patience`: path length & early stopping.
  * `use_crossfit = TRUE`, `stop_mode = "cv"` (AIC/AIC_c is supported).

### In `run_vfl_privacy_sweep.R` / `mia_blackbox.R`

* `sx_grid`: vector of noise levels to sweep (e.g., `c(0, 0.1, 0.2, 0.3)`).
* `workers_grid`: vector of worker counts to sweep.
* `delta_priv`: the (\delta) used in ((\varepsilon,\delta)) conversion from zCDP.
* Attack choices: **loss‑based** vs **confidence‑based**; both are computed.

---

## Privacy accounting (zCDP → ε, δ)

* Replace‑one adjacency; per‑record Gaussian mechanism in whitened coords with clipping at (S_x) (ℓ₂ sensitivity (2S_x)).
* Per‑feature zCDP: (\rho_j = \frac{(2S_x)^2}{2 s_x^2} = \frac{2 S_x^2}{s_x^2}).
* Composition over features: (\rho_{\text{total}} = \sum_{j=1}^p \rho_j).
* Convert to ((\varepsilon,\delta)): (\varepsilon = \rho_{\text{total}} + 2\sqrt{\rho_{\text{total}}\log(1/\delta)}).

**Helper (invert approx. to pick `sx`):**

```r
# Solve for s_x given target epsilon, delta, p, Sx (replace-one adjacency)
sx_for_epsilon <- function(eps, delta, p, Sx) {
  # crude invert via uniroot on rho_total
  f <- function(rho) rho + 2*sqrt(rho*log(1/delta)) - eps
  rho_tot <- uniroot(f, c(1e-12, 1e12))$root
  # rho_tot = sum_j 2 Sx^2 / s_x^2 = 2 p Sx^2 / s_x^2
  sqrt( (2 * p * Sx^2) / rho_tot )
}
```

> **Note:** Local per‑record DP with many features can produce very **large** ε unless `Sx` is tight or `sx` is large. If you need small ε with good utility, consider **central‑DP on moments** (sensitivity (O(1/N))) as an alternative design.

---

## Outputs & how to read them

* **Experiment runner** prints per‑`k` summary:

  * **Utility**: WMAPE/sMAPE/MAPE, RMSE/NRMSE, functional IL2/RIL2 (basis‑aware).
  * **Selection**: sensitivity (TPR on `active_idx`), specificity (TNR on inactives).
  * **Systems**: time (hours), approximate one‑time **communication cost** (MB) for DP coefficient releases.
* **Privacy sweep + MIA** prints per‑(`workers`, `sx`) aggregates:

  * **AUC / ADV (= 2·AUC − 1) / TPR@10%FPR** for loss‑ and confidence‑based attacks (lower is better).
  * **Eps_train** from zCDP accounting (∞ when `sx = 0`).

**Typical pattern:** as `sx` increases, MIAs degrade toward random (AUC→0.5), but ε may still be large under local per‑record DP unless `Sx` is tight or feature‑wise composition is mitigated.

---

## Reproducibility

* The generator and runner set deterministic seeds (`future.seed = TRUE` for parallel jobs).
* If you change `N_global`, `p`, basis, or worker counts, **delete old data files** to avoid stale‑file mismatches:

```r
setwd("/path/to/VFL_code")
file.remove(list.files(pattern = "^(yfdobj|predictorLst)_.*\\.RData$"))
source("generator_vfl.R")
source("run_vfl_experiment_parallel.R")
```

---

## Troubleshooting

* **`Length mismatch in predictorLst_*: length(predictorLst)!=p`**
  You likely have stale files from a previous run with different `p`. **Delete** old `yfdobj_*` and `predictorLst_*` and regenerate.

* **`predictorLst[[j]] is NULL but worker should own feature j`**
  Ownership is `((j-1) %% k) + 1`. Ensure `generator_vfl.R` wasn’t edited to change ownership silently; regenerate all files.

* **`no rows to aggregate` in privacy sweep**
  The sweep created no successful jobs (e.g., empty `sx_grid` or all jobs failed earlier). Check console for earlier errors, verify `sx_grid`, `workers_grid`, and that the generator has produced files for those `k, hh`.

* **Windows parallel**
  `future::plan(multisession)` is used (Windows‑safe). If you see stuck workers, try reducing `workers`, or set `plan(sequential)` to debug.

---

## Notes: VFL vs HFL

* **VFL (this repo):** same individuals across workers; workers own **disjoint feature subsets**.
* **HFL:** workers own **disjoint subsets of individuals**. This code **does not** concatenate samples across workers (that would be HFL).

---

## Extending to central‑DP (optional)

Instead of per‑record DP on features, add Gaussian noise to the released **moments** (\overline\Gamma_{xx}), (\overline\Gamma_{xy}). Sensitivity scales as (1/N) (with clipping in whitened space), enabling far smaller ε for comparable utility. The `functions.R` solver already works with noisy moments (post‑processing).

---

## Minimal API cheatsheet
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
# Centralized DP FoF (optional path)
fit_c <- dp_fof_fit(xfd, yfd, Sx, sx, lambda_s=0.05, lambda_t=0.05)

# VFL DP boosting (main path)
fit <- vfl_dp_foboost(
  xfd_list, yfd, Sx_vec, sx_vec,
  lambda_s=0.05, lambda_t=0.05, lambda_st=0,
  nu=0.3, max_steps=30, crossfit=TRUE, stop_mode="cv",
  min_steps=10, patience=6
)

# Predict on new data (same basis)
yhat <- predict_vfl_dp_foboost(fit, xfd_list_new)
```

---

## License

Research/academic use. Add a formal license if needed.

---

## Acknowledgments

Built on `fda`, `Matrix`, `future.apply`, and standard linear‑algebra routines. The implementation matches the accompanying write‑up on DP‑corrected FoF regression and VFL functional boosting with principled stopping.

# Differentially Private Function‑on‑Function Regression (FoF)

### Vertical Federated Functional Boosting with Principled Stopping

This repository implements a differentially‑private (DP) **function‑on‑function (FoF)** regression learner and a **vertical federated learning (VFL) functional boosting** algorithm with DP‑aware selection and principled stopping (CV or AIC_c). It includes:

* A stable synthetic **data generator** (orthonormalized coefficients, controlled SNR).
* A **parallel runner** that varies the number of VFL **parties** (a.k.a. “workers” in earlier scripts) with **constant sample size** for fair comparisons.
* A core **`functions.R`** module (DP moment correction, penalized FoF solver with Sylvester/Kronecker structure, DP‑aware boosting with cross‑fit selection + CV/AIC_c stopping).

---

## Contents

```
.
├── README.md                  # This file
├── functions.R                # DP FoF + VFL functional boosting (core)
├── generator.R                # Stable data generator (constant N across parties)
├── run_vfl_parallel.R         # Parallel experiment driver (varies #parties J)
├── vfl_dp_foboost_results.RData  # Saved after a run (metrics arrays & summary)
└── *.RData files              # Generated datasets:
    ├── yfdobj_<hh>.RData
    ├── predictorLst_<hh>.RData
    └── truth_active_idx.RData
```

> **Terminology:** This is **VFL (vertical)**: each party holds disjoint **predictor sets** (basis coefficients of functional predictors). The **active party** holds the response curves $Y$. The term “workers” in older scripts refers to **parties**.

---

## Installation

* **R ≥ 4.2** recommended
* CRAN packages:

  ```r
  install.packages(c("fda", "Matrix", "future.apply"))
  ```
* Optionally set a project directory (used by the scripts):

  ```r
  root_dir <- "/Users/yuding/Dropbox/VFL_code"  # change to your path
  setwd(root_dir)
  ```

---

## Quick Start

1. **Generate data** (constant $N$ per replicate, independent of #parties):

```r
source("generator.R")
# Creates:
#  - yfdobj_<hh>.RData
#  - predictorLst_<hh>.RData
#  - truth_active_idx.RData
```

2. **Run experiments in parallel** (vary #parties; keep N fixed):

```r
source("run_vfl_parallel.R")
# Produces: vfl_dp_foboost_results.RData
# Prints a summary data.frame 'res' to the console.
```

3. **Inspect results**:

```r
load("vfl_dp_foboost_results.RData")
print(res)        # summary across (replicate × parties × folds)
str(res_df)       # per-job metrics if you need fine-grained analysis
```

---

## What’s Implemented

### Differential Privacy (DP)

* **Per‑record clipping in coefficient space** (RKHS norm from basis Gram matrix):

  $$
  c \leftarrow c \cdot \min\!\Big(1,\ \frac{S_x}{\|c\|_{M_x}}\Big),\qquad 
  \|c\|_{M_x}^2 = c^\top M_x c
  $$
* **Gaussian mechanism**: add $e \sim \mathcal{N}(0, s_x^2\,M_x)$ to each clipped coefficient vector.
* **Finite‑sample, centered moment correction** (unbiased under noisy centering):

  $$
  \overline\Gamma_{xx}^{dp} = \frac{N}{N-1}\Gamma^{dp}_{xx,c} - \Sigma_x,\quad
  \overline\Gamma_{xy}^{dp} = \frac{N}{N-1}\Gamma^{dp}_{xy,c}
  $$

  where $\Sigma_x = s_x^2 M_x$.
* **DP‑aware selection score** for boosting (subtracts expected DP inflation):

  $$
  S_{\mathrm{corr}}(j)=\|Z_j\Delta B_j - R_{-j}\|_F^2 - (N-1)\,\mathrm{tr}(\Delta B_j^\top \Sigma_{xj}\Delta B_j).
  $$

> Optional: set `keep_total_privacy <- TRUE` in the runner to approximately **hold the total zCDP budget constant** across different numbers of parties by scaling per‑predictor noise $s_x \propto \sqrt{J}$.

### Penalized FoF with Sylvester/Kronecker Structure

* Solve for $B$ in:

  $$
  (\overline\Gamma_{xx}^{dp}+\lambda_s\Omega_x)B
  +B(\lambda_t\Omega_y)
  +\lambda_{st}\Omega_x B\Omega_y
  = \overline\Gamma_{xy}^{dp}.
  $$
* Implemented via a **Kronecker linear system** with **robust Cholesky** solver and tiny ridge, with fall‑back to `solve()` if needed.

### VFL Functional Boosting

* **Base learner:** one FoF per party; partial residual updates.
* **Two‑fold cross‑fit selection:** compute $\Delta B_j$ on fold A and score on B (and vice versa). Reduces adaptivity bias.
* **Stopping:** cross‑validation (CV) SSE early stopping by default; **AIC/AIC_c** also available.
* **Degrees of freedom:** Hutchinson trace estimation along the frozen boosting path.

---

## Scripts & Key Settings

### `generator.R`

* Produces **one dataset per replicate** (not per party). All parties then see vertical splits of the same $N$ curves.
* Controls:

  * `N_total`: total number of samples per replicate (constant across experiments).
  * `p`: number of predictors.
  * `active_idx`: ground‑truth active predictors (used for sensitivity/specificity).
  * `SNR_target`: sets noise level in whitened response space.

**Outputs:**

* `yfdobj_<hh>.RData`, `predictorLst_<hh>.RData`, `truth_active_idx.RData`.

### `run_vfl_parallel.R`

* Runs **parallel jobs** over `replicate × parties × folds`.
* Controls:

  * `parties_seq`: vector of party counts $J$ (e.g., `c(2,4,6,8,10)`).
  * `sx_default`: DP noise std per predictor (scaled if `keep_total_privacy=TRUE`).
  * Penalties & boosting: `lambda_s`, `lambda_t`, `lambda_st`, `nu`, `max_steps`, `min_steps`, `patience`.
  * Stopping mode: CV (default), AIC, or AIC_c.

**Outputs:**

* `vfl_dp_foboost_results.RData` with full arrays and `res` summary.

### `functions.R`

* Core module:

  * `dp_release_coefficients()`
  * `form_dp_moments()`
  * `solve_penalized_fof()` (robust symmetric solve)
  * `vfl_dp_foboost()` and `predict_vfl_dp_foboost()`

---

## How Parties Are Formed (VFL)

* With $p$ predictors fixed (e.g., 20), and $J$ parties, we **distribute predictors** to parties:

  * Round‑robin split by default (configurable).
* The **sample size $N$ is fixed** and **shared** across parties. Increasing $J$ **does not** change $N$ in this setup.

> If you observed accuracy increasing with more “workers” before, it was likely due to **increasing $N$** alongside $J$. The current design keeps $N$ fixed to make comparisons across $J$ meaningful.

---

## Communication Cost

Per party $g$ with $m_g$ predictors of dimension $Q_x$ (basis size):

$$
\text{bytes}_g \approx \big(N_{\text{train}} \cdot (m_g Q_x) + (m_g Q_x)^2\big)\times 8.
$$

We report MB aggregated across parties.

---

## Common Hyperparameters (where to tune)

* **DP:**

  * `sx_default` (per‑predictor noise std). Smaller improves utility (less privacy).
  * `adapt_Sx()` uses the 95th percentile of $M_x$‑norms per predictor for clipping radii.
  * Optionally set `keep_total_privacy <- TRUE` to compensate for changing $J$.

* **Penalties & Boosting:**

  * `lambda_s`, `lambda_t`, `lambda_st`: smoothness control.
  * `nu`: shrinkage (e.g., 0.1–0.5). Smaller nu = more conservative boosting.
  * `max_steps`, `min_steps`, `patience`: path length & stopping.
  * `use_crossfit = TRUE`: recommended for robust selection under DP.

---

## Expected Behavior

* With **constant $N$** and **fixed DP calibration**, performance should be **roughly stable** across $J$ (minor differences from how predictors cluster in parties).
* If you set `keep_total_privacy <- TRUE`, per‑predictor noise grows with $\sqrt{J}$, keeping total privacy roughly constant; utility then should **not** systematically improve with $J$.

---

## Reproducibility

* **Deterministic folds** independent of $J$ (contiguous splits) are used.
* `future_lapply(..., future.seed=TRUE)` ensures reproducible parallel RNG streams.
* You can set a global `set.seed()` at the top of runner/generator for fully fixed runs.

---

## Troubleshooting

* **`unused argument (coefs=...)`**
  Always construct `fd` as `fd(coef = ..., basisobj = ...)`. (The code already does this.)

* **`subscript out of bounds` when building datasets**
  Ensure `p` in the runner equals `length(predictorLst)` in the generated dataset.

* **Linear solver instability / singular `H`**
  Increase `lambda_s`, or adjust `stabilize = list(alpha=0.05, ridge=1e-6, H_ridge=1e-10)` in `solve_penalized_fof()` call sites.

* **Weird accuracy trends with #parties**
  Confirm you are using the **new generator** (constant $N$). If you reused old per‑worker files, delete them and regenerate.

---

## Calibrating DP to $(\varepsilon,\delta)$ (quick guide)

In whitened coordinates (sensitivity $\Delta = 2S_x$ for replace‑one), a conservative Gaussian calibration is:

$$
s_x \ \ge\ \frac{\Delta \sqrt{2\log(1.25/\delta)}}{\varepsilon}.
$$

For zCDP, each per‑predictor mechanism satisfies $\rho = \Delta^2/(2 s_x^2)$; across predictors/parties $\rho$ **adds**.
Adjust `sx_default` (and optionally `keep_total_privacy`) accordingly. This code does **not** implement a full accountant; calibrate offline and set `sx_default` to the value you want.

---

## Example: Minimal Sanity Run

```r
# 1) Generate fewer replicates for a quick smoke test
source("generator.R")   # optionally set num_duplicate <- 2, N_total <- 400 inside

# 2) Edit run_vfl_parallel.R for a light run:
#    parties_seq <- c(2, 4)
#    num_duplicate <- 2
#    folds_per_worker <- 2
source("run_vfl_parallel.R")

load("vfl_dp_foboost_results.RData")
print(res)
```

---

## Extending / Modifying

* **Central‑DP on Y:** `dp_fof_fit()` supports `sy > 0`; not used in VFL by default (since Y stays at the active party).
* **Alternative penalties:** provide `Omega_x_list` / `Omega_y` directly to `vfl_dp_foboost()`.
* **Stopping criterion:** switch `stop_mode` to `"aic_train"` or `"aic_train_dp"` (train‑set AIC/AIC_c, optionally DP‑corrected SSE).

---

## Notes on Licensing & Use

This code is for research and prototyping. While it implements recognized DP mechanisms, **end‑to‑end privacy accounting** (including composition across all releases) is not automated here. Use with care for any high‑stakes deployment.

---

## Contact / Questions

* If you encounter a specific error message, copy the full console output and note:

  * The exact commit of `functions.R`, and the values of `parties_seq`, `sx_default`, `lambda_*`, and `nu`.
  * Whether you regenerated data with the **new generator**.
* I’m happy to help diagnose with that information.

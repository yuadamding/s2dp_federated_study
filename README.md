# **Comprehensive Simulation Suite for S2DP-FGB-LDP**

This repository contains the complete Python source code for an integrated suite of experiments designed to rigorously validate the **S2DP-FGB-LDP** framework, as described in the paper *"S2DP-FGB-LDP: A Clipping-Free, Single-Shot Privacy Framework for Federated Function-on-Function Regression"*.

## **1. High-Level Summary**

The S2DP-FGB-LDP algorithm introduces a novel "clipping-free" mechanism for achieving differential privacy in federated learning on functional data. Instead of truncating high-magnitude signals (clipping), it uses a quadratic roughness penalty (`lambda`) to analytically bound the data's sensitivity. This creates a "single-shot" privacy framework where the entire privacy budget is spent once, decoupling privacy cost from model complexity.

This simulation suite is designed to stress-test the algorithm along every dimension that is critical for a robust evaluation in a top-tier systems or ML privacy venue.

## **2. Core Scientific Questions Investigated**

This suite is designed to produce empirical evidence to answer the following key research questions, as outlined in the experimental roadmap:

| Question                                                              | Evidence We Produce                                                                 |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Utility**  Does clipping‑free DP preserve accuracy?                 | RMSE for FoFR, compared against non-private and iterative DP baselines.             |
| **Privacy**  Does the analytic radius truly control disclosure risk?       | Empirical Membership & Attribute Inference Attacks (MIA/AIA).                     |
| **Trade‑off**  Can smoothness λ jointly optimise utility & sensitivity? | Visualization of Pareto frontiers showing MSE vs. Sensitivity as a function of `lambda`. |
| **Robustness**  What if assumptions fail?                             | Stress tests for basis mis-specification, public bound violations, and party drop-out. |
| **Scalability**  How does performance scale with n, K, and M?          | Wall-clock runtime and communication cost plots against key problem dimensions.        |

## **3. Repository Structure**

The project is organized into a modular structure for clarity and extensibility:

```
s2dp_federated_study/
├── experiments.yaml             # Defines all experimental parameters.
├── runner.py                    # Main orchestrator script that reads the YAML and runs experiments.
├── requirements.txt             # Lists all necessary Python packages for reproducibility.
├── notebooks/
│   └── analysis_dashboard.ipynb # Jupyter notebook to load and visualize experiment results.
└── src/
    ├── __init__.py
    ├── simgen/
    │   └── data_factory.py      # Data-generation factory with multiple kernels and error models.
    ├── privacy/
    │   └── mechanisms.py        # Implements the core privacy mechanisms.
    ├── learners/
    │   └── boosting.py          # Implements the VFL Gradient Boosting back-end with scalability/dropout hooks.
    └── eval/
        └── metrics.py           # Implements evaluation metrics and privacy attacks (MIA, AIA).
```

## **4. Step-by-Step Execution Guide**

1.  **Set Up the Environment:**
    ```bash
    # Create and activate a Python virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install all dependencies
    pip install -r requirements.txt
    ```

2.  **Configure Experiments (Optional):**
    Open `experiments.yaml`. You can enable or disable entire studies by setting `enabled: true` or `enabled: false`. You can also adjust parameters like `n_subjects` or `repeats` for quicker test runs.

3.  **Run the Simulation Experiments:**
    Execute the main runner script from the project's root directory. This will run all studies where `enabled: true`.
    ```bash
    python runner.py
    ```
    The script will create a `results/` directory and save the output of each experiment in a corresponding subfolder (e.g., `results/sensitivity_check/`).

4.  **Analyze and Visualize the Results:**
    Once the runner script has finished, launch Jupyter and open the analysis notebook.
    ```bash
    jupyter lab notebooks/analysis_dashboard.ipynb
    ```
    Run all the cells in the notebook to generate the final plots for each completed study.

## **5. Guide to the Simulation Studies**

This suite implements the full experimental roadmap. Here is a guide to what each study tests:

*   **`sensitivity_check`**: Validates the paper's core theoretical claim by plotting the analytically derived sensitivity bound `Δ₂` against the empirically measured maximum norm of the smoothed coefficients. This demonstrates how well the theory matches practice across different `lambda` values.
*   **`basis-ablation`**: Tests the algorithm's robustness to a mis-specified basis, a common real-world scenario. It compares the model's utility (MSE) when using the correct B-spline basis versus a less suitable Fourier basis.
*   **`scalability`**: Measures the wall-clock runtime and total communication bytes as a function of the number of subjects (`n`), parties (`K`), and boosting rounds (`M`), providing critical performance benchmarks.
*   **`attribute_inference_attack`**: Implements a strong privacy test where an attacker attempts to infer a sensitive binary attribute (e.g., high/low latent risk) from the sanitized model outputs. It produces ROC curves to visualize the attacker's success under different privacy settings.
*   **(Future Work) `harmonisation-mispec`, `dropout`, `central-vs-local`**: The framework is designed to easily accommodate these additional stress tests, which are defined in the YAML file but are currently disabled in the runner.

## **6. Component Guide: What Each File Does**

*   **`experiments.yaml`**: The central control file. Use it to enable/disable studies and modify parameters without changing Python code.

*   **`src/simgen/data_factory.py`**:
    *   **`DataFactory`**: A class that generates the complex 10-party dataset. **Updated** to support different error distributions (e.g., `student_t`) for robustness testing.

*   **`src/privacy/mechanisms.py`**:
    *   **`BasisFactory`**: **New module** to generate different families of basis functions (B-spline, Fourier) for the basis-ablation study.
    *   **`s2dp_clipping_free()`**: The core implementation of the proposed mechanism. **Updated** to return both the analytic sensitivity and the empirical sensitivity for direct comparison in the `sensitivity_check` study.

*   **`src/learners/boosting.py`**:
    *   **`VFLGradientBoosting`**: Implements the federated boosting algorithm. **Updated** with hooks to measure `runtime` and `comms_bytes` for the `scalability` study and to handle party `dropout` for robustness checks.

*   **`src/eval/metrics.py`**:
    *   **`perform_mia_attack()`**: Implements the standard Membership Inference Attack.
    *   **`perform_aia_attack()`**: **New function** that implements the stronger Attribute Inference Attack, training a simple classifier to predict a hidden feature from model outputs.

*   **`runner.py`**:
    *   The main orchestrator. **Heavily updated** with logic to parse the `exp_id` for each experiment and run the correct combination of functions with the correct parameters, saving the appropriate results for each study.

*   **`notebooks/analysis_dashboard.ipynb`**:
    *   The final visualization step. **Updated** with new cells to load and plot the results from all the newly added studies, creating a comprehensive dashboard of the framework's performance.
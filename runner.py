# ======================================================================
# runner.py
#
# Main orchestrator for the S2DP-FGB-LDP simulation suite.
# This version corrects the unpacking ValueError.
# ======================================================================
import yaml
import pandas as pd
import numpy as np
import os
import itertools
import time
import ray
import psutil

from src.simgen.data_factory import DataFactory
from src.privacy.mechanisms import BasisFactory, s2dp_clipping_free, central_dp_gaussian
from src.learners.boosting import VFLGradientBoosting
from src.eval.metrics import perform_mia_attack, perform_aia_attack
from sklearn.model_selection import train_test_split

# --- Global Constants ---
B_PUBLIC = 15.0
N_BASIS = 30
BASIS_ORDER = 4

@ray.remote
def run_repetition(exp_config, p_set, repeat_idx):
    """
    This function is a Ray remote task, executing one full Monte Carlo repetition for a single parameter set.
    """
    seed_val = abs(repeat_idx + hash(str(tuple(sorted(p_set.items()))))) % (2**32)
    np.random.seed(seed_val)
    
    factory_args = {
        'seed': seed_val,
        'n_subjects': exp_config.get('n_subjects') or p_set.get('n_list'),
        'n_parties': exp_config.get('n_parties') or p_set.get('K_list'),
        'error_dist': p_set.get('response_noise_dist', 'gaussian'),
        'df': p_set.get('df', 3),
        'amplitude_scale': np.sqrt(p_set.get('B_true', 10.0) / 10.0)
    }
    factory = DataFactory(**{k: v for k, v in factory_args.items() if v is not None})
    X, Y, Y_truth, latent_risk, time_s, time_t = factory.generate_ecg_data()
    
    results_list = []
    exp_id = exp_config['exp_id']
    
    if exp_id == 'sensitivity_check':
        _, analytic_sens, emp_sens, _ = s2dp_clipping_free(Y, time_t, 'bspline', N_BASIS, BASIS_ORDER, p_set['lambda_regs'], B_PUBLIC, p_set['epsilon'], p_set['delta'])
        results_list.append({'lambda': p_set['lambda_regs'], 'analytic_sensitivity': analytic_sens, 'empirical_sensitivity': emp_sens})

    elif exp_id == 'harmonisation_mis_specification':
        public_b = p_set['B_public_list']
        sanitized_y,_,_,violations = s2dp_clipping_free(Y, time_t, 'bspline', N_BASIS, BASIS_ORDER, 1e-4, public_b, 1.0, 1e-5)
        sanitized_x = [s2dp_clipping_free(X[:, k, :], time_s, 'bspline', N_BASIS, BASIS_ORDER, 1e-4, public_b, 1.0, 1e-5)[0] for k in range(X.shape[1])]
        model,_,_,_ = VFLGradientBoosting().fit(sanitized_y, sanitized_x)
        basis_t_matrix = BasisFactory.get_basis_matrix(time_t, n_basis=N_BASIS, order=BASIS_ORDER)
        mse = np.mean((Y_truth.T - (basis_t_matrix @ model.predict().T))**2)
        results_list.append({'B_public': public_b, 'mse': mse, 'violations': violations})

    elif exp_id == 'basis-ablation':
        basis_family = p_set['basis_families']
        # CORRECTED UNPACKING: Always expect 4 return values
        sanitized_y,_,_,_ = s2dp_clipping_free(Y, time_t, basis_family, N_BASIS, BASIS_ORDER, p_set['lambda_reg'], B_PUBLIC, p_set['epsilon'], p_set['delta'])
        sanitized_x = [s2dp_clipping_free(X[:,k,:], time_s, basis_family, N_BASIS, BASIS_ORDER, p_set['lambda_reg'], B_PUBLIC, p_set['epsilon'], p_set['delta'])[0] for k in range(X.shape[1])]
        model,_,_,_ = VFLGradientBoosting().fit(sanitized_y, sanitized_x)
        basis_matrix = BasisFactory.get_basis_matrix(time_t, family=basis_family, n_basis=N_BASIS, order=BASIS_ORDER)
        mse = np.mean((Y_truth.T - (basis_matrix @ model.predict().T))**2)
        results_list.append({'basis_family': basis_family, 'mse': mse})
    
    elif exp_id == 'scalability':
        model, runtime, comms_bytes, peak_mem = VFLGradientBoosting(n_rounds=p_set['M_list']).fit(Y, [X[:,k,:] for k in range(X.shape[1])])
        results_list.append({'n_subjects': p_set['n_list'], 'n_parties': p_set['K_list'], 'M_rounds': p_set['M_list'], 'runtime': runtime, 'comms_bytes': comms_bytes, 'peak_rss_mb': peak_mem})

    elif exp_id == 'attribute_inference_attack':
        train_idx, test_idx = train_test_split(np.arange(exp_config['n_subjects']), test_size=0.5, random_state=repeat_idx)
        # CORRECTED UNPACKING: Always expect 4 return values, use _ for unused ones
        sanitized_y,_,_,_ = s2dp_clipping_free(Y[train_idx], time_t, 'bspline', N_BASIS, BASIS_ORDER, p_set['lambda_reg'], B_PUBLIC, p_set['epsilons'], p_set['delta'])
        sanitized_x = [s2dp_clipping_free(X[train_idx,k,:],time_s,'bspline',N_BASIS,BASIS_ORDER,p_set['lambda_reg'],B_PUBLIC,p_set['epsilons'],p_set['delta'])[0] for k in range(X.shape[1])]
        model,_,_,_ = VFLGradientBoosting().fit(sanitized_y, sanitized_x)
        _,_,auc_score = perform_aia_attack(model.predict(), latent_risk[train_idx], model.predict(), latent_risk[test_idx])
        results_list.append({'epsilon': p_set['epsilons'], 'auc': auc_score})

    elif exp_id == 'party_dropout':
        sanitized_y,_,_,_ = s2dp_clipping_free(Y, time_t, 'bspline', N_BASIS, BASIS_ORDER, p_set['lambda_reg'], B_PUBLIC, p_set['epsilon'], p_set['delta'])
        sanitized_x = [s2dp_clipping_free(X[:,k,:], time_s, 'bspline', N_BASIS, BASIS_ORDER, p_set['lambda_reg'], B_PUBLIC, p_set['epsilon'], p_set['delta'])[0] for k in range(X.shape[1])]
        model,_,_,_ = VFLGradientBoosting(n_rounds=p_set['M_rounds']).fit(sanitized_y, sanitized_x, dropout_mode=p_set['dropout_mode'], dropout_rate=p_set['dropout_rate'])
        basis_t_matrix = BasisFactory.get_basis_matrix(time_t, n_basis=N_BASIS, order=BASIS_ORDER)
        mse = np.mean((Y_truth.T - (basis_t_matrix @ model.predict().T))**2)
        results_list.append({'dropout_mode': p_set['dropout_mode'], 'mse': mse})
    
    elif exp_id == 'central_vs_local_dp':
        basis_t_matrix = BasisFactory.get_basis_matrix(time_t, n_basis=N_BASIS, order=BASIS_ORDER)
        sanitized_y_local,_,_,_ = s2dp_clipping_free(Y, time_t, 'bspline', N_BASIS, BASIS_ORDER, p_set['lambda_reg'], B_PUBLIC, p_set['epsilons'], p_set['delta'])
        sanitized_x_local = [s2dp_clipping_free(X[:,k,:],time_s,'bspline',N_BASIS,BASIS_ORDER,p_set['lambda_reg'],B_PUBLIC,p_set['epsilons'],p_set['delta'])[0] for k in range(X.shape[1])]
        model_local,_,_,_ = VFLGradientBoosting().fit(sanitized_y_local, sanitized_x_local)
        mse_local = np.mean((Y_truth.T - (basis_t_matrix @ model_local.predict().T))**2)
        results_list.append({'mechanism': 'S2DP (Local)', 'epsilon': p_set['epsilons'], 'mse': mse_local})
        
        sanitized_y_central = central_dp_gaussian(Y, time_t, 'bspline', N_BASIS, BASIS_ORDER, p_set['epsilons'], p_set['delta'])
        sanitized_x_central = [central_dp_gaussian(X[:,k,:], time_s, 'bspline', N_BASIS, BASIS_ORDER, p_set['epsilons'], p_set['delta']) for k in range(X.shape[1])]
        model_central,_,_,_ = VFLGradientBoosting().fit(sanitized_y_central, sanitized_x_central)
        mse_central = np.mean((Y_truth.T - (basis_t_matrix @ model_central.predict().T))**2)
        results_list.append({'mechanism': 'Central DP', 'epsilon': p_set['epsilons'], 'mse': mse_central})

    elif exp_id == 'heavy_tailed_error':
        sanitized_y,_,_,_ = s2dp_clipping_free(Y, time_t, 'bspline', N_BASIS, BASIS_ORDER, p_set['lambda_reg'], B_PUBLIC, p_set['epsilon'], p_set['delta'])
        sanitized_x = [s2dp_clipping_free(X[:,k,:], time_s, 'bspline', N_BASIS, BASIS_ORDER, p_set['lambda_reg'], B_PUBLIC, p_set['epsilon'], p_set['delta'])[0] for k in range(X.shape[1])]
        model,_,_,_ = VFLGradientBoosting().fit(sanitized_y, sanitized_x)
        basis_t_matrix = BasisFactory.get_basis_matrix(time_t, n_basis=N_BASIS, order=BASIS_ORDER)
        mse = np.mean((Y_truth.T - (basis_t_matrix @ model.predict().T))**2)
        results_list.append({'error_dist': p_set['error_distribution'], 'mse': mse})
            
    return results_list

def run_experiment_suite(exp_config):
    if not exp_config.get('enabled', False):
        print(f"--- Skipping Experiment: {exp_config['exp_id']} (disabled in YAML) ---")
        return
    print(f"\n--- Running Experiment Suite: {exp_config['exp_id']} ---")
    
    params = exp_config.get('parameters', {})
    param_keys = list(params.keys()); param_values = [v if isinstance(v, list) else [v] for v in params.values()]
    param_grid = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]

    futures = [run_repetition.remote(exp_config, p_set, i) for i in range(exp_config.get('repeats', 1)) for p_set in param_grid]
    
    results_nested = ray.get(futures)
    results = [item for sublist in results_nested for item in sublist]

    df = pd.DataFrame(results)
    output_path = os.path.join("results", exp_config['exp_id']); os.makedirs(output_path, exist_ok=True)
    df.to_csv(os.path.join(output_path, "results.csv"), index=False)
    print(f"--- Experiment {exp_config['exp_id']} finished. Results saved. ---")

if __name__ == '__main__':
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    with open('experiments.yaml', 'r') as f:
        experiments = yaml.safe_load(f)
    for exp_config in experiments:
        run_experiment_suite(exp_config)
    ray.shutdown()
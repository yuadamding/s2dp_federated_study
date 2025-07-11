import numpy as np
from patsy import dmatrix

class BasisFactory:
    @staticmethod
    def get_basis_matrix(time_points, family='bspline', n_basis=30, order=4):
        if family == 'bspline':
            sorted_indices = np.argsort(time_points); knots = np.quantile(time_points[sorted_indices], np.linspace(0, 1, n_basis - order + 2))
            return np.asarray(dmatrix(f"bs(x, knots=knots, degree={order-1}, include_intercept=True)-1", {"x": time_points[sorted_indices]}, return_type='matrix'))[np.argsort(sorted_indices)]
        elif family == 'fourier':
            n_pairs = (n_basis - 1) // 2; freqs = np.arange(n_pairs + 1); t_norm = 2*np.pi*(time_points-time_points.min())/(time_points.max()-time_points.min())
            matrix = [np.ones_like(t_norm)] + [f(freq * t_norm) for freq in freqs[1:] for f in (np.sin, np.cos)]
            return np.vstack(matrix).T[:, :n_basis]

def get_penalty_matrix(actual_n_basis):
    D = np.diff(np.eye(actual_n_basis), n=2, axis=0); return D.T @ D

def s2dp_clipping_free(raw_data, time_points, basis_family, n_basis, order, lambda_reg, B_public, epsilon, delta):
    n_obs, n_pts = raw_data.shape; Phi = BasisFactory.get_basis_matrix(time_points, family=basis_family, n_basis=n_basis, order=order)
    actual_n_basis = Phi.shape[1]; Omega = get_penalty_matrix(actual_n_basis)
    matrix_to_invert = Phi.T @ Phi + lambda_reg * Omega
    condition_number = np.linalg.cond(matrix_to_invert)
    if condition_number > 1e12: print(f"Warning: High condition number ({condition_number:.1e}) for λ={lambda_reg:.1e}")
    S_lambda = np.linalg.pinv(matrix_to_invert) @ Phi.T
    smooth_coefs = (S_lambda @ raw_data.T).T
    norm_S_lambda = np.linalg.norm(S_lambda, 2)
    analytic_sensitivity = 2 * norm_S_lambda * np.sqrt(n_pts * B_public)
    empirical_sensitivity = 2 * np.max(np.linalg.norm(smooth_coefs, axis=1)) if n_obs > 0 else 0
    sigma = (analytic_sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon if epsilon > 0 and epsilon != np.inf else 0
    noise = np.random.normal(0, sigma, smooth_coefs.shape)
    violations = np.sum((np.sum(raw_data**2, axis=1) / n_pts) > B_public)
    return smooth_coefs + noise, analytic_sensitivity, empirical_sensitivity, violations

def central_dp_gaussian(raw_data, time_points, basis_family, n_basis, order, epsilon, delta):
    Phi = BasisFactory.get_basis_matrix(time_points, family=basis_family, n_basis=n_basis, order=order)
    coefs = (np.linalg.pinv(Phi) @ raw_data.T).T
    norms = np.linalg.norm(coefs, axis=1)
    clipping_threshold = np.median(norms)
    scale_factors = np.minimum(1, clipping_threshold / (norms + 1e-9))
    clipped_coefs = coefs * scale_factors[:, np.newaxis]
    sensitivity = 2 * clipping_threshold
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    noise = np.random.normal(0, sigma, clipped_coefs.shape)
    return clipped_coefs + noise
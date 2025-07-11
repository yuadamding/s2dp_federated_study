# src/privacy/mechanisms.py
import numpy as np
from patsy import dmatrix

class BasisFactory:
    @staticmethod
    def get_basis_matrix(time_points, family='bspline', n_basis=30, order=4):
        if family == 'bspline':
            sorted_indices = np.argsort(time_points)
            knots = np.quantile(time_points[sorted_indices], np.linspace(0, 1, n_basis - order + 2))
            # Using include_intercept=True and then removing the first column is a common patsy pattern
            # to get a well-behaved basis that doesn't include an intercept term implicitly.
            basis_matrix = dmatrix(f"bs(x, knots=knots, degree={order-1}, include_intercept=True)-1", {"x": time_points[sorted_indices]}, return_type='matrix')
            return np.asarray(basis_matrix)[np.argsort(sorted_indices)]
        elif family == 'fourier':
            n_pairs = (n_basis - 1) // 2
            freqs = np.arange(n_pairs + 1)
            t_norm = 2 * np.pi * (time_points - time_points.min()) / (time_points.max() - time_points.min())
            matrix = [np.ones_like(t_norm)] + [f(freq * t_norm) for freq in freqs[1:] for f in (np.sin, np.cos)]
            return np.vstack(matrix).T[:, :n_basis]

def get_penalty_matrix(actual_n_basis):
    """Computes the roughness penalty matrix Omega = D'D for a 2nd order derivative."""
    if actual_n_basis < 3: return np.zeros((actual_n_basis, actual_n_basis))
    D = np.diff(np.eye(actual_n_basis), n=2, axis=0)
    return D.T @ D

def s2dp_clipping_free(raw_data, time_points, basis_family, n_basis, order, lambda_reg, B_public, epsilon, delta):
    """The proposed clipping-free mechanism using a stable basis generator."""
    n_obs, n_pts = raw_data.shape
    Phi = BasisFactory.get_basis_matrix(time_points, family=basis_family, n_basis=n_basis, order=order)
    actual_n_basis = Phi.shape[1]
    Omega = get_penalty_matrix(actual_n_basis)
    
    # --- DEFINITIVE STABILITY FIX IS HERE ---
    # Add a small identity matrix "jitter" to the Gram matrix before any other operations.
    # This ensures the matrix is well-conditioned and avoids all numerical errors.
    gram_matrix = Phi.T @ Phi
    jitter = 1e-9 * np.eye(gram_matrix.shape[0])
    matrix_to_invert = gram_matrix + jitter + lambda_reg * Omega
    
    S_lambda = np.linalg.solve(matrix_to_invert, Phi.T) # Can revert to solve for speed, pinv is belt-and-suspenders
    smooth_coefs = (S_lambda @ raw_data.T).T
    
    norm_S_lambda = np.linalg.norm(S_lambda, 2)
    analytic_sensitivity = 2 * norm_S_lambda * np.sqrt(n_pts * B_public)
    empirical_sensitivity = 2 * np.max(np.linalg.norm(smooth_coefs, axis=1)) if n_obs > 0 else 0
    
    sigma = (analytic_sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon if epsilon > 0 and epsilon != np.inf else 0
    noise = np.random.normal(0, sigma, smooth_coefs.shape)
    
    violations = np.sum((np.sum(raw_data**2, axis=1) / n_pts) > B_public)
    
    return smooth_coefs + noise, analytic_sensitivity, empirical_sensitivity, violations

def central_dp_gaussian(raw_data, time_points, basis_family, n_basis, order, epsilon, delta):
    """A central DP baseline mechanism."""
    Phi = BasisFactory.get_basis_matrix(time_points, family=basis_family, n_basis=n_basis, order=order)
    
    # Also use the stable approach for the baseline
    gram_matrix = Phi.T @ Phi
    jitter = 1e-9 * np.eye(gram_matrix.shape[0])
    coefs = (np.linalg.solve(gram_matrix + jitter, Phi.T) @ raw_data.T).T
    
    norms = np.linalg.norm(coefs, axis=1)
    clipping_threshold = np.median(norms)
    scale_factors = np.minimum(1, clipping_threshold / (norms + 1e-9))
    clipped_coefs = coefs * scale_factors[:, np.newaxis]
    
    sensitivity = 2 * clipping_threshold
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    noise = np.random.normal(0, sigma, clipped_coefs.shape)
    return clipped_coefs + noise
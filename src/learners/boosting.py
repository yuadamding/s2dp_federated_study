from sklearn.linear_model import LinearRegression
import numpy as np, time, psutil, os

class VFLGradientBoosting:
    def __init__(self, n_rounds=50, learning_rate=0.1):
        self.n_rounds, self.nu = n_rounds, learning_rate; self.model_ = None

    def fit(self, Y_private, X_private_list, dropout_mode="none", dropout_rate=0.2, dropout_interval=5):
        process = psutil.Process(os.getpid()); start_mem = process.memory_info().rss
        start_time = time.perf_counter()
        n_obs, n_basis = Y_private.shape
        d_hat = np.mean(Y_private, axis=0)[np.newaxis, :].repeat(n_obs, axis=0)
        comms_bytes = 0; initial_party_indices = list(range(len(X_private_list)))
        party_indices = list(initial_party_indices)
        
        for m in range(self.n_rounds):
            if dropout_mode == "permanent" and m == dropout_interval:
                n_to_keep = int(len(party_indices)*(1-dropout_rate)); party_indices = np.random.choice(party_indices,size=n_to_keep,replace=False)
            elif dropout_mode == "intermittent" and m % dropout_interval == 0:
                n_to_keep = int(len(initial_party_indices)*(1-dropout_rate)); party_indices = np.random.choice(initial_party_indices,size=n_to_keep,replace=False)

            residuals = Y_private - d_hat; comms_bytes += residuals.nbytes
            party_preds, party_mses = [], []
            for idx in party_indices:
                model = LinearRegression(fit_intercept=False).fit(X_private_list[idx], residuals)
                preds = model.predict(X_private_list[idx]); party_preds.append(preds); party_mses.append(np.mean((residuals - preds)**2)); comms_bytes += preds.nbytes
            
            if not party_preds: continue # No parties left
            d_hat += self.nu * party_preds[np.argmin(party_mses)]

        runtime = time.perf_counter() - start_time; peak_mem = (process.memory_info().rss - start_mem) / 1024**2
        self.model_ = d_hat
        return self, runtime, comms_bytes, peak_mem

    def predict(self, X_test_list=None): return self.model_
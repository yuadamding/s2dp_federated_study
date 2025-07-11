# src/simgen/data_factory.py
import numpy as np
import neurokit2 as nk

class DataFactory:
    def __init__(self, n_subjects=400, n_parties=10, duration=5, sampling_rate=100, signal_to_noise_ratio=5.0, error_dist='gaussian', df=3, amplitude_scale=1.0, seed=42):
        self.n_subjects, self.n_parties, self.duration, self.sampling_rate = n_subjects, n_parties, duration, sampling_rate
        self.signal_to_noise_ratio = signal_to_noise_ratio; self.amplitude_scale = amplitude_scale
        self.error_dist, self.df, self.seed = error_dist, df, seed
        self.n_signal_parties = 5 if n_parties >= 5 else n_parties
        self.time_s = np.linspace(0, duration, duration * sampling_rate); self.time_t = np.linspace(0, 1, 100)
        np.random.seed(seed)

    def _generate_regression_surfaces(self):
        s, t = np.meshgrid(self.time_s, self.time_t, indexing='ij')
        beta_surfaces = [ 3.0*np.exp(-((s-self.duration*0.3)**2/1.5)-((t-0.3)**2/0.1)), 1.5*np.sin(2*np.pi*s/self.duration)*np.cos(2*np.pi*t), 2.0*np.exp(-((s/self.duration-t)**2/0.05)) ]
        return beta_surfaces + [beta_surfaces[0]*0.5, beta_surfaces[1]*0.8]

    def generate_ecg_data(self):
        latent_risk = np.random.uniform(0, 1, self.n_subjects)
        X_data = np.zeros((self.n_subjects, self.n_parties, len(self.time_s)))
        for i in range(self.n_subjects):
            risk = latent_risk[i]
            # Signal Parties
            X_data[i,0,:]=nk.ecg_simulate(duration=self.duration,sampling_rate=self.sampling_rate,heart_rate=60+30*risk,random_state=self.seed+i*10+1) * self.amplitude_scale
            for k in range(1, self.n_signal_parties): X_data[i,k,:] = nk.ecg_simulate(duration=self.duration,sampling_rate=self.sampling_rate,heart_rate=70+k*5,random_state=self.seed+i*10+k+1) * self.amplitude_scale
            # Noise Parties
            for j in range(self.n_signal_parties, self.n_parties): X_data[i,j,:] = nk.ecg_simulate(duration=self.duration,sampling_rate=self.sampling_rate,heart_rate=np.random.uniform(60,90),random_state=self.seed+i*10+j+1) * self.amplitude_scale
        
        beta_surfaces = self._generate_regression_surfaces()
        Y_truth = np.zeros((self.n_subjects, len(self.time_t)))
        for i in range(self.n_subjects):
            for k in range(self.n_signal_parties): Y_truth[i,:] += np.trapz(X_data[i,k,:,np.newaxis]*beta_surfaces[k],axis=0,dx=self.time_s[1]-self.time_s[0])
        
        signal_var = np.var(Y_truth)
        noise_var = signal_var / self.signal_to_noise_ratio if self.signal_to_noise_ratio > 0 else 0
        
        if self.error_dist == 'student_t':
            error_scale = np.sqrt(noise_var * (self.df - 2) / self.df) if self.df > 2 else np.sqrt(noise_var)
            error_term = error_scale * np.random.standard_t(self.df, Y_truth.shape)
        else: # Default to Gaussian
            error_term = np.random.normal(0, np.sqrt(noise_var), Y_truth.shape)
            
        Y_noisy = Y_truth + error_term
        return X_data, Y_noisy, Y_truth, latent_risk, self.time_s, self.time_t
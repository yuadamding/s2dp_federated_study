import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

def perform_mia_attack(model_coefs, basis_t_matrix, Y_truth_members, Y_truth_non_members):
    model_funcs = basis_t_matrix @ model_coefs.T
    loss_members = np.mean((Y_truth_members.T - model_funcs)**2, axis=0)
    loss_non_members = np.mean((Y_truth_non_members.T - model_funcs)**2, axis=0)
    scores = np.concatenate([-loss_members, -loss_non_members]); labels = np.concatenate([np.ones_like(loss_members), np.zeros_like(loss_non_members)])
    fpr, tpr, _ = roc_curve(labels, scores); return fpr, tpr, auc(fpr, tpr)

def perform_aia_attack(model_coefs_train, latent_risk_train, model_coefs_test, latent_risk_test):
    y_attack_train = (latent_risk_train > 0.5).astype(int); y_attack_test = (latent_risk_test > 0.5).astype(int)
    if len(np.unique(y_attack_train)) < 2 or len(np.unique(y_attack_test)) < 2: return 0, 0, 0.5
    attacker = LogisticRegression(penalty=None, solver='lbfgs').fit(model_coefs_train, y_attack_train)
    pred_probs = attacker.predict_proba(model_coefs_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_attack_test, pred_probs); return fpr, tpr, auc(fpr, tpr)
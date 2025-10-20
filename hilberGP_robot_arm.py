import numpy as np
from scipy.io import loadmat
from scipy.linalg import cholesky, solve
from scipy.sparse import diags
from Hilbert_GP_functions import basisfunctionsSE

# --- Load dataset ---
data = loadmat('/Users/hakilic/Downloads/robot_arm_data/inverse_identification_without_raw_data.mat')

X_train = data['u_train'].squeeze().T
X_test = data['u_test'].squeeze().T
y_train = data['y_train'].squeeze().T[:, 0]
y_test = data['y_test'].squeeze().T[:, 0]

# --- Hyperparameters ---
ell2 = 0.4706**2       # ℓ²
sigma_f2 = 2.8853**2   # σ_f²
sigma_n2 = 0.6200**2   # σ_n²
D = X_train.shape[1]

# --- Compute L per dimension ---
L_vec = np.ones(D) + 2 * np.sqrt(ell2)

# --- Budget ---
budget = X_train.shape[0]

# --- Basis functions ---
def colectofbasisfunc(budget, X, ell2, sigma_f2, L_vec):
    ell = np.sqrt(ell2)
    sigma_f = np.sqrt(sigma_f2)
    ΦR, ΛR, ind = basisfunctionsSE(budget, X, ell, sigma_f, L_vec)
    return ΦR, ΛR

ΦR, ΛR = colectofbasisfunc(budget, X_train, ell2, sigma_f2, L_vec)
ΦsR, ΛsR = colectofbasisfunc(budget, X_test, ell2, sigma_f2, L_vec)

# --- Compute predictive posterior ---
Λ_diag = np.diag(ΛR)
Z = sigma_n2 * np.diag(1 / Λ_diag) + ΦR.T @ ΦR

# Cholesky decomposition
Lchol = cholesky(Z, lower=True)

# Solve for mstar
temp = solve(Lchol, ΦR.T @ y_train, lower=True)
mstar = ΦsR.T @ solve(Lchol.T, temp, lower=False)

# Solve for vstar
temp2 = solve(Lchol, ΦsR, lower=True)
vstar = sigma_n2 * ΦsR.T @ solve(Lchol.T, temp2, lower=False)

# --- Optional: print shapes ---
print("ΦR shape:", ΦR.shape)
print("ΛR shape:", ΛR.shape)
print("mstar shape:", mstar.shape)
print("vstar shape:", vstar.shape)

import numpy as np
from scipy.io import loadmat
from Hilbert_GP_functions import fullGP, SE

# --- Load the .mat file ---
data = loadmat('/Users/hakilic/Downloads/robot_arm_data/inverse_identification_without_raw_data.mat')

# Ensure proper shapes for X

X_train = data['u_train'].squeeze().T
X_test = data['u_test'].squeeze().T
y_train = data['y_train'].squeeze().T[:,0]
y_test = data['y_test'].squeeze().T[:,0]


# --- Hyperparameters ---
lengthscale = 0.4706
sigma_f2 = 2.8853
sigma_n2 = 0.6200
# Keep hyp for covSE as only ell and sigma_f
hyp_cov = [lengthscale, sigma_f2, sigma_n2]

# Compute K_train with noise added
K_train = SE(X_train, X_train, hyp_cov[0], hyp_cov[1]) + sigma_n2 * np.eye(X_train.shape[0])

# Pass only hyp_cov to fullGP if it expects only covariance hyperparameters
mstar, vstar = fullGP(K_train, X_train, X_test, y_train, hyp_cov)


rmse = np.sqrt(np.mean((mstar - y_test) ** 2))

print(rmse)

# nll using raw values
nll = 0.5 * np.log(2 * np.pi * mstar**2) + 0.5 * (
    (y_test - mstar) ** 2
) / (vstar**2)
print("NLL:", np.mean(nll))

# --- Print results ---
print("mstar shape:", mstar.shape)
print("vstar shape:", vstar.shape)

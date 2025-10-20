import numpy as np
from scipy.sparse import diags, kron
from scipy.linalg import cholesky, solve
from scipy.spatial.distance import cdist

def basisfunctionsSE(budget, X, l, sigma_f, L):
    """
    Computes Φ and Λ such that Φ * sqrt(Λ) * sqrt(Λ) * Φ.T approximates the SE kernel matrix K.
    
    Parameters:
        budget (int): Number of basis functions to keep.
        X (np.ndarray): N x D matrix of input points.
        l (float): Lengthscale.
        sigma_f (float): Kernel variance.
        L (list or np.ndarray): Domain lengths for each dimension.
        
    Returns:
        ΦR (np.ndarray): N x budget matrix of basis functions.
        ΛR (np.ndarray): budget x budget diagonal matrix of eigenvalues.
        ind (np.ndarray): D x budget matrix of indices.
    """
    N, D = X.shape
    M = int(np.ceil(budget ** (1 / D)))

    # Start with scalar 1 for Kron product
    Lambda = np.array([1.0])

    # Construct the Kronecker product of eigenvalues
    for d in reversed(range(D)):
        w = np.arange(1, M+1)  # 1..M
        # Compute the diagonal entries for this dimension
        diag_vals = sigma_f**(1/D) * np.sqrt(2*np.pi*l) * np.exp(-l/2 * ((np.pi * w) / (2*L[d]))**2)
        Lambda = kron(Lambda, diags(diag_vals)).toarray()  # Convert sparse to dense

    # Select the top 'budget' eigenvalues
    p = np.argsort(np.diag(Lambda))[::-1][:budget]
    LambdaR = Lambda[np.ix_(p, p)]

    # Cartesian product indices
    all_indices = np.array(np.meshgrid(*[np.arange(1, M+1) for _ in range(D)], indexing='ij'))
    all_indices = all_indices.reshape(D, -1)
    ind = all_indices[:, p].astype(float)

    # Construct the basis function matrix ΦR
    PhiR = np.ones((N, budget))
    for d in range(D):
        PhiR *= (1 / np.sqrt(L[d])) * np.sin(np.pi * ((X[:, d:d+1] + L[d]) / (2 * L[d])) * ind[d, :])

    return PhiR, LambdaR, ind

def SE(X1, X2, lengthscale, sigma_f):
    """
    Squared Exponential (RBF) kernel.

    Parameters:
    X1 : ndarray of shape (n_samples1, n_features)
    X2 : ndarray of shape (n_samples2, n_features)
    lengthscale : float, the lengthscale parameter

    Returns:
    K : ndarray of shape (n_samples1, n_samples2), the kernel matrix
    """
    lengthscale = np.array(lengthscale, dtype=float)
    # Compute pairwise Euclidean distances
    dists = cdist(X1, X2, metric='euclidean')
    # Compute the squared exponential kernel
    K = sigma_f*np.exp(-0.5 * (dists ** 2) / (lengthscale ** 2))
    return K

def fullGP(K, X, Xstar, y, hyp):
    """
    Full Gaussian Process prediction.

    Parameters
    ----------
    K : ndarray, shape (N, N)
        Covariance matrix on training data
    X : ndarray, shape (N, D)
        Training inputs
    Xstar : ndarray, shape (N*, D)
        Test inputs
    y : ndarray, shape (N,)
        Training targets
    hyp : list or ndarray
        Hyperparameters [ell, sigma_f, sigma_n]

    Returns
    -------
    mstar : ndarray, shape (N*,)
        Predictive mean
    vstar : ndarray, shape (N*,)
        Predictive variance (diagonal of covariance)
    """
    sigma_n = hyp[2]  # observation noise
    N = X.shape[0]

    # Add small jitter for numerical stability
    jitter = np.sqrt(np.finfo(float).eps)
    L = cholesky(K + (sigma_n + jitter) * np.eye(N), lower=True)

    # Compute cross-covariance and test covariance
    Ks = SE(Xstar, X, hyp[0], hyp[1])
    Kss = SE(Xstar, Xstar, hyp[0], hyp[1])

    # Solve for alpha
    alpha = solve(L.T, solve(L, y, lower=True))

    # Predictive mean
    mstar = Ks @ alpha

    # Predictive covariance
    v = solve(L, Ks.T, lower=True)
    Pstar = Kss - v.T @ v

    # Return diagonal as predictive variance
    vstar = np.diag(Pstar)

    return mstar, vstar
"""Volterra feature map and small linear-algebra helpers used by the model."""

import numpy as np


def volterra_basis(u, M):
    """Lagged-input matrix of the Volterra model.

    Row n is [1, u(n), u(n-1), ..., u(n-M+1)] (inputs before time 0 are zero),
    so for a single input u (N x 1) the result is N x (M + 1).
    """
    N, p = u.shape
    u_padded = np.vstack([np.zeros((M - 1, p)), u])
    U = np.ones((N, p * M + 1))
    for j in range(1, M + 1):
        U[:, 1 + (j - 1) * p : 1 + j * p] = u_padded[M - j : N + M - j]
    return U


def dotkron(L, R):
    """Row-wise Kronecker product: row n is kron(R[n], L[n])."""
    N, c1 = L.shape
    c2 = R.shape[1]
    if R.shape[0] != N:
        raise ValueError("Matrices should have equal rows!")
    return (R[:, :, None] * L[:, None, :]).reshape(N, c1 * c2)


def columnwise_kronecker(A, B):
    """Column-wise Kronecker product: column i is kron(A[:, i], B[:, i])."""
    if A.shape[1] != B.shape[1]:
        raise ValueError("Number of columns in A and B must be the same.")
    m, n = A.shape
    p = B.shape[0]
    return (A[:, None, :] * B[None, :, :]).reshape(m * p, n)


def phi_cov(Phi, V, R):
    """Per-sample second-moment term of one factor's covariance.

    Row n holds the R x R block phi_n^T V_(r, r') phi_n, flattened to R^2 entries.
    """
    I = Phi.shape[1]
    V = np.reshape(V, (I, R, I, R), order="F").transpose(0, 2, 1, 3)
    return dotkron(Phi, Phi) @ np.reshape(V, (I**2, R**2))


def gram_and_rhs(A, B, y):
    """Return K^T K and K^T y for K = dotkron(A, B)."""
    K = dotkron(A, B)
    return K.T @ K, K.T @ y.reshape(-1, 1)


def safe_division(X, A, epsilon=1):
    """X / A, with zeros in A replaced by epsilon."""
    return X / np.where(A == 0, epsilon, A)


def safelog(x):
    """log(x) with x clipped to [1e-10, 1e10]."""
    return np.log(np.clip(x, 1e-10, 1e10))

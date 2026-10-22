"""BTN-V: Bayesian tensor network Volterra model.

Model: y(n) = sum_r prod_{d=1..D} phi(n)^T w_r^(d) + e(n), where phi(n) is the
lagged-input vector [1, u(n), ..., u(n-M+1)] and D is the Volterra degree.
The Volterra kernel is a CP (CPD) tensor with D factor matrices W^(d) of size
(M+1) x R, all built from the same phi(n).

Priors (all with Gamma hyperpriors):
  - lambda_r: one precision per rank component (shared by all factors) -> prunes the rank,
  - delta_m:  one precision per lag (shared by all factors)            -> learns fading memory,
  - tau:      noise precision.

`train` runs mean-field variational inference (a Bayesian ALS sweep over the
factors, then closed-form updates of delta and lambda); `predict` returns the
predictive mean and standard deviation.
"""

import time

import numpy as np
from tqdm import trange

from .utils import (
    volterra_basis,
    dotkron,
    columnwise_kronecker,
    phi_cov,
    gram_and_rhs,
    safe_division,
    safelog,
)


class BTNV:
    def train(
        self,
        u,
        y,
        M,
        D,
        R,
        a0,
        b0,
        c0,
        d0,
        g0,
        h0,
        max_iter=100,
        seed=0,
        lambda_update=True,
        delta_update=True,
        precision_update=True,
        prune_rank=True,
        rank_tol=1e-5,
        lower_bound_tol=1e-4,
        verbose=True,
    ):
        """Fit the model.

        u, y      : training input (N, 1) and output (N,)
        M, D, R   : memory length, Volterra degree and initial (maximum) rank
        a0, b0    : Gamma prior of the noise precision tau
        c0, d0    : Gamma prior of the rank precisions lambda
        g0, h0    : Gamma prior of the lag precisions delta
        rank_tol  : rank components explaining less than this % of the variance are pruned

        During training tau is kept at a0 / b0; its posterior is computed once
        after the last iteration (if precision_update) and used for prediction.

        After training, the model stores:
          rank, rank_history, lower_bound, delta, lambda_R, W_D (posterior means)
          train_time  wall-clock time of the training loop in seconds
        """
        np.random.seed(seed)
        I = M + 1
        N = u.shape[0]

        c0 = c0 * np.ones(R)
        d0 = d0 * np.ones(R)
        g0 = g0 * np.ones(I)
        h0 = h0 * np.ones(I)

        tau = a0 / b0
        c_N, d_N = c0, d0
        g_N, h_N = g0, h0
        lambda_R = c0 / d0
        delta = g0 / h0

        # Factor means and covariances
        W_D = [np.random.randn(I, R) for _ in range(D)]
        WSigma_D = [0.1 * np.kron(np.eye(R), np.eye(I)) for _ in range(D)]

        Phi = volterra_basis(u, M)

        LB = np.zeros(max_iter)
        rank_history = []

        # Hadamard products over all factors of the second moments and of the means
        hadamard_V = np.ones((N, R**2))
        hadamard_mean = np.ones((N, R))
        for d in range(D):
            hadamard_V = hadamard_V * phi_cov(Phi, WSigma_D[d], R)
            hadamard_mean = hadamard_mean * (Phi @ W_D[d])

        start_time = time.time()
        pbar = trange(max_iter, desc="Training", leave=True, disable=not verbose)
        for it in pbar:

            # ---- Factor updates q(W^(d)) ----
            for d in range(D):
                # remove factor d from the running products
                hadamard_V = safe_division(hadamard_V, phi_cov(Phi, WSigma_D[d], R))
                hadamard_mean = safe_division(hadamard_mean, Phi @ W_D[d])

                # E[G G^T] = mean part (cc) + covariance part (V_temp)
                PhiPhi_V = dotkron(Phi, Phi).T @ hadamard_V
                cc, cy = gram_and_rhs(Phi, hadamard_mean, y)
                V_temp = np.reshape(
                    np.transpose(np.reshape(PhiPhi_V, (I, I, R, R), order="F"), axes=(0, 2, 1, 3)),
                    (I * R, I * R),
                    order="F",
                )

                A = tau * (cc + V_temp) + np.kron(lambda_R * np.eye(R), delta * np.eye(I))
                try:
                    WSigma_D[d] = np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    WSigma_D[d] = np.linalg.pinv(A)
                W_D[d] = np.reshape(tau * WSigma_D[d] @ cy, (I, R), order="F")

                # add the updated factor back
                hadamard_V = hadamard_V * phi_cov(Phi, WSigma_D[d], R)
                hadamard_mean = hadamard_mean * (Phi @ W_D[d])

            # ---- Lag precisions q(delta) ----
            if delta_update:
                g_N = (D * R * 0.5) + g0
                h_N = 0
                for d in range(D):
                    mtemp = np.diag(W_D[d] @ (lambda_R * np.eye(R)) @ W_D[d].T)
                    vtemp = np.diag(
                        np.reshape(
                            WSigma_D[d].reshape(I, R, I, R).transpose(0, 2, 1, 3).reshape(I**2, R**2)
                            @ (lambda_R * np.eye(R)).ravel(order="F"),
                            (I, I),
                        )
                    )
                    h_N += mtemp + vtemp
                h_N = h0 + (0.5 * h_N)
                delta = g_N / h_N
                delta[delta < 1e-5] = 1e-5

            # ---- Rank precisions q(lambda) ----
            if lambda_update:
                c_N = (0.5 * D * I) + c0
                d_N = 0
                for d in range(D):
                    mtemp = np.diag(W_D[d].T @ (delta * np.eye(I)) @ W_D[d])
                    vtemp = np.diag(
                        np.reshape(
                            (delta * np.eye(I)).ravel(order="F").T
                            @ WSigma_D[d].reshape(I, R, I, R).transpose(0, 2, 1, 3).reshape(I**2, R**2),
                            (R, R),
                        )
                    )
                    d_N += mtemp + vtemp
                d_N = d0 + (0.5 * d_N)
                lambda_R = c_N / d_N
                lambda_R[lambda_R < 1e-5] = 1e-5

            # ---- Residual terms (tau itself stays fixed during training) ----
            residual = y - np.sum(hadamard_mean, axis=1)
            ss_error = np.dot(residual, residual)
            covariance = np.sum(np.sum(hadamard_V, axis=1))
            a_N, b_N = a0, b0
            tau = a_N / b_N

            # ---- Variational lower bound ----
            LB[it] = self._lower_bound(
                W_D, WSigma_D, lambda_R, delta, ss_error, a_N, b_N, b0, c_N, d_N, d0, g_N, h_N, h0, I, R
            )

            # ---- Rank pruning ----
            rankest = R
            if prune_rank:
                Wall = np.vstack(W_D)
                power = np.diag(Wall.T @ Wall)
                var_explained = power / np.sum(power) * 100
                rankest = np.sum(var_explained > rank_tol)
                if rankest == 0:
                    print("Rank becomes 0 !!!")
                    break
                if R != rankest:
                    removed = np.where(~(var_explained > rank_tol))[0]
                    lambda_R = np.delete(lambda_R, removed)
                    rows = np.r_[[np.arange(I * r, I * r + I) for r in removed]]
                    for d in range(D):
                        W_D[d] = np.delete(W_D[d], removed, axis=1)
                        WSigma_D[d] = np.delete(WSigma_D[d], rows, axis=1)
                        WSigma_D[d] = np.delete(WSigma_D[d], rows, axis=0)
                    hadamard_mean = np.delete(hadamard_mean, removed, axis=1)
                    # hadamard_V holds R x R blocks: drop every row/column of a removed component
                    hadamard_V = np.delete(
                        hadamard_V, [i * R + j for i in range(R) for j in range(R) if i in removed or j in removed], axis=1
                    )
                    c0 = c0[:rankest]
                    d0 = d0[:rankest]
                    R = rankest
            rank_history.append(R)

            # ---- Convergence check ----
            rel_change = (LB[it] - LB[it - 1]) / LB[2] if it > 2 else np.nan
            pbar.set_postfix({"RelChan": f"{rel_change:.2e}", "R": rankest})
            if it > 5 and abs(rel_change) < lower_bound_tol:
                if verbose:
                    print("======= Converged =======")
                break

        # Noise precision posterior, from the last iteration's residual
        if precision_update:
            a_N = a0 + (N / 2)
            b_N = b0 + (0.5 * (ss_error + covariance))
        self.train_time = time.time() - start_time

        # Store the posterior
        self.W_D, self.V, self.a, self.b = W_D, WSigma_D, a_N, b_N
        self.M = M
        self.rank = R
        self.rank_history = rank_history
        self.lower_bound = LB[:it]
        self.delta = delta
        self.lambda_R = lambda_R
        return self

    @staticmethod
    def _lower_bound(W_D, WSigma_D, lambda_R, delta, ss_error, a_N, b_N, b0, c_N, d_N, d0, g_N, h_N, h0, I, R):
        """Variational lower bound (up to constants); only used to check convergence."""
        t1 = -0.5 * (a_N / b_N) * ss_error
        t2 = np.zeros((I * R, I * R))
        for d in range(len(W_D)):
            w = W_D[d].ravel(order="F")
            t2 += np.kron(lambda_R * np.eye(R), delta * np.eye(I)) @ (np.outer(w, w) + WSigma_D[d])
        t2 = -0.5 * np.trace(t2)
        t3 = 0.5 * np.sum([safelog(np.linalg.slogdet(S)) for S in WSigma_D])
        t4 = np.sum(np.multiply(c_N, (1 - safelog(d_N) - (d0 / d_N))))
        t5 = np.sum(np.multiply(g_N, (1 - safelog(h_N) - (h0 / h_N))))
        t6 = a_N * (1 - safelog(b_N) - (b0 / b_N))
        return t1 + t2 + t3 + t4 + t5 + t6

    def predict(self, u):
        """Predictive mean and standard deviation for input sequence u (N, 1)."""
        Phi = volterra_basis(u, self.M)
        N, D = u.shape[0], len(self.W_D)
        R = self.W_D[0].shape[1]

        prod = np.ones((N, R))
        for d in range(D):
            prod = np.multiply(prod, Phi @ self.W_D[d])
        mean = np.sum(prod, axis=1)

        # Sum over factors of g_d^T Sigma_d g_d.
        # NOTE: kept exactly as in the code used for the paper. Here `others` is
        # built from single ROWS of Phi (rows 0..D-1, skipping row d) instead of
        # Phi @ W^(k) for every sample, so it is the same for all test points.
        quad = 0
        for d in range(D):
            W_others = [W for k, W in enumerate(self.W_D) if k != d]
            Phi_rows = [row for k, row in enumerate(Phi) if k != d]
            others = np.ones((N, R))
            for k in range(len(W_others)):
                others *= Phi_rows[k] @ W_others[k]
            g = columnwise_kronecker(Phi.T, others.T)
            quad += g.T @ self.V[d] @ g

        S = (2 * self.a / (2 * self.a - 2)) * ((self.b / self.a) + (quad / (D * self.M * R)))
        std = np.sqrt(np.diag(S))
        return mean, std

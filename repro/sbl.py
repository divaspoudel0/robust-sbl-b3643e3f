"""Joint ARD (heteroscedastic SBL) implementations per ICML 2026 paper
"Joint Model and Data Sparsification via the Marginal Likelihood".

Implements Algorithms 1-4 (EM, MacKay, l2-IRLS, l1-IRLS) plus homoscedastic
variants and the gradient-based updates, following the closed-form update
rules summarised in the paper's Table 5 / App. B and the practical safeguards
of App. D.1 (damping, clipping, convergence via log-parameter relative change).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, cho_factor

DEFAULT_EPS_MIN = 1e-3
DEFAULT_EPS_MAX = 1e6
DEFAULT_ETA_GAMMA = 2e-2
DEFAULT_ETA_LAMBDA = 1e-3
DEFAULT_TOL = 1e-6
DEFAULT_PATIENCE = 5
DEFAULT_MAXITER = 20000


def _chol_solve(Lc, B):
    """Solve (Lc Lc^T) Z = B for Z using the lower Cholesky factor Lc."""
    return cho_solve((Lc, True), B)


def _chol_factor(S):
    try:
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(S + np.eye(S.shape[0]) * 1e-9)


def _chol_logdet(Lc):
    return 2.0 * np.sum(np.log(np.diag(Lc)))


def _chol_diag_inv(Lc):
    """Diagonal of (Lc Lc^T)^{-1} via column-wise triangular solve."""
    d = Lc.shape[0]
    diag = np.empty(d)
    for j in range(d):
        e = np.zeros(d)
        e[j] = 1.0
        v = cho_solve((Lc, True), e)
        diag[j] = v[j]
    return diag


def ess_from_scores(scores: np.ndarray, eps: float = 1e-12) -> float:
    """Effective support size (normalised perplexity) from relevance scores.

    scores = 1/gamma (weight relevance) or 1/lambda (sample relevance).
    ESS = exp(H(p)) / m with p the normalised relevance mass.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    scores = np.clip(scores, eps, None)
    p = scores / scores.sum()
    h = -np.sum(p * np.log(p))
    return float(np.exp(h) / len(p))


class JointARD:
    """Joint model+data ARD via the heteroscedastic marginal likelihood.

    Model: y = X theta + eps, eps_i ~ N(0, lambda_i), theta_j ~ N(0, 1/gamma_j).
    Methods: 'em', 'mackay', 'l2_irls', 'l1_irls'.
    Homoscedastic counterpart obtained with heteroscedastic=False (single lambda).
    """

    def __init__(self, method="em", heteroscedastic=True,
                 damp_gamma=DEFAULT_ETA_GAMMA, damp_lambda=DEFAULT_ETA_LAMBDA,
                 eps_min=DEFAULT_EPS_MIN, eps_max=DEFAULT_EPS_MAX,
                 tol=DEFAULT_TOL, max_iter=DEFAULT_MAXITER, patience=DEFAULT_PATIENCE,
                 warmup_steps=100, lambda_freq=3, seed=0):
        assert method in ("em", "mackay", "l2_irls", "l1_irls")
        self.method = method
        self.heteroscedastic = heteroscedastic
        self.damp_gamma = damp_gamma
        self.damp_lambda = damp_lambda
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.tol = tol
        self.max_iter = max_iter
        self.patience = patience
        self.warmup_steps = warmup_steps  # update only gamma first (App. D.1)
        self.lambda_freq = lambda_freq      # update lambda every K outer iters
        self.seed = seed

    def _clip(self, x):
        return np.clip(x, self.eps_min, self.eps_max)

    def fit(self, X, y, gamma0=None, lambda0=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        rng = np.random.default_rng(self.seed)

        if gamma0 is None:
            gamma = np.full(d, 1.0 / max(np.var(y), 1e-6)) if np.var(y) > 0 else np.ones(d)
        else:
            gamma = np.asarray(gamma0, dtype=float).copy()
        if self.heteroscedastic:
            if lambda0 is None:
                lam = np.full(n, max(np.var(y), 1e-6) * 0.1 if np.var(y) > 0 else 0.1)
            else:
                lam = np.asarray(lambda0, dtype=float).copy()
        else:
            if lambda0 is None:
                lam = np.array([max(np.var(y), 1e-6) * 0.1 if np.var(y) > 0 else 0.1])
            else:
                lam = np.asarray(lambda0, dtype=float).ravel()[:1].copy()

        obj_hist = []
        converged = False
        t = 0
        stall = 0
        prev_g = np.log(self._clip(gamma))
        prev_l = np.log(self._clip(lam if self.heteroscedastic else np.full(n, lam[0])))
        I = np.eye(d)
        while t < self.max_iter:
            t += 1
            L = lam if self.heteroscedastic else np.full(n, lam[0])
            G = gamma
            invL = 1.0 / L
            # Sigma_theta^{-1} = G + X^T diag(1/L) X ; Cholesky + triangular solves
            Xt = X.T
            A = Xt * invL  # d x n
            Sinv = np.diag(G) + A @ X
            Lc = _chol_factor(Sinv)
            tilded = A @ y
            mu = _chol_solve(Lc, tilded)          # posterior mean
            Stheta = _chol_solve(Lc, I)           # full d x d covariance
            diag_S = np.diag(Stheta)
            XSt = X @ Stheta                      # X Sigma_theta, n x d
            q_i = np.sum(XSt * XSt, axis=1)       # diag(X Sigma_theta X^T)

            # dual objective for diagnostics
            logdet_Sinv = _chol_logdet(Lc)
            logdet_G = np.sum(np.log(self._clip(G)))
            logdet_L = np.sum(np.log(self._clip(L)))
            quad = y @ (invL * y) - tilded @ mu
            obj = logdet_Sinv - logdet_G + logdet_L + quad
            obj_hist.append(obj)

            r = y - X @ mu

            if self.method == "em":
                gamma_new = 1.0 / (mu ** 2 + diag_S)
            elif self.method == "mackay":
                gamma_new = (1.0 - gamma * diag_S) / np.maximum(mu ** 2, self.eps_min)
            elif self.method == "l2_irls":
                PiX = self._xpi_x(X, G, invL)  # diag(X^T Pi_y X)
                gamma_new = 1.0 / (mu ** 2 + 1.0 / gamma - PiX / (gamma ** 2))
            elif self.method == "l1_irls":
                PiX = self._xpi_x(X, G, invL)
                gamma_new = np.sqrt(PiX) / np.maximum(np.abs(mu), self.eps_min)
            gamma_new = self._clip(gamma_new)
            gamma_new = np.where(np.isfinite(gamma_new), gamma_new, self.eps_max)
            gamma = (1.0 - self.damp_gamma) * gamma + self.damp_gamma * gamma_new

            if self.heteroscedastic:
                if self.method == "em":
                    lam_new = r ** 2 + q_i
                elif self.method == "mackay":
                    lam_new = r ** 2 + q_i
                elif self.method == "l2_irls":
                    diag_Pi = self._diag_pi(X, invL, q_i)
                    lam_new = r ** 2 + lam - (lam ** 2) * diag_Pi
                elif self.method == "l1_irls":
                    diag_Pi = self._diag_pi(X, invL, q_i)
                    lam_new = np.abs(r) / np.maximum(np.sqrt(diag_Pi), self.eps_min)
                lam_new = self._clip(lam_new)
                if t > self.warmup_steps and t % self.lambda_freq == 0:
                    lam = (1.0 - self.damp_lambda) * lam + self.damp_lambda * lam_new
            else:
                if self.method == "em":
                    lam_new = (np.sum(r ** 2) + np.sum(q_i)) / n
                elif self.method == "mackay":
                    lam_new = np.sum(r ** 2) / max(n - np.sum(1.0 - gamma * diag_S), self.eps_min)
                elif self.method == "l2_irls":
                    PiX = self._xpi_x(X, G, invL)
                    trace_Pi = np.sum(self._diag_pi(X, invL, q_i))
                    lam_new = (np.sum(r ** 2) + lam[0] - (lam[0] ** 2) * trace_Pi) / n
                elif self.method == "l1_irls":
                    PiX = self._xpi_x(X, G, invL)
                    trace_Pi = np.sum(self._diag_pi(X, invL, q_i))
                    lam_new = np.sqrt(np.sum(r ** 2) / max(trace_Pi, self.eps_min))
                lam_new = np.clip(lam_new, self.eps_min, self.eps_max)
                if t > self.warmup_steps and t % self.lambda_freq == 0:
                    lam = np.array([(1.0 - self.damp_lambda) * lam[0] + self.damp_lambda * lam_new])

            cur_g = np.log(self._clip(gamma))
            cur_l = np.log(self._clip(lam if self.heteroscedastic else np.full(n, lam[0])))
            dg = np.max(np.abs(cur_g - prev_g)) / (1.0 + np.max(np.abs(cur_g)))
            dl = np.max(np.abs(cur_l - prev_l)) / (1.0 + np.max(np.abs(cur_l)))
            prev_g, prev_l = cur_g, cur_l
            if max(dg, dl) < self.tol:
                stall += 1
                if stall >= self.patience:
                    converged = True
                    break
            else:
                stall = 0

        # final posterior (unchanged code path)
        G = gamma
        L = lam if self.heteroscedastic else np.full(n, lam[0])
        invL = 1.0 / L
        Xt = X.T
        A = Xt * invL
        Sinv = np.diag(G) + A @ X
        Lc = _chol_factor(Sinv)
        mu = _chol_solve(Lc, A @ y)
        Stheta = _chol_solve(Lc, I)

        self.gamma_ = gamma
        self.lambda_ = lam if self.heteroscedastic else np.full(n, lam[0])
        self.mu_ = mu
        self.Sigma_ = Stheta
        self.n_iter_ = t
        self.converged_ = converged
        self.obj_hist_ = np.array(obj_hist)
        return self

    def _xpi_x(self, X, G, invL):
        """diag(X^T Pi_y X) via Woodbury: Pi_y = invL - invL X (G + X^T invL X)^{-1} X^T invL.
        Then X^T Pi_y X = B - B Sigma_theta B with B = X^T invL X (d x d).
        """
        d = X.shape[1]
        A = X.T * invL  # d x n = X^T invL
        B = A @ X       # d x d = X^T invL X
        Sinv = np.diag(G) + B
        Lc = _chol_factor(Sinv)
        MB = _chol_solve(Lc, B)          # Sigma_theta B
        diagB = np.diag(B)
        diagBMB = np.diag(B @ MB)
        return diagB - diagBMB

    def _diag_pi(self, X, invL, q_i):
        """diag(Pi_y) = invL - invL^2 * diag(X Sigma_theta X^T) (Woodbury)."""
        return invL - invL ** 2 * q_i

    def predict(self, X_test, lam_base="mean"):
        """Posterior predictive mean and variance.

        mu* = x*^T mu_theta ; var* = lambda_b + x*^T Sigma_theta x*
        lambda_b defaults to mean(lambda) (App. D / Sec 5).
        """
        Xt = np.asarray(X_test, dtype=float)
        mu = Xt @ self.mu_
        var = np.sum((Xt @ self.Sigma_) * Xt, axis=1)
        if lam_base == "mean":
            lam_b = np.mean(self.lambda_)
        elif lam_base == "median":
            lam_b = np.median(self.lambda_)
        else:
            lam_b = lam_base
        return mu, var + lam_b


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def nll(y_true, mu, var, eps=1e-12):
    y_true = np.asarray(y_true)
    mu = np.asarray(mu)
    var = np.clip(np.asarray(var), eps, None)
    return float(0.5 * np.mean(np.log(2 * np.pi * var) + (y_true - mu) ** 2 / var))

"""Baseline models: OLS, Ridge, exact GP, Student-t EM, Huber (App. D.2)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from .sbl import rmse, nll


class OLSBaseline:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        self.train_var_ = np.var(y - self.model.predict(X))
        return self

    def predict(self, X):
        mu = self.model.predict(X)
        return mu, np.full(len(X), self.train_var_)


class RidgeBaseline:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.alpha = alpha

    def fit(self, X, y):
        self.model.fit(X, y)
        self.train_var_ = np.var(y - self.model.predict(X))
        return self

    def predict(self, X):
        mu = self.model.predict(X)
        return mu, np.full(len(X), self.train_var_)


class HuberBaseline:
    def __init__(self, epsilon=1.35):
        self.model = HuberRegressor(epsilon=epsilon, alpha=1e-4)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.train_var_ = np.var(y - self.model.predict(X))
        return self

    def predict(self, X):
        mu = self.model.predict(X)
        return mu, np.full(len(X), self.train_var_)


class GPBaseline:
    """Exact RBF GP with optimised lengthscale (reference on clean data)."""

    def __init__(self, seed=0):
        self.kernel = RBF()
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-4,
                                              normalize_y=True, random_state=seed,
                                              optimizer="fmin_l_bfgs_b", n_restarts_optimizer=2)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        mu, std = self.model.predict(X, return_std=True)
        return mu, std ** 2


class StudentTBaseline:
    """Robust regression with Student-t likelihood via EM-style iterative steps
    with map-back to Gaussian variance (App. D.2 description)."""

    def __init__(self, nu=4.0, n_iter=200, alpha=1e-6, seed=0):
        self.nu = nu
        self.n_iter = n_iter
        self.alpha = alpha
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        Xt = np.eye(d)
        # ridge-type closed form with sample weights w_i
        theta = np.zeros(d)
        lam = np.var(y)
        for it in range(self.n_iter):
            r = y - X @ theta
            # posterior weights from t distribution (EM)
            w = (self.nu + 1.0) / (self.nu + r ** 2 / lam)
            # weighted ridge
            A = (X * w[:, None]).T @ X + self.alpha * np.eye(d)
            b = (X * w[:, None]).T @ y
            try:
                theta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                A += np.eye(d) * 1e-9
                theta = np.linalg.solve(A, b)
            r = y - X @ theta
            lam = np.sum(w * r ** 2) / max(n - d, 1)
        self.theta_ = theta
        self.lam_ = lam
        return self

    def predict(self, X):
        mu = np.asarray(X) @ self.theta_
        return mu, np.full(len(X), self.lam_)

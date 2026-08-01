"""Data loading, random Fourier features, and contamination protocol for the
joint ARD reproduction (App. D.2 of the paper)."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.environ.get(
    "REPRO_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)


def load_energy():
    df = pd.read_excel(f"{DATA_DIR}/energy/ENB2012_data.xlsx", header=0)
    X = df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values.astype(float)
    y = df["Y1"].values.astype(float)  # first target only (paper: 2 targets, use first)
    return X, y


def load_carbon():
    df = pd.read_csv(f"{DATA_DIR}/carbon/carbon_nanotubes.csv", sep=";", decimal=",")
    X = df[["Chiral indice n", "Chiral indice m", "Initial atomic coordinate u",
            "Initial atomic coordinate v", "Initial atomic coordinate w"]].values.astype(float)
    y = df["Calculated atomic coordinates u'"].values.astype(float)  # first target
    return X, y


def load_protein():
    df = pd.read_csv(f"{DATA_DIR}/protein/CASP.csv")
    y = df["RMSD"].values.astype(float)
    X = df[["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]].values.astype(float)
    return X, y


def load_boston():
    """Boston housing from UCI via scikit-learn's fetch (removed from sklearn);
    use the OpenML/raw UCI version instead."""
    df = pd.read_csv(f"{DATA_DIR}/boston/housing.data", sep=r"\s+", header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)
    return X, y


class _RFFMap:
    """Standardise features then approximate an RBF kernel (gamma='scale') with
    random Fourier features (App. D.2). Fit on the training split only."""

    def __init__(self, d_out=256, seed=0):
        self.d_out = d_out
        self.seed = seed

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        self.sampler = RBFSampler(gamma="scale", n_components=self.d_out, random_state=self.seed)
        self.sampler.fit(self.scaler.transform(X))
        return self

    def transform(self, X):
        return self.sampler.transform(self.scaler.transform(X)).astype(float)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def random_fourier_features(X, d_out=256, gamma=None, seed=0, X_ref=None):
    """Random Fourier features approximating an RBF kernel (App. D.2).

    ``X_ref`` (when given) is used to fit the standardisation/kernel bandwidth
    (training split); ``X`` is then transformed. This keeps the map well
    conditioned (cond(Phi^T Phi) ~ 1e9) unlike a naive median-distance scheme.
    """
    if X_ref is None:
        X_ref = X
    mapper = _RFFMap(d_out=d_out, seed=seed).fit(X_ref)
    return mapper.transform(X)


def rbf_kernel_matrix(X, gamma=None, X_ref=None, seed=0):
    """Exact RBF kernel matrix (for the RVM / kernel regression experiment)."""
    rng = np.random.default_rng(seed)
    if X_ref is None:
        X_ref = X
    if gamma is None:
        n_ref = min(X_ref.shape[0], 1000)
        sub = X_ref[:n_ref]
        d2 = []
        for _ in range(3):
            idx = rng.choice(sub.shape[0], size=min(200, sub.shape[0]), replace=False)
            i2 = rng.choice(sub.shape[0], size=min(200, sub.shape[0]), replace=False)
            d2.append(np.median(np.sum((sub[idx] - sub[i2]) ** 2, axis=1)))
        gamma = 1.0 / (2.0 * np.mean(d2) + 1e-12)
    D2 = np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * X @ X.T
    return np.exp(-gamma * D2).astype(float)


def rbf_kernel_cross(X_tr, X_te, gamma):
    D2 = np.sum(X_tr ** 2, axis=1)[:, None] + np.sum(X_te ** 2, axis=1)[None, :] - 2 * X_tr @ X_te.T
    return np.exp(-gamma * D2).astype(float)


def contaminate_targets(y, rho=0.1, amp=3.0, seed=0, return_mask=True):
    """Tabular contamination protocol (App. D.2): sign flip with heterogeneous
    magnitude around amp: y~i = yi + a*si*delta_i, delta_i ~ N(1, 0.25^2)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    k = int(np.floor(rho * n))
    idx = rng.choice(n, size=k, replace=False)
    si = rng.choice([-1.0, 1.0], size=k)
    delta = rng.normal(1.0, 0.25, size=k)
    yc = y.copy()
    yc[idx] = yc[idx] + amp * si * delta
    if return_mask:
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        return yc, mask
    return yc


def split_data(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    ntest = int(np.floor(test_frac * n))
    test_idx = idx[:ntest]
    train_idx = idx[ntest:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def cap_train(X_tr, y_tr, nmax=2000, seed=0):
    """Cap large training splits to nmax (App. D.2)."""
    n = X_tr.shape[0]
    if n <= nmax:
        return X_tr, y_tr
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=nmax, replace=False)
    return X_tr[idx], y_tr[idx]

"""Claim 1 & 2 verification: objective structure, dual equivalence,
outlier downweighting, closed-form updates across methods."""

import sys
import numpy as np

sys.path.insert(0, ".")
from repro.sbl import JointARD, ess_from_scores, rmse, nll
from repro.data import split_data, contaminate_targets


def primal_dual_audit():
    """Numerical audit: primal (Eq.3) == dual (Eq.4) objective."""
    np.random.seed(0)
    n, d = 100, 20
    X = np.random.randn(n, d)
    G = np.abs(np.random.randn(d)) + 0.1
    lam = np.abs(np.random.randn(n)) + 0.1
    theta = np.random.randn(d) * 0.5
    y = X @ theta + np.random.randn(n) * np.sqrt(lam)

    Sigma_y = np.diag(lam) + X @ np.diag(1 / G) @ X.T
    sign, ld = np.linalg.slogdet(Sigma_y)
    primal = ld + y @ np.linalg.solve(Sigma_y, y)

    Stheta_inv = np.diag(G) + X.T @ np.diag(1 / lam) @ X
    sign2, ld2 = np.linalg.slogdet(Stheta_inv)
    Stheta = np.linalg.inv(Stheta_inv)
    ytilde = X.T @ (y / lam)
    dual = ld2 - np.sum(np.log(G)) + np.sum(np.log(lam)) + (
        np.sum(y * y / lam) - ytilde @ Stheta @ ytilde)
    return abs(primal - dual)


def outlier_downweighting_demo():
    """Synthetic demo (Sec 5.1 style): joint ARD flags outliers via lambda and
    prunes features via gamma, versus homoscedastic baseline."""
    np.random.seed(1)
    n, d = 500, 50
    X = np.random.randn(n, d)
    sparsity = 0.2
    S = np.random.choice(d, int(sparsity * d), replace=False)
    theta = np.zeros(d)
    theta[S] = np.random.randn(len(S))
    sigma = 0.2
    y = X @ theta + np.random.randn(n) * sigma
    rho, m = 0.2, 10.0
    O = np.random.choice(n, int(np.floor(rho * n)), replace=False)
    y[O] += np.random.randn(len(O)) * (m - 1) * sigma
    mask = np.zeros(n, dtype=bool)
    mask[O] = True

    results = {}
    for method in ["em", "mackay", "l2_irls", "l1_irls"]:
        het = JointARD(method=method, heteroscedastic=True, max_iter=1500, damp_lambda=2e-2)
        het.fit(X, y)
        homo = JointARD(method=method, heteroscedastic=False, max_iter=1500, damp_lambda=2e-2)
        homo.fit(X, y)
        lam = het.lambda_
        # top-k outlier recovery: fraction of true outliers in top |O| lambda ranks
        k = int(np.floor(rho * n))
        topk = np.argsort(lam)[-k:]
        recovery = mask[topk].mean()
        results[method] = {
            "lambda_outlier_mean": float(lam[mask].mean()),
            "lambda_inlier_mean": float(lam[~mask].mean()),
            "ess_theta_het": ess_from_scores(1 / het.gamma_) * 100,
            "ess_y_het": ess_from_scores(1 / het.lambda_) * 100,
            "ess_theta_homo": ess_from_scores(1 / homo.gamma_) * 100,
            "ess_y_homo": ess_from_scores(1 / homo.lambda_) * 100,
            "n_iter_het": het.n_iter_,
            "outlier_recovery_topk": float(recovery),
        }
    return results


def em_objective_monotonic():
    """Check the EM objective (primal eq.3) is non-increasing along iterations."""
    np.random.seed(2)
    n, d = 150, 25
    X = np.random.randn(n, d)
    S = np.random.choice(d, 8, replace=False)
    theta = np.zeros(d)
    theta[S] = np.random.randn(8)
    y = X @ theta + np.random.randn(n) * 0.1
    O = np.random.choice(n, 15, replace=False)
    y[O] += np.random.randn(15) * 5
    m = JointARD(method="em", heteroscedastic=True, max_iter=5000)
    m.fit(X, y)
    obj = m.obj_hist_
    diffs = np.diff(obj)
    return {"frac_decreasing": float((diffs <= 1e-8).mean()), "final": float(obj[-1])}


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "dual"):
        print("PRIMAL-DUAL MAX ABS DIFF:", primal_dual_audit())
    if what in ("all", "outlier"):
        res = outlier_downweighting_demo()
        for k, v in res.items():
            print(k, v)
    if what in ("all", "mono"):
        print("EM MONOTONIC:", em_objective_monotonic())

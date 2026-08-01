"""Claim 4: synthetic signal recovery (Fig 3) + RVM kernel regression (Fig 4).

- Synthetic: sparse linear data (n=500,d=50), weight/outlier top-k recovery
  for l2-IRLS hetero vs homo, scanning sparsity and noise / contamination.
- RVM: Boston with RBF kernel basis (d=n), l2-IRLS, vs contamination fraction.
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from repro.sbl import JointARD, ess_from_scores, rmse, nll
from repro.data import (load_boston, split_data, cap_train,
                        contaminate_targets, rbf_kernel_matrix,
                        rbf_kernel_cross)
from repro.baselines import OLSBaseline, RidgeBaseline, GPBaseline


def topk_recovery(scores, truth, k):
    """Fraction of true positives in the top-k ranks of `scores`."""
    k = min(k, len(scores))
    topk = np.argsort(scores)[-k:]
    return float(truth[topk].mean())


def synthetic_recovery(n_trials=5, n=500, d=50):
    """Sec 5.1: weight/outlier recovery, hetero vs homo (l2-IRLS)."""
    out = {"weight_recovery": {}, "outlier_recovery": {}, "ess_y": {}}
    # fix |S| = 0.2d, sigma=0.2, vary contamination (rho, m)
    for (rho, mf) in [(0.1, 10.0), (0.2, 10.0), (0.2, 3.0), (0.4, 10.0)]:
        wr_h = wr_o = 0.0
        or_h = or_o = 0.0
        essy_h = essy_o = 0.0
        for tr in range(n_trials):
            rng = np.random.default_rng(1000 + tr)
            S = rng.choice(d, int(0.2 * d), replace=False)
            theta = np.zeros(d)
            theta[S] = rng.normal(0, 1, len(S))
            X = rng.normal(0, 1, (n, d))
            sigma = 0.2
            y = X @ theta + rng.normal(0, sigma, n)
            O = rng.choice(n, int(np.floor(rho * n)), replace=False)
            y[O] += rng.normal(0, (mf - 1) * sigma, len(O))
            mask = np.zeros(n, bool)
            mask[O] = True
            for label, het in (("h", True), ("o", False)):
                m = JointARD(method="l2_irls", heteroscedastic=het,
                             max_iter=1500, damp_lambda=2e-2)
                m.fit(X, y)
                wrec = topk_recovery(1.0 / m.gamma_, np.isin(np.arange(d), S),
                                     len(S))
                orec = topk_recovery(1.0 / m.lambda_, mask, len(O))
                if het:
                    wr_h += wrec / n_trials
                    or_h += orec / n_trials
                    essy_h += ess_from_scores(1.0 / m.lambda_) / n_trials
                else:
                    wr_o += wrec / n_trials
                    or_o += orec / n_trials
                    essy_o += ess_from_scores(1.0 / m.lambda_) / n_trials
        key = f"rho={rho:.1f},m={mf:.0f}"
        out["weight_recovery"][key] = {"hetero": wr_h, "homo": wr_o}
        out["outlier_recovery"][key] = {"hetero": or_h, "homo": or_o}
        out["ess_y"][key] = {"hetero": essy_h, "homo": essy_o}
    return out


def rvm_boston(contamination=0.1, max_iter=1500, seed=0):
    """Boston RVM: RBF kernel basis (d=n), l2-IRLS hetero vs homo vs baselines."""
    X, y = load_boston()
    Xtr, ytr, Xte, yte = split_data(X, y, test_frac=0.2, seed=seed)
    ymean, ystd = ytr.mean(), ytr.std()
    ytr_s = (ytr - ymean) / ystd
    yte_s = (yte - ymean) / ystd
    if contamination > 0:
        yc_s, mask = contaminate_targets(ytr_s, rho=contamination, amp=3.0, seed=seed)
    else:
        yc_s, mask = ytr_s, np.zeros(len(ytr_s), bool)

    # RBF kernel basis (RVM): d = n. Median-distance bandwidth on training set.
    rng = np.random.default_rng(seed)
    sub = Xtr[: min(Xtr.shape[0], 500)]
    d2 = []
    for _ in range(3):
        i1 = rng.choice(sub.shape[0], size=min(200, sub.shape[0]), replace=False)
        i2 = rng.choice(sub.shape[0], size=min(200, sub.shape[0]), replace=False)
        d2.append(np.median(np.sum((sub[i1] - sub[i2]) ** 2, axis=1)))
    gamma = 1.0 / (2.0 * np.mean(d2) + 1e-12)

    Phi_tr = rbf_kernel_matrix(Xtr, gamma=gamma, seed=seed)
    Phi_te = rbf_kernel_cross(Xtr, Xte, gamma).T  # (n_te, n_tr)
    res = {}
    for label, het in (("hetero", True), ("homo", False)):
        m = JointARD(method="l2_irls", heteroscedastic=het,
                     max_iter=max_iter, damp_lambda=2e-2)
        m.fit(Phi_tr, yc_s)
        mu, var = m.predict(Phi_te)
        res[label] = {
            "rmse": float(rmse(yte_s, mu)),
            "ess_theta": ess_from_scores(1 / m.gamma_) * 100,
            "ess_y": ess_from_scores(1 / m.lambda_) * 100,
            "n_iter": m.n_iter_,
        }
    return res


def run_all():
    print("=== CLAIM4 SYNTHETIC RECOVERY (l2-IRLS, avg over trials) ===")
    t0 = time.time()
    res = synthetic_recovery()
    for key in res["weight_recovery"]:
        w = res["weight_recovery"][key]
        o = res["outlier_recovery"][key]
        e = res["ess_y"][key]
        print(f"  {key}: weight_rec het={w['hetero']:.3f} homo={w['homo']:.3f} | "
              f"outlier_rec het={o['hetero']:.3f} homo={o['homo']:.3f} | "
              f"ESSy het={e['hetero']:.3f} homo={e['homo']:.3f}")
    print("=== CLAIM4 RVM BOSTON (l2-IRLS, RBF basis d=n) ===")
    for rho in [0.0, 0.1, 0.2]:
        r = rvm_boston(contamination=rho)
        print(f"  contamination={rho}: " + " | ".join(
            f"{k} RMSE={v['rmse']:.3f} ESSth={v['ess_theta']:.1f}% ESSy={v['ess_y']:.1f}%"
            for k, v in r.items()))
    print(f"  _wallclock_s={time.time() - t0:.1f}")
    print("CLAIM4_DONE")


if __name__ == "__main__":
    run_all()

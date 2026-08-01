"""Claim 3: tabular regression benchmarks (Table 3).
Energy, Carbon, Protein with 10% contamination, RFF features d=256.
Reproduces RMSE, ESS(theta), ESS(y) for heteroscedastic SBL methods
(EM, MacKay, l2-IRLS, l1-IRLS) + homoscedastic counterparts, and the
relative RMSE improvement over the homoscedastic baseline."""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
import time
import json
import numpy as np

sys.path.insert(0, ".")
from repro.sbl import JointARD, ess_from_scores, rmse
from repro.data import (load_energy, load_carbon, load_protein,
                        contaminate_targets, split_data, cap_train,
                        random_fourier_features)
from repro.baselines import OLSBaseline, RidgeBaseline, GPBaseline, HuberBaseline, StudentTBaseline

METHODS = ["em", "mackay", "l2_irls", "l1_irls"]
DATASETS = {"energy": load_energy, "carbon": load_carbon, "protein": load_protein}
NMAX = 2000
RHO = 0.1
D_OUT = 256
N_TRIALS = 10
MAX_ITER = 2000


def run_trial(dataset, trial_seed, rho=RHO):
    X, y = DATASETS[dataset]()
    Xtr, ytr, Xte, yte = split_data(X, y, test_frac=0.2, seed=trial_seed)
    Xtr, ytr = cap_train(Xtr, ytr, nmax=NMAX, seed=trial_seed)

    # RFF features
    Phi_tr = random_fourier_features(Xtr, d_out=D_OUT, seed=trial_seed)
    Phi_te = random_fourier_features(Xte, d_out=D_OUT, seed=trial_seed, X_ref=Xtr)

    # App D.2: standardise targets to zero mean/unit var FIRST (uncontaminated
    # stats), THEN contaminate in standardised space with amplitude a=3.0.
    ymean, ystd = ytr.mean(), ytr.std()
    ytr_s = (ytr - ymean) / ystd
    yte_s = (yte - ymean) / ystd
    yc_s, mask = contaminate_targets(ytr_s, rho=rho, amp=3.0, seed=trial_seed)

    out = {}
    for method in METHODS:
        for het in (True, False):
            m = JointARD(method=method, heteroscedastic=het, max_iter=MAX_ITER, seed=trial_seed)
            m.fit(Phi_tr, yc_s)
            mu, var = m.predict(Phi_te)
            r = rmse(yte_s, mu)
            ess_t = ess_from_scores(1 / m.gamma_) * 100
            ess_y = ess_from_scores(1 / m.lambda_) * 100
            out[f"{method}_het={int(het)}"] = {
                "rmse": r, "ess_theta": ess_t, "ess_y": ess_y,
                "n_iter": m.n_iter_, "conv": bool(m.converged_)}
    return out


def run(dataset, n_trials=N_TRIALS, rho=RHO, seed0=0):
    all_res = {}
    t0 = time.time()
    for tr in range(n_trials):
        res = run_trial(dataset, seed0 + tr, rho)
        for k, v in res.items():
            all_res.setdefault(k, []).append(v)
    summary = {}
    for k, v in all_res.items():
        arr = np.array([x["rmse"] for x in v])
        ess_t = np.array([x["ess_theta"] for x in v])
        ess_y = np.array([x["ess_y"] for x in v])
        summary[k] = {
            "rmse_mean": float(arr.mean()), "rmse_std": float(arr.std()),
            "ess_theta_mean": float(ess_t.mean()), "ess_theta_std": float(ess_t.std()),
            "ess_y_mean": float(ess_y.mean()), "ess_y_std": float(ess_y.std()),
        }
    for method in METHODS:
        hkey = f"{method}_het=0"
        if hkey in summary:
            base = summary[hkey]["rmse_mean"]
            for het in (True, False):
                key = f"{method}_het={int(het)}"
                if key in summary:
                    rel = (summary[key]["rmse_mean"] - base) / base * 100
                    summary[key]["rel_vs_homo_pct"] = rel
    summary["_wallclock_s"] = time.time() - t0
    return summary


if __name__ == "__main__":
    dataset = sys.argv[1]
    out = run(dataset)
    print(json.dumps(out, indent=2, sort_keys=True))

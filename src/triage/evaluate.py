import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RepeatedKFold

from triage.config import SEED
from triage.models import build_gbr, build_ridge, build_v01, build_v02
from triage.risk import fit_threshold, flag_metrics, risk_label_cutoff
from triage.train import load_data

CANDIDATES = {
    "v0.1": build_v01,
    "ridge": build_ridge,
    "gbr": build_gbr,
    "v0.2": build_v02,
}


def _mean(rows: list[dict]) -> dict:
    return {key: round(float(np.mean([r[key] for r in rows])), 4) for key in rows[0]}


def evaluate(n_splits: int = 5, n_repeats: int = 5) -> dict:
    X, y = load_data()
    X, y = X.to_numpy(), y.to_numpy()
    folds = list(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED).split(X))

    results = {}
    per_fold_rmse = {}
    for name, builder in CANDIDATES.items():
        rmses, naive, calibrated = [], [], []
        for train_idx, test_idx in folds:
            X_tr, y_tr, X_te, y_te = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
            pred = clone(builder()).fit(X_tr, y_tr).predict(X_te)
            rmses.append(root_mean_squared_error(y_te, pred))
            cutoff = risk_label_cutoff(y_tr)
            naive.append(flag_metrics(y_te, pred, cutoff, cutoff))
            _, threshold = fit_threshold(builder(), X_tr, y_tr, SEED)
            calibrated.append(flag_metrics(y_te, pred, cutoff, threshold))
        per_fold_rmse[name] = np.array(rmses)
        results[name] = {
            "rmse_mean": round(float(np.mean(rmses)), 3),
            "rmse_std": round(float(np.std(rmses)), 3),
            "flag_naive": _mean(naive),
            "flag_calibrated": _mean(calibrated),
        }

    diff = per_fold_rmse["v0.2"] - per_fold_rmse["v0.1"]
    return {
        "protocol": {
            "splitter": f"RepeatedKFold(n_splits={n_splits}, n_repeats={n_repeats})",
            "seed": SEED,
            "n_folds": len(folds),
            "high_risk_label": "true progression >= 75th percentile of the fold's training targets",
        },
        "models": results,
        "paired_v0.2_vs_v0.1": {
            "rmse_diff_mean": round(float(diff.mean()), 3),
            "folds_where_v0.2_is_better": int((diff < 0).sum()),
            "n_folds": len(folds),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="metrics/cv_comparison.json", type=Path)
    parser.add_argument("--repeats", default=5, type=int, help="number of 5-fold repeats")
    args = parser.parse_args()

    report = evaluate(n_repeats=args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

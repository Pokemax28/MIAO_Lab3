"""Train the progression model and save it together with its metrics.

Usage:
    python -m triage.train --version v0.1 --out artifacts
"""

import argparse
import json
import platform
import random
import time
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from triage.config import SEED, TEST_SIZE
from triage.features import FEATURES
from triage.models import RISK_FLAG_VERSIONS, build_model
from triage.risk import fit_threshold, flag_metrics


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    """Load the scikit-learn diabetes dataset as features and progression index."""
    frame = load_diabetes(as_frame=True).frame
    X = frame.drop(columns=["target"])
    y = frame["target"]
    if list(X.columns) != FEATURES:
        raise RuntimeError(f"Unexpected feature columns: {list(X.columns)}")
    return X, y


def measure_latency_ms(model, row: np.ndarray, repeats: int = 200) -> float:
    """Average single-row prediction time in milliseconds."""
    model.predict(row)  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(row)
    return (time.perf_counter() - start) / repeats * 1000


def train(version: str, out_dir: Path) -> dict:
    """Fit the model for `version`, save it, and return the metrics dictionary."""
    set_seeds()
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(), y.to_numpy(), test_size=TEST_SIZE, random_state=SEED
    )

    model = build_model(version)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    pred = model.predict(X_test)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "model_version": version,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "rmse": round(float(root_mean_squared_error(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "runtime": {
            "fit_seconds": round(fit_seconds, 3),
            "latency_ms_single_row": round(measure_latency_ms(model, X_test[:1]), 4),
            "model_size_kb": round(model_path.stat().st_size / 1024, 1),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    if version in RISK_FLAG_VERSIONS:
        cutoff, threshold = fit_threshold(build_model(version), X_train, y_train, SEED)
        metrics["risk_flag"] = {
            "label_definition": "true progression >= 75th percentile of training targets",
            "label_cutoff": round(cutoff, 4),
            "threshold": round(threshold, 4),
            "test": {
                "calibrated": flag_metrics(y_test, pred, cutoff, threshold),
                "naive": flag_metrics(y_test, pred, cutoff, cutoff),
            },
        }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v0.1", help="model version to train")
    parser.add_argument("--out", default="artifacts", type=Path, help="output directory")
    args = parser.parse_args()

    metrics = train(args.version, args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

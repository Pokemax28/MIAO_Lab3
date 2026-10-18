import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_predict

RISK_QUANTILE = 0.75
CALIBRATION_FOLDS = 5
CANDIDATE_QUANTILES = np.linspace(0.30, 0.95, 66)


def risk_label_cutoff(y_train) -> float:
    return float(np.quantile(y_train, RISK_QUANTILE))


def fit_threshold(model, X_train, y_train, seed: int) -> tuple[float, float]:
    cutoff = risk_label_cutoff(y_train)
    oof = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=KFold(CALIBRATION_FOLDS, shuffle=True, random_state=seed),
    )
    labels = (y_train >= cutoff).astype(int)
    candidates = np.quantile(oof, CANDIDATE_QUANTILES)
    scores = [f1_score(labels, (oof >= t).astype(int)) for t in candidates]
    return cutoff, float(candidates[int(np.argmax(scores))])


def flag_metrics(y_true, pred, cutoff: float, threshold: float) -> dict:
    labels = (np.asarray(y_true) >= cutoff).astype(int)
    flags = (np.asarray(pred) >= threshold).astype(int)
    return {
        "precision": round(float(precision_score(labels, flags, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, flags, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, flags, zero_division=0)), 4),
    }

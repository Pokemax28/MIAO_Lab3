"""High-risk flag: label definition and threshold calibration.

A patient is labelled high risk when the true progression index is in the top quarter of the
training targets. The service flag compares the predicted score with a threshold. Using the
label cutoff itself as the threshold misses many high-risk patients, because predictions are
shrunk towards the mean. The threshold is therefore calibrated on out-of-fold predictions of
the training set to maximize F1, and the test set is only used to report the result.
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_predict

RISK_QUANTILE = 0.75
CALIBRATION_FOLDS = 5
# Candidate thresholds are quantiles of the out-of-fold predictions.
CANDIDATE_QUANTILES = np.linspace(0.30, 0.95, 66)


def risk_label_cutoff(y_train) -> float:
    """Progression index above which a patient counts as high risk."""
    return float(np.quantile(y_train, RISK_QUANTILE))


def fit_threshold(model, X_train, y_train, seed: int) -> tuple[float, float]:
    """Return (label_cutoff, score_threshold) calibrated on training data only."""
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
    """Precision, recall and F1 of the flag `pred >= threshold` against the true label."""
    labels = (np.asarray(y_true) >= cutoff).astype(int)
    flags = (np.asarray(pred) >= threshold).astype(int)
    return {
        "precision": round(float(precision_score(labels, flags, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, flags, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, flags, zero_division=0)), 4),
    }

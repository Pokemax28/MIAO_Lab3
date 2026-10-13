import numpy as np
import pytest

from tests.test_api import PAYLOAD
from triage.evaluate import evaluate
from triage.risk import flag_metrics


def test_flag_metrics_on_known_values():
    y_true = np.array([100, 200, 300, 400])
    pred = np.array([150, 250, 350, 120])
    # cutoff 250 labels the last two as high risk, threshold 200 flags the 2nd and 3rd
    metrics = flag_metrics(y_true, pred, cutoff=250, threshold=200)
    assert metrics == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_risk_metadata_is_saved(trained):
    version, _, metrics = trained
    if version != "v0.2":
        assert "risk_flag" not in metrics
        return
    risk = metrics["risk_flag"]
    assert risk["threshold"] < risk["label_cutoff"]
    assert risk["test"]["calibrated"]["recall"] >= risk["test"]["naive"]["recall"]


def test_risk_endpoint(client, trained):
    version, _, metrics = trained
    response = client.post("/predict/risk", json=PAYLOAD)
    if version != "v0.2":
        assert response.status_code == 404
        assert response.json()["error"] == "not_available"
        return
    assert response.status_code == 200
    body = response.json()
    assert body["threshold"] == metrics["risk_flag"]["threshold"]
    assert body["high_risk"] == (body["prediction"] >= body["threshold"])


def test_risk_endpoint_validates_input(client, trained):
    version, _, _ = trained
    if version != "v0.2":
        pytest.skip("risk flag only exists from v0.2")
    payload = {k: v for k, v in PAYLOAD.items() if k != "bmi"}
    response = client.post("/predict/risk", json=payload)
    assert response.status_code == 422
    assert response.json()["details"][0]["field"] == "bmi"


def test_evaluate_runs_quickly():
    report = evaluate(n_splits=3, n_repeats=1)
    assert set(report["models"]) == {"v0.1", "ridge", "gbr", "v0.2"}
    assert report["paired_v0.2_vs_v0.1"]["n_folds"] == 3

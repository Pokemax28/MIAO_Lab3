import json

import joblib
import numpy as np
import pytest

from triage.features import FEATURES
from triage.models import build_model
from triage.train import load_data, train


def test_data_matches_feature_contract():
    X, y = load_data()
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == 442


def test_unknown_version_is_rejected():
    with pytest.raises(ValueError, match="Unknown model version"):
        build_model("v9.9")


def test_training_writes_artifacts(trained):
    version, out_dir, metrics = trained
    assert (out_dir / "model.joblib").exists()
    saved = json.loads((out_dir / "metrics.json").read_text())
    assert saved["model_version"] == version
    assert saved["rmse"] == metrics["rmse"]


def test_rmse_is_reasonable(trained):
    """The mean predictor scores about 77 on this split, a useful model must beat it clearly."""
    _, _, metrics = trained
    assert metrics["rmse"] < 65
    assert metrics["r2"] > 0.3


def test_training_is_deterministic(trained, tmp_path):
    version, out_dir, metrics = trained
    again = train(version, tmp_path)
    assert again["rmse"] == metrics["rmse"]
    first = joblib.load(out_dir / "model.joblib")
    second = joblib.load(tmp_path / "model.joblib")
    rows = np.random.default_rng(0).normal(0, 0.05, size=(20, len(FEATURES)))
    np.testing.assert_allclose(first.predict(rows), second.predict(rows))

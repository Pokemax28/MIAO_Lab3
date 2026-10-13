import pytest
from fastapi.testclient import TestClient

from triage.models import MODEL_BUILDERS
from triage.train import train


@pytest.fixture(scope="session", params=sorted(MODEL_BUILDERS))
def trained(request, tmp_path_factory):
    """Train every released model version once and expose (version, dir, metrics)."""
    version = request.param
    out_dir = tmp_path_factory.mktemp(f"artifacts_{version}")
    metrics = train(version, out_dir)
    return version, out_dir, metrics


@pytest.fixture()
def client(trained, monkeypatch):
    from triage.app import app

    _, out_dir, _ = trained
    monkeypatch.setenv("MODEL_DIR", str(out_dir))
    with TestClient(app) as test_client:
        yield test_client

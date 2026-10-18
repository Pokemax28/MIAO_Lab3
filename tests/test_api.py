import math

PAYLOAD = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


def test_health_reports_version(client, trained):
    version, _, _ = trained
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": version}


def test_predict_returns_float(client):
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction"}
    assert isinstance(body["prediction"], float)
    assert math.isfinite(body["prediction"])


def test_predict_is_deterministic(client):
    first = client.post("/predict", json=PAYLOAD).json()
    second = client.post("/predict", json=PAYLOAD).json()
    assert first == second


def test_higher_bmi_never_lowers_the_score_much(client):
    low = client.post("/predict", json={**PAYLOAD, "bmi": -0.05}).json()["prediction"]
    high = client.post("/predict", json={**PAYLOAD, "bmi": 0.10}).json()["prediction"]
    assert high > low


def test_missing_feature_returns_422_with_json_error(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "s6"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["error"] == "validation_error"
    assert [d["field"] for d in body["details"]] == ["s6"]
    assert body["details"][0]["type"] == "missing"


def test_wrong_type_returns_422(client):
    response = client.post("/predict", json={**PAYLOAD, "bmi": "high"})
    assert response.status_code == 422
    assert response.json()["details"][0]["field"] == "bmi"


def test_non_finite_value_is_rejected(client):
    response = client.post(
        "/predict",
        content='{"age": NaN, "sex": 0, "bmi": 0, "bp": 0, "s1": 0, "s2": 0, "s3": 0, '
        '"s4": 0, "s5": 0, "s6": 0}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_malformed_json_returns_422(client):
    response = client.post(
        "/predict", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_all_errors_are_listed(client):
    response = client.post("/predict", json={"age": 0.02})
    assert response.status_code == 422
    fields = {d["field"] for d in response.json()["details"]}
    assert fields == {"sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"}

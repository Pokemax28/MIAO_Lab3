# Virtual Diabetes Clinic Triage

A small ML service that predicts short-term diabetes progression and returns a continuous risk
score. The triage dashboard calls `POST /predict` for each patient and sorts nurses' follow-up
calls by the predicted value, highest first.

The data is the open scikit-learn Diabetes dataset, used as a stand-in for de-identified EHR
features. The target is a progression index (higher means worse).

Current release: **v0.1** (baseline `StandardScaler` + `LinearRegression`).
See [CHANGELOG.md](CHANGELOG.md) for metrics.

## Run the released image

Replace `OWNER/REPO` with the GitHub namespace and repository name (lowercase).

```bash
docker pull ghcr.io/OWNER/REPO:v0.1
docker run -it -p 80:80 ghcr.io/OWNER/REPO:v0.1
```

Stop one version before starting another, both use port 80.

```bash
curl http://localhost:80/health
```

```json
{"status": "ok", "model_version": "v0.1"}
```

```bash
curl -X POST http://localhost:80/predict -H "Content-Type: application/json" \
  -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
```

```json
{"prediction": 235.94963722176266}
```

## API

### `GET /health`

Returns `200` with `{"status": "ok", "model_version": "<version>"}`.

### `POST /predict`

Body: JSON object with the ten features `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`,
`s5`, `s6`, all numbers (finite). Unknown extra keys are ignored.

Returns `200` with `{"prediction": <float>}`.

### Error format

Invalid input (missing field, wrong type, NaN or infinity, malformed JSON) returns HTTP `422`
with a JSON body:

```json
{
  "error": "validation_error",
  "message": "Invalid request payload",
  "details": [
    {"field": "s6", "issue": "Field required", "type": "missing"}
  ]
}
```

- `error`: stable machine-readable code (`validation_error`).
- `message`: short human-readable summary.
- `details`: one entry per problem. `field` is the feature name (or `body` when the whole
  body is not valid JSON), `issue` is the validation message, `type` is the validation error
  type (`missing`, `float_parsing`, `finite_number`, `json_invalid`, ...).

Unexpected server errors return HTTP `500` with `{"error": "internal_error", "message": "..."}`.

## Run locally

Requires Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

python -m triage.train --version v0.1 --out artifacts   # trains and writes artifacts/
uvicorn triage.app:app --port 8000                      # serves on http://localhost:8000
```

Quality checks (the same ones CI runs):

```bash
ruff check .
ruff format --check .
pytest
```

Build and run the image yourself:

```bash
docker build --build-arg MODEL_VERSION=v0.1 -t triage:v0.1 .
docker run -it -p 80:80 triage:v0.1
bash scripts/smoke_test.sh http://localhost:80 v0.1      # needs curl and jq
```

## Reproducibility

- Seeds: `random`, `numpy` and the train/test split use seed 42 (`src/triage/train.py`).
- Environment: every dependency is pinned in `requirements*.txt`, Docker uses `python:3.13-slim`.
- The Docker build trains the model itself (first stage), so an image is rebuilt from source.
- `artifacts/metrics.json` records RMSE, MAE, R2, latency, model size and library versions.
  The same code and the same pinned environment give the same metrics.

## CI and releases

- `.github/workflows/ci.yml` runs on every push and pull request: ruff lint and format check,
  unit tests, a training smoke test per model version. The trained model and `metrics.json`
  are uploaded as workflow artifacts. Any failure fails the run.
- `.github/workflows/release.yml` runs on tags matching `v*`: builds the image, starts the
  container and runs `scripts/smoke_test.sh` against it, pushes
  `ghcr.io/<owner>/<repo>:<tag>`, then publishes a GitHub Release with the matching
  `CHANGELOG.md` section and the metrics of the released image.

To publish a version:

```bash
git tag v0.1
git push origin v0.1
```

After the first push, check on GitHub (Packages, package settings) that the image visibility is
**Public**, otherwise `docker pull` needs a login.

## Layout

```
src/triage/        features, models, training CLI, FastAPI app
tests/             unit tests for training and the API
scripts/           container smoke test
Dockerfile         multi-stage build, model baked in
```

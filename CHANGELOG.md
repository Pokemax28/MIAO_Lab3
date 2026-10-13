# Changelog

Metrics are computed on the same held-out split for every version (20 percent of the 442
patients, `random_state=42`, 353 train and 89 test rows). `rmse` is in progression index units.
The release workflow copies `metrics.json` out of each released image, so these numbers can be
checked against the GitHub Release.

## v0.1 (2026-10-13)

Baseline.

- Model: `StandardScaler` then `LinearRegression`.
- Service: `GET /health` and `POST /predict` with JSON errors (HTTP 422) on invalid input.
- Packaging: multi-stage Docker image with the trained model baked in, listening on port 80.
- CI: lint, unit tests and a training smoke test on every push and pull request. Tag workflow
  builds the image, runs container smoke tests, pushes to GHCR and publishes a GitHub Release.

| Metric (held-out split) | v0.1 |
| ----------------------- | ---- |
| RMSE                    | 53.85 |
| MAE                     | 42.79 |
| R2                      | 0.453 |
| Single-row latency (ms) | 0.31 |
| Model size (KB)         | 1.4 |

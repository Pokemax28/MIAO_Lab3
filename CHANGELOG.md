# Changelog

Held-out metrics use the same split for every version (20 percent of the 442 patients,
`random_state=42`, 353 train and 89 test rows). `rmse` is in progression index units.
Because 89 rows is a small test set, v0.2 is also compared with repeated 5-fold cross-validation
(5 repeats, 25 folds), reproducible with `python -m triage.evaluate` and saved in
`metrics/cv_comparison.json`. The release workflow copies `metrics.json` out of each released
image, so the held-out numbers can be checked against the GitHub Release.

## Unreleased

Changes on main after the v0.2 tag. The released v0.1 and v0.2 images are not affected.

- Removed all comments and docstrings from the code and config files.
- The API logs the loaded model version at startup and every score it returns.
- New test checking that unknown extra keys in the payload are ignored.
- CI builds the Docker image and runs the smoke test on the container, and no longer cancels a run when a new push arrives.
- README rewritten with the real image names and screenshots of both versions running.

## v0.2 (2026-10-13)

Better score and a calibrated high-risk flag.

### What changed

- Model: ensemble model that thakes the average output of two models, a `RidgeCV` pipeline (standardized features, penalty chosen by
  built-in leave-one-out validation) and a small `GradientBoostingRegressor` (100 shallow trees,
  learning rate 0.05, subsampling). The v0.1 model `StandardScaler` + `LinearRegression` is removed.
- New endpoint `POST /predict/risk` returns `prediction`, `high_risk` and the `threshold` used.
  `POST /predict` and `GET /health` keep the exact same format as v0.1.
- High-risk label: true progression in the top 25 percent of the training targets (the "high risk" treashold in this case is taken arbitrarily). The flag
  threshold is calibrated on out-of-fold predictions of the training set to maximize F1, the
  test set is only used to report results. Training saves it in `metrics.json`.
- New `triage.evaluate` command and a CI job that runs the cross-validation comparison and
  uploads the report. CI trains both model versions.

### Metrics

Held-out split (what the released image reports):

| Metric                   | v0.1  | v0.2  | Change |
| ------------------------ | ----- | ----- | ------ |
| RMSE                     | 53.85 | 52.18 | -1.68  |
| MAE                      | 42.79 | 41.91 | -0.88  |
| R2                       | 0.453 | 0.486 | +0.033 |
| Single-row latency (ms)  | 0.3   | 0.7   | slower |
| Model size (KB)          | 1.4   | 85.8  | larger |

Repeated 5-fold cross-validation (25 folds, mean over folds):

| Metric                              | v0.1          | v0.2          |
| ----------------------------------- | ------------- | ------------- |
| RMSE                                | 54.93 (sd 2.63) | 54.70 (sd 2.34) |
| Folds where v0.2 has the lower RMSE |               | 14 of 25 |
| Reference: Ridge alone / GBR alone  |               | 55.01 / 56.02 |

High-risk flag of the v0.2 model, threshold at the label cutoff versus calibrated threshold:

| Flag threshold                  | Precision | Recall | F1    |
| ------------------------------- | --------- | ------ | ----- |
| Held-out, at label cutoff (214.0)  | 1.00   | 0.33   | 0.50  |
| Held-out, calibrated (180.7)       | 0.70   | 0.76   | 0.73  |
| CV, at label cutoff                | 0.81   | 0.50   | 0.61  |
| CV, calibrated                     | 0.65   | 0.68   | 0.66  |

### Why this is an improvement

- Lower error: RMSE drops by 1.68 on the held-out split and by 0.23 on average during cross validation,
  with less spread between folds (sd 2.34 versus 2.63). The average of the two models beats each
  model alone (Ridge 55.01, GBR 56.02), which is the reason for combining them.
- Fewer missed patients: with the threshold at the label cutoff, only half of the high-risk
  patients are flagged, because predictions are shrunk towards the mean. The calibrated
  threshold raises recall from 0.50 to 0.68 in cross-validation (0.33 to 0.76 on the held-out
  split) and F1 from 0.61 to 0.66. For a follow-up call list, missing a patient costs more than
  one extra call.

### Limits

- The RMSE gain is small compared with the fold-to-fold noise (v0.2 wins in 14 of 25 folds).
  This dataset is close to its ceiling for linear models, so large gains are not expected.
- Most of the flag gain comes from calibrating the threshold: the same calibration applied to
  the v0.1 model gives a similar CV F1 (0.66).
- Cost: predictions are about twice as slow and the model file is larger, both still negligible
  for this service (under 1 ms per patient).

## v0.1 (2026-10-13)

Baseline.

- Model: `StandardScaler` then `LinearRegression`.
- Service: `GET /health` and `POST /predict` with JSON errors (HTTP 422) on invalid input.
- Packaging: multi-stage Docker image with the trained model baked in, listening on port 80.
- CI: lint, unit tests and a training smoke test on every push and pull request. Tag workflow
  builds the image, runs container smoke tests, pushes to GHCR and publishes a GitHub Release.

| Metric (held-out split) | v0.1  |
| ----------------------- | ----- |
| RMSE                    | 53.85 |
| MAE                     | 42.79 |
| R2                      | 0.453 |
| Single-row latency (ms) | 0.3   |
| Model size (KB)         | 1.4   |

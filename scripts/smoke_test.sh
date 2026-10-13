#!/usr/bin/env bash
# Smoke test for a running service.
# Usage: scripts/smoke_test.sh [base_url] [expected_model_version]
set -euo pipefail

BASE_URL="${1:-http://localhost:80}"
EXPECTED_VERSION="${2:-}"
BODY_FILE="$(mktemp)"
trap 'rm -f "$BODY_FILE"' EXIT

fail() {
  echo "SMOKE TEST FAILED: $1" >&2
  exit 1
}

echo "Waiting for ${BASE_URL}/health ..."
for attempt in $(seq 1 30); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  if [ "$attempt" -eq 30 ]; then
    fail "service did not become healthy within 30 seconds"
  fi
  sleep 1
done

echo "GET /health"
curl -fsS "${BASE_URL}/health" >"$BODY_FILE"
cat "$BODY_FILE"
echo
[ "$(jq -r '.status' "$BODY_FILE")" = "ok" ] || fail "status is not ok"
if [ -n "$EXPECTED_VERSION" ]; then
  [ "$(jq -r '.model_version' "$BODY_FILE")" = "$EXPECTED_VERSION" ] ||
    fail "model_version is not ${EXPECTED_VERSION}"
fi

PAYLOAD='{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'

echo "POST /predict (valid payload)"
CODE="$(curl -sS -o "$BODY_FILE" -w '%{http_code}' -X POST "${BASE_URL}/predict" \
  -H 'Content-Type: application/json' -d "$PAYLOAD")"
cat "$BODY_FILE"
echo
[ "$CODE" = "200" ] || fail "expected HTTP 200, got ${CODE}"
[ "$(jq -r '.prediction | type' "$BODY_FILE")" = "number" ] || fail "prediction is not a number"

echo "POST /predict (missing s6)"
INVALID='{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02}'
CODE="$(curl -sS -o "$BODY_FILE" -w '%{http_code}' -X POST "${BASE_URL}/predict" \
  -H 'Content-Type: application/json' -d "$INVALID")"
cat "$BODY_FILE"
echo
[ "$CODE" = "422" ] || fail "expected HTTP 422, got ${CODE}"
[ "$(jq -r '.details[0].field' "$BODY_FILE")" = "s6" ] || fail "error does not name the missing field s6"

echo "All smoke tests passed."

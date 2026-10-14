#!/usr/bin/env bash
# End-to-end smoke test of the running docker compose stack:
# train + register + promote -> API serves it -> sync detect -> async job -> metrics.
#
#   docker compose up -d --build --wait && ./scripts/e2e.sh
set -euo pipefail

API="${API_URL:-http://localhost:8000}"
KEY="${API_KEY:-change-me}"
CONFIG="${CONFIG:-configs/train_synthetic.yaml}"
REQUEST="${REQUEST:-examples/request_synthetic.json}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

step() { printf '\n==> %s\n' "$*"; }

step "Train, register and promote ($CONFIG)"
docker compose --profile train run --rm trainer train --config "$CONFIG" --register --promote

step "Reload the model in the API (instead of waiting for the refresh loop)"
curl -fsS -X POST -H "X-API-Key: $KEY" "$API/admin/reload"
echo
for _ in $(seq 1 30); do
  curl -fsS "$API/health/ready" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "$API/health/ready"
echo

step "Synchronous detection: POST /v1/detect"
curl -fsS -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  --data @"$REQUEST" "$API/v1/detect" >"$TMP/sync.json"
python3 - "$TMP/sync.json" <<'EOF'
import json, sys
body = json.load(open(sys.argv[1]))
anomalies = body["anomalies"]
assert anomalies, "expected anomalies in the example request"
print(f"model {body['model']['name']} v{body['model']['version']}: {len(anomalies)} anomalies")
print(anomalies[0]["explanation"])
EOF

step "Asynchronous job: POST /v1/jobs + GET /v1/jobs/{id}"
job_url=$(curl -fsS -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  --data @"$REQUEST" "$API/v1/jobs" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status_url"])')
echo "job: $job_url"
for _ in $(seq 1 60); do
  curl -fsS -H "X-API-Key: $KEY" "$API$job_url" >"$TMP/job.json"
  status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$TMP/job.json")
  [ "$status" = "succeeded" ] && break
  [ "$status" = "failed" ] && { cat "$TMP/job.json"; exit 1; }
  sleep 1
done
[ "$status" = "succeeded" ] || { echo "job did not finish: $status"; exit 1; }
python3 -c 'import json,sys; r=json.load(open(sys.argv[1]))["result"]; print("job succeeded:", len(r["anomalies"]), "anomalies")' "$TMP/job.json"

step "Metrics"
curl -fsS "$API/metrics" | grep -E '^anomaly_(model_info|detections_total)'

printf '\nE2E OK\n'

#!/usr/bin/env bash
# End-to-end test of the faithfulness alert (rubric R08): push a deliberately low RAGAS faithfulness under the run
# label "alert-test", wait until Grafana's notification reaches the alert receiver, then delete the test value and
# wait for the "resolved" notification. Needs deploy/monitoring up. The value is synthetic and labelled as a test:
# it exercises the alert path, it is not a measurement.  Usage: scripts/test_alert.sh [value]   (default 0.62)
set -euo pipefail
cd "$(dirname "$0")/.."
VALUE="${1:-0.62}"
PUSH="${PUSHGATEWAY_URL:-http://localhost:9091}"
COMPOSE=(docker compose -f deploy/monitoring/docker-compose.yml)

wait_for() {  # wait_for <pattern> <seconds>
  for _ in $(seq 1 "$2"); do
    "${COMPOSE[@]}" logs alert-receiver 2>/dev/null | grep -q "$1" && return 0
    sleep 1
  done
  echo "timed out waiting for: $1"; return 1
}

echo "pushing rag_ragas_faithfulness=$VALUE (label alert-test) at $(date -u +%H:%M:%SZ)"
# same HELP and TYPE as rag.evaluation.ragas_eval pushes, or the Pushgateway rejects the value as inconsistent
printf '# HELP rag_ragas_faithfulness RAGAS faithfulness (mean over the evaluation set)\n# TYPE rag_ragas_faithfulness gauge\nrag_ragas_faithfulness %s\n' "$VALUE" \
  | curl -sf --data-binary @- "$PUSH/metrics/job/ragas_eval/label/alert-test"
wait_for "FIRING: RAGAS faithfulness below 0.80 (run=alert-test)" 180
echo "firing notification received at $(date -u +%H:%M:%SZ); removing the test value"
curl -sf -X DELETE "$PUSH/metrics/job/ragas_eval/label/alert-test"
wait_for "RESOLVED: RAGAS faithfulness below 0.80 (run=alert-test)" 300
echo "resolved notification received at $(date -u +%H:%M:%SZ)"
"${COMPOSE[@]}" logs --no-log-prefix alert-receiver | grep "alert-test"

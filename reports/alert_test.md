# Faithfulness alert: end-to-end test

Run with `scripts/test_alert.sh` against `deploy/monitoring` (Grafana 12.1, Prometheus 3.5, Pushgateway 1.11) on
2026-10-09.

**Rule** (provisioned from `deploy/monitoring/grafana/provisioning/alerting/alerting.yml`): *RAGAS faithfulness below
0.80*: the last `rag_ragas_faithfulness` of every run label, evaluated every 30 s, fires when it is below **0.80**,
and notifies the `alert-log` contact point, a webhook into the `alert-receiver` container that stands in for
Slack or email.

**Test:** a deliberately low value (0.62) pushed under the run label `alert-test`, then deleted. The value is
synthetic and labelled as a test: it exercises the alert path, it is not a measurement. The real runs (`baseline`,
`reranker`) are in [ragas_eval.md](ragas_eval.md).

| Step | Time (UTC) |
|---|---|
| test value pushed | 12:17:46 |
| **firing** notification received | 12:18:05 (19 s later) |
| test value deleted | 12:18:05 |
| **resolved** notification received | 12:19:05 |

Notifications as the receiver logged them (`docker compose -f deploy/monitoring/docker-compose.yml logs alert-receiver`):

```text
[2026-10-09 12:18:05 UTC] FIRING: RAGAS faithfulness below 0.80 (run=alert-test) RAGAS faithfulness 0.620 is below the 0.80 threshold
[2026-10-09 12:19:05 UTC] RESOLVED: RAGAS faithfulness below 0.80 (run=alert-test) RAGAS faithfulness 0.620 is below the 0.80 threshold
```

Note on the threshold: the production system's faithfulness on the 58 questions is 0.79–0.81 between runs (the 7B
judge is noisy, see [judge_comparison.md](judge_comparison.md)), so a real run can trip the 0.80 alert without any
regression. The handbook fixes 0.80; the CI gate fails only below 0.75.

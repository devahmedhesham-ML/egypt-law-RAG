# Canary rollout test

Run on 2026-10-08 with `deploy/canary/` (nginx 1.27 in front of two copies of the API image built by CI from `main`,
`ghcr.io/devahmedhesham-ml/egypt-law-rag-api:latest`, labelled `stable` and `canary` through `APP_RELEASE`). Every
request went through the entry point `http://localhost:8080/health`; the `X-Release` response header shows which
version answered. Weights were changed with `deploy/canary/set_weights.sh`, which reloads nginx without downtime.

| Step | Weights (stable/canary) | Requests | Answered by stable | Answered by canary | Errors |
|---|---|---|---|---|---|
| Start the canary | 90/10 | 300 | 270 | 30 | 0 |
| Widen | 50/50 | 300 | 151 | 149 | 0 |
| Promote | 0/100 | 100 | 0 | 100 | 0 |
| Roll back | 100/0 | 100 | 100 | 0 | 0 |
| Canary container stopped mid-rollout | 90/10 | 100 | 100 | 0 | 0 |

The split follows the weights, promotion and rollback take effect immediately, and when the canary dies nginx moves
its share to stable (`max_fails`, `proxy_next_upstream`) without a failed request. nginx's access log records the
release of every request (`release=canary` on 279 lines here), which is what a promotion decision reads.

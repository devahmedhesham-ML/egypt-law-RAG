#!/usr/bin/env bash
# Set the canary traffic split and apply it without downtime: deploy/canary/set_weights.sh STABLE CANARY
#   90 10 → start the canary · 50 50 → widen · 0 100 → promote · 100 0 → roll back
set -euo pipefail
cd "$(dirname "$0")"
stable="${1:?usage: set_weights.sh STABLE_WEIGHT CANARY_WEIGHT}"
canary="${2:?usage: set_weights.sh STABLE_WEIGHT CANARY_WEIGHT}"
render() {  # nginx needs weight >= 1: a 0 weight marks that server "down" instead
  local s="$1" c="$2"
  sed -e "s/weight=\${STABLE_WEIGHT}/$( [ "$s" -eq 0 ] && echo 'weight=1 down' || echo "weight=$s" )/" \
      -e "s/weight=\${CANARY_WEIGHT}/$( [ "$c" -eq 0 ] && echo 'weight=1 down' || echo "weight=$c" )/" \
      nginx.conf.template > nginx.conf
}
render "$stable" "$canary"
if docker compose ps --status running nginx 2>/dev/null | grep -q nginx; then
  docker compose exec -T nginx nginx -s reload
  echo "traffic now ${stable}% stable / ${canary}% canary (reloaded)"
else
  echo "wrote nginx.conf (${stable}/${canary}); start with: docker compose -f deploy/canary/docker-compose.yml up -d"
fi

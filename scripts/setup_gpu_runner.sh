#!/usr/bin/env bash
# Register this machine (WSL with the NVIDIA GPU, vLLM and the app venv) as a self-hosted GitHub Actions runner
# with the label `gpu`, for the CI quality gate. Needs the GitHub CLI logged in with admin rights on the repo.
#
#   scripts/setup_gpu_runner.sh          # install, register, set GPU_RUNNER=true
#   ~/actions-runner/run.sh              # start it (keep it running while CI should gate PRs)
#   scripts/setup_gpu_runner.sh --remove # unregister and set GPU_RUNNER=false
#
# Security: the workflow only sends jobs here for pushes to main, manual runs and pull requests from branches of
# this repository, never from forks, because a job runs with this account's access to the machine.
set -euo pipefail
REPO="${REPO:-devahmedhesham-ML/egypt-law-RAG}"
DIR="${RUNNER_DIR:-$HOME/actions-runner}"
GH="$(command -v gh || command -v gh.exe)"

if [ "${1:-}" = "--remove" ]; then
  token=$("$GH" api -X POST "repos/$REPO/actions/runners/remove-token" --jq .token)
  (cd "$DIR" && ./config.sh remove --token "$token")
  "$GH" variable set GPU_RUNNER --repo "$REPO" --body false
  exit 0
fi

mkdir -p "$DIR"
cd "$DIR"
if [ ! -x ./config.sh ]; then
  version=$("$GH" api repos/actions/runner/releases/latest --jq .tag_name | sed 's/^v//')
  curl -fsSL -o runner.tgz "https://github.com/actions/runner/releases/download/v${version}/actions-runner-linux-x64-${version}.tar.gz"
  tar xzf runner.tgz && rm runner.tgz
fi
token=$("$GH" api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token)
./config.sh --unattended --replace --url "https://github.com/$REPO" --token "$token" \
  --name "$(hostname)-gpu" --labels gpu --work _work
"$GH" variable set GPU_RUNNER --repo "$REPO" --body true
echo "Registered. Start it with: $DIR/run.sh"

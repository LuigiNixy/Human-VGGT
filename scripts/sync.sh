#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 1 && $# -ne 2 ]] || [[ $# -eq 2 && "$2" != "--apply" ]]; then
  echo "Usage: scripts/sync.sh user@host:/path/to/existing-checkout/ [--apply]" >&2
  exit 2
fi
opts=()
if [[ $# -eq 1 ]]; then opts+=(--dry-run); fi
exec rsync -av "${opts[@]}" --exclude=.git --exclude=.venv --exclude=.python \
  --exclude=ENV.md --exclude=.aws --exclude=.env --exclude='.env.*' --exclude='*.local.yaml' --exclude=third_party \
  --exclude=__pycache__ --exclude='*.egg-info' --exclude=.ruff_cache --exclude=.pytest_cache \
  --exclude=/data --exclude=/checkpoints --exclude=/cache --exclude=/outputs -- ./ "$1"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) lock=env/mac.lock.txt ;;
  Linux-x86_64) lock=env/linux-cu128.lock.txt ;;
  *) echo "No lock for this platform" >&2; exit 1 ;;
esac
git submodule update --init --recursive
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12.13 .venv
fi
.venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:3] == (3, 12, 13) else "Bootstrap requires Python 3.12.13 in .venv")'
uv pip sync --python .venv/bin/python --require-hashes "$lock"
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python

#!/usr/bin/env bash
set -euo pipefail

python -m unittest discover -s tests -p 'test_*.py'

if command -v openenv >/dev/null 2>&1; then
  openenv validate openenv.yaml
else
  echo "openenv CLI not installed; skipped 'openenv validate openenv.yaml'."
fi

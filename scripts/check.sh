#!/usr/bin/env bash
# Local gate = what CI runs: lint, format check, unit + regression tests (CPU).
# Usage: scripts/check.sh          full gate (= CI `check` + `weights` jobs; the weight-dependent tests skip
#                                  without `bloom download --what model,genes`)
#        scripts/check.sh fast     = the CI `check` job on pull requests: deselects the `weights` marker
#                                  (tests that need model.safetensors; issue #51), ~3 min on 4 vCPU
set -euo pipefail
cd "$(dirname "$0")/.."

EXPR="unit or regression"
if [[ "${1:-}" == "fast" ]]; then
  EXPR="(unit or regression) and not weights"
elif [[ -n "${1:-}" ]]; then
  echo "usage: scripts/check.sh [fast]" >&2
  exit 2
fi

echo "== ruff check"
ruff check .
echo "== ruff format --check"
ruff format --check .
echo "== pytest -m '$EXPR'"
pytest -m "$EXPR" --durations=20
echo "== check.sh: OK"

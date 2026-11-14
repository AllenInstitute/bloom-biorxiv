#!/usr/bin/env bash
# Create a conda environment for bloom-biorxiv and install the package in editable mode.
#
# Usage: scripts/setup_env.sh <env-name-or-prefix> [cpu|cu128]
#
#   <env-name-or-prefix>  a bare name (placed in conda's first `envs_dirs`; ~/.condarc may redirect
#                         that to another disk) or an absolute path
#   cpu|cu128             PyTorch build to install (default: cpu). cu128 = CUDA 12.8 wheels.
#
# What it does (see docs/standards.md item 4):
#   1. conda create -p <prefix> python=3.12 pip      (conda-forge envs ship without pip)
#   2. pip install "torch>=2.10" from the PyTorch index for the chosen build
#   3. pip install -e .[dev]                          (the only dependency file is pyproject.toml)
#   4. pre-commit install (if inside a git checkout) and `bloom info`
#
# One environment per worktree / parallel task. Never `conda activate` here (fails non-interactively
# on some hosts); the env is addressed by <prefix>/bin/python throughout.
set -euo pipefail

ENV_ARG=${1:?usage: scripts/setup_env.sh <env-name-or-prefix> [cpu|cu128]}
FLAVOR=${2:-cpu}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
TORCH_SPEC=${TORCH_SPEC:-"torch>=2.10"}   # override to pin, e.g. TORCH_SPEC="torch==2.11.*"

case "$FLAVOR" in
  cpu)   TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
  cu128) TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
  *) echo "unknown torch build '$FLAVOR' (expected cpu or cu128)" >&2; exit 2 ;;
esac

# --- locate conda ------------------------------------------------------------------------------
if [ -z "${CONDA_EXE:-}" ] || [ ! -x "${CONDA_EXE:-}" ]; then
  CONDA_EXE=""
  for cand in "$(command -v conda || true)" "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda" \
              "$HOME/mambaforge/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then CONDA_EXE=$cand; break; fi
  done
fi
if [ -z "$CONDA_EXE" ]; then
  echo "conda not found; install miniforge/miniconda or export CONDA_EXE=/path/to/conda" >&2; exit 1
fi
CONDA_BASE_PY="$(dirname "$CONDA_EXE")/python"

# --- resolve the env prefix (honour ~/.condarc envs_dirs) ----------------------------------------
case "$ENV_ARG" in
  /*|./*|../*) PREFIX=$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$ENV_ARG" 2>/dev/null || readlink -f "$ENV_ARG") ;;
  *)
    ENVS_DIR=$("$CONDA_EXE" info --json | "$CONDA_BASE_PY" -c 'import json,sys; print(json.load(sys.stdin)["envs_dirs"][0])')
    PREFIX="$ENVS_DIR/$ENV_ARG" ;;
esac

echo "== env prefix : $PREFIX"
echo "== torch      : $TORCH_SPEC from $TORCH_INDEX"
echo "== repo       : $REPO_ROOT"

# --- 1. conda env ----------------------------------------------------------------------------
if [ -x "$PREFIX/bin/python" ]; then
  echo "== env exists, reusing"
else
  "$CONDA_EXE" create -y -p "$PREFIX" python=3.12 pip
fi
PY="$PREFIX/bin/python"

# --- 2. torch ----------------------------------------------------------------------------------
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"

# --- 3. the package ------------------------------------------------------------------------------
cd "$REPO_ROOT"
"$PY" -m pip install -e ".[dev]"

# --- 4. hooks + smoke test ---------------------------------------------------------------------
if [ -d "$REPO_ROOT/.git" ] || git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  "$PREFIX/bin/pre-commit" install --install-hooks >/dev/null && echo "== pre-commit hook installed"
fi
"$PREFIX/bin/bloom" info
echo
echo "Done. Use it as:  $PREFIX/bin/python   (or add $PREFIX/bin to PATH)"

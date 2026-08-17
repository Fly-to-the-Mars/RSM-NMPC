#!/usr/bin/env bash
set -euo pipefail

# Optional convenience script for reviewers who want an isolated Python
# environment for the Python-only reproduction path.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements.txt"

echo "Environment ready."
echo "Activate with: source $VENV_DIR/bin/activate"

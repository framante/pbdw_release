#!/usr/bin/env bash
set -euo pipefail

# Repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Setup environment
source "$REPO_ROOT/venv/bin/activate"

LOGDIR="$REPO_ROOT/log_dir"
mkdir -p "$LOGDIR"
rm -f "$LOGDIR/err_pbdw.log" "$LOGDIR/out_pbdw.log"

# Accept JSON either from env OR first argument
JSON="${JSON:-${1:-}}"

if [[ -z "$JSON" ]]; then
  echo "ERROR: JSON file not provided."
  echo
  echo "Usage:"
  echo "  ./simple_run.sh path/to/file.json"
  echo
  echo "OR:"
  echo "  JSON=path/to/file.json ./simple_run.sh"
  exit 1
fi

if [[ ! -f "$JSON" ]]; then
  echo "ERROR: JSON file not found: $JSON"
  exit 1
fi

echo "Using JSON: $JSON"

python mains/run_pbdw.py --json="$JSON" 1>"$LOGDIR/out_pbdw.log" 2>"$LOGDIR/err_pbdw.log"

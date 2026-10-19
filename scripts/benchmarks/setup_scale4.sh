#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "$PWD" == */refgrader-reproductions/scale ]]; then
  echo 'Use the independent refgrader-scale4 workspace, not the SCALE workspace.' >&2
  exit 1
fi
mkdir -p reports .cache
export PIP_CACHE_DIR="$PWD/.cache/pip"
if [[ ! -x .venv/bin/python ]]; then
  candidate=''
  for interpreter in python3.11 python3.12 python3; do
    if command -v "$interpreter" >/dev/null && "$interpreter" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,14)' 2>/dev/null; then
      candidate="$(command -v "$interpreter")"
      break
    fi
  done
  if [[ -z "$candidate" ]]; then
    if ! command -v conda >/dev/null; then
      for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/.conda/bin/conda"; do
        if [[ -x "$c" ]]; then
          export PATH="$(dirname "$c"):$PATH"
          break
        fi
      done
    fi
    if ! command -v conda >/dev/null; then
      echo 'Python 3.11+ and conda not found. Install Python 3.11 or ask the administrator; do not alter the old scale-bert environment.' >&2
      exit 1
    fi
    conda create -y -p "$PWD/.runtime-python" python=3.11
    candidate="$PWD/.runtime-python/bin/python"
  fi
  "$candidate" -m venv .venv
fi
.venv/bin/python -m pip install -r scripts/benchmarks/scale4_requirements.txt
.venv/bin/python scripts/benchmarks/verify_scale4_bundle.py
.venv/bin/python scripts/benchmarks/run_scale4.py prepare
.venv/bin/python scripts/benchmarks/run_scale4.py plan --tag server_preflight_v1 > reports/preflight.log 2>&1
tail -n 1 reports/preflight.log
echo 'SETUP_COMPLETE: run bash scripts/benchmarks/launch_scale4.sh inside tmux.'

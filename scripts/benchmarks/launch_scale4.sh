#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
test -x .venv/bin/python
mkdir -p reports
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
# The API key belongs only to this terminal environment and never enters the bundle.
if [[ -z "${ZHIPUAI_API_KEY:-}" && -z "${ZHIPU_API_KEY:-}" ]]; then
  read -r -s -p 'GLM API Key (hidden): ' ZHIPUAI_API_KEY
  printf '\n'
  export ZHIPUAI_API_KEY
fi
.venv/bin/python scripts/benchmarks/verify_scale4_bundle.py
stamp="$(date +%Y%m%d_%H%M%S)"
set +e
.venv/bin/python -u scripts/benchmarks/run_scale4.py run --tag refgrader_glm53_scale4_v1 --repeats 5 --stage-attempts 3 2>&1 | tee "reports/refgrader_scale4_${stamp}.log"
code=${PIPESTATUS[0]}
set -e
printf '\nExit code: %s\n' "$code"
printf 'Summary: %s/results_runs/scale4_refgrader/refgrader_glm53_scale4_v1/summary.json\n' "$PWD"
if [[ "$code" -ne 0 ]]; then
  echo 'Saved results retained. Re-run this exact script to resume; do not delete or rename run directories.'
fi
exit "$code"

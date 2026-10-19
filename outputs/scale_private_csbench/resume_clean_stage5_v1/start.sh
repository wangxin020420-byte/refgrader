#!/usr/bin/env bash
set -euo pipefail
umask 077
project=/data0/E125221219/projects/scale-private
python=/data0/E125221219/projects/refgrader-scale4/.venv/bin/python
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$project/reports"
log="$project/reports/clean_v2_stage5_$(date +%Y%m%d_%H%M%S).log"
printf '%s\n' "$log" > "$project/reports/LATEST_CLEAN_V2_STAGE5_LOG.txt"
echo "日志：$log"
"$python" -u "$here/resume.py" prepare --project "$project" 2>&1 | tee "$log"
if [ -z "${ZHIPUAI_API_KEY:-}" ]; then
  read -r -s -p '请输入 Coding API Key（不回显）：' ZHIPUAI_API_KEY </dev/tty
  printf '\n'
fi
test -n "$ZHIPUAI_API_KEY"
export ZHIPUAI_API_KEY
"$python" -u "$here/resume.py" run --project "$project" 2>&1 | tee -a "$log"
echo "完成：$project/runs/online_experiments/csbench_scale_online_clean_v2/scale_stage5/summary.json"

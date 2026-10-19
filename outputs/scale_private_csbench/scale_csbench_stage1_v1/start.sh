#!/usr/bin/env bash
set -euo pipefail
umask 077

project=/data0/E125221219/projects/scale-private
python=/data0/E125221219/projects/refgrader-scale4/.venv/bin/python
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment=csbench_scale_online_v1
test -x "$python"
mkdir -p "$project/reports"
log="$project/reports/${experiment}_stage1_$(date +%Y%m%d_%H%M%S).log"
printf '%s\n' "$log" > "$project/reports/LATEST_CSBENCH_STAGE1_LOG.txt"
exec > >(tee -a "$log") 2>&1
echo "STAGE1_START experiment=$experiment"
echo "日志：$log"
"$python" -c 'import openai; print("DEPENDENCIES_OK")'
"$python" -u "$here/run_stage1.py" prepare --project "$project" --experiment-id "$experiment"
if [ -z "${ZHIPUAI_API_KEY:-}" ]; then
  read -r -s -p '请输入 Coding API Key（不回显）：' ZHIPUAI_API_KEY < /dev/tty
  printf '\n'
fi
export ZHIPUAI_API_KEY
test -n "$ZHIPUAI_API_KEY"
set +e
"$python" -u "$here/run_stage1.py" run --project "$project" --experiment-id "$experiment" --attempts 8
rc=$?
set -e
echo "退出码：$rc"
echo "汇总：$project/runs/online_experiments/$experiment/scale_stage1/summary.json"
echo "日志：$log"
exit "$rc"

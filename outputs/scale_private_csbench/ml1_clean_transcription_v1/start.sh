#!/usr/bin/env bash
set -euo pipefail
umask 077
project=/data0/E125221219/projects/scale-private
python=/data0/E125221219/projects/refgrader-scale4/.venv/bin/python
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$python" -c 'import openai; from PIL import Image; print("DEPENDENCIES_OK")'
"$python" -u "$script_dir/run.py" --project "$project"
if [ -z "${ZHIPUAI_API_KEY:-}" ]; then
  read -r -s -p '请输入 Coding API Key（不回显）：' ZHIPUAI_API_KEY </dev/tty
  printf '\n'
fi
test -n "$ZHIPUAI_API_KEY"
export ZHIPUAI_API_KEY
mkdir -p "$project/reports"
log="$project/reports/ml1_clean_transcription_$(date +%Y%m%d_%H%M%S).log"
printf '%s\n' "$log" > "$project/reports/LATEST_ML1_CLEAN_LOG.txt"
echo "日志：$log"
set +e
"$python" -u "$script_dir/run.py" --project "$project" --run 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
echo "退出码：$rc"
exit "$rc"

#!/usr/bin/env bash
set -euo pipefail
umask 077
project=/data0/E125221219/projects/scale-private
model_root=/data0/E125221219/projects/refgrader-reproductions/scale
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$project/reports"
log="$project/reports/csbench_scale_online_v1_stage2_$(date +%Y%m%d_%H%M%S).log"
printf '%s\n' "$log" > "$project/reports/LATEST_CSBENCH_STAGE2_LOG.txt"
exec > >(tee -a "$log") 2>&1
echo "日志：$log"
# Use an existing environment only; do not install or upgrade old environments.
candidates=()
if [ -n "${SCALE_PYTHON:-}" ]; then candidates+=("$SCALE_PYTHON"); fi
if [ -n "${CONDA_PREFIX:-}" ]; then candidates+=("$CONDA_PREFIX/bin/python"); fi
candidates+=("$model_root/.venv/bin/python" "/data0/E125221219/projects/refgrader-scale4/.venv/bin/python")
if command -v python >/dev/null 2>&1; then candidates+=("$(command -v python)"); fi
python=""
for candidate in "${candidates[@]}"; do
  if [ -x "$candidate" ] && "$candidate" -c 'import torch, transformers, numpy, safetensors; from transformers import AutoModel, AutoTokenizer' >/dev/null 2>&1; then
    python="$candidate"
    break
  fi
done
if [ -z "$python" ]; then
  echo '未找到已有的 torch/transformers/numpy/safetensors 环境。'
  echo '请在此前 ADS 的训练环境中执行本脚本，或先 export SCALE_PYTHON=该环境的python绝对路径。'
  exit 1
fi
echo "PYTHON=$python"
"$python" -c 'import torch,transformers,numpy; print("VERSIONS",torch.__version__,transformers.__version__,numpy.__version__)'
set +e
"$python" -u "$here/run_stage2.py" run --project "$project" --model-root "$model_root" --experiment-id csbench_scale_online_v1 --top-k 5 --seed 42
rc=$?
set -e
echo "退出码：$rc"
echo "汇总：$project/runs/online_experiments/csbench_scale_online_v1/scale_stage2/summary.json"
exit "$rc"

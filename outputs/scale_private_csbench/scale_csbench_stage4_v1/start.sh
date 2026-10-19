#!/usr/bin/env bash
set -euo pipefail
umask 077
project=/data0/E125221219/projects/scale-private
model_root=/data0/E125221219/projects/refgrader-reproductions/scale
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="$(cat "$project/reports/STAGE2_PYTHON.txt")"
test -x "$python"
mkdir -p "$project/reports"
log="$project/reports/csbench_scale_online_v1_stage4_$(date +%Y%m%d_%H%M%S).log"
printf '%s\n' "$log" > "$project/reports/LATEST_CSBENCH_STAGE4_LOG.txt"
exec > >(tee -a "$log") 2>&1
echo "日志：$log"
echo "PYTHON=$python"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$python" -c 'import torch,transformers,numpy,safetensors; print("DEPENDENCIES_OK")'
set +e
"$python" -u "$here/run_stage4.py" run --project "$project" --model-root "$model_root" --experiment-id csbench_scale_online_v1 --neighbors 5 --budget 5000
rc=$?
set -e
echo "退出码：$rc"
echo "汇总：$project/runs/online_experiments/csbench_scale_online_v1/scale_stage4/summary.json"
exit "$rc"

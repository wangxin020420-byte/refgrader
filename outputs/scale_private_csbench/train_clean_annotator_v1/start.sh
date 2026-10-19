#!/usr/bin/env bash
set -euo pipefail
umask 077
project=/data0/E125221219/projects/scale-private
model_root=/data0/E125221219/projects/refgrader-reproductions/scale
python=/data0/E125221219/tools/envs/csbench-scale-stage2-v1/bin/python
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
test -x "$python"
mkdir -p "$project/reports"
log="$project/reports/clean_v2_annotator_$(date +%Y%m%d_%H%M%S).log"
printf '%s\n' "$log" > "$project/reports/LATEST_CLEAN_V2_ANNOTATOR_LOG.txt"
exec > >(tee -a "$log") 2>&1
echo "日志：$log"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$python" -c 'import torch,transformers,numpy,safetensors; print("VERSIONS",torch.__version__,transformers.__version__); assert torch.cuda.is_available(), "CUDA unavailable"'
"$python" -u "$here/run_annotators.py" all \
  --root "$project" --model-root "$model_root" --epochs 10
cat "$project/runs/online_experiments/csbench_scale_online_clean_v2/annotator_bert_valqwk_v1/summary.json"

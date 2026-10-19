#!/usr/bin/env bash
set -euo pipefail
umask 077
project=/data0/E125221219/projects/scale-private
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$project/reports"
log="$project/reports/clean_v2_train_entities_recovery_$(date +%Y%m%d_%H%M%S)_$$.log"
printf '%s\n' "$log" > "$project/reports/LATEST_CLEAN_V2_TRAIN_ENTITIES_LOG.txt"
exec > >(tee -a "$log") 2>&1
trap 'rc=$?; printf "退出码：%s\n日志：%s\n" "$rc" "$log"' EXIT
echo "日志：$log"
"${SCALE_PYTHON:-python3}" -u "$here/recover.py"

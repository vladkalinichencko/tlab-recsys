#!/bin/bash
# One Yambda task from the raw files: 1_data/prep_yambda.py writes runs/yambda_<task>/data/*.parquet,
# then run_comparison.sh trains every model in that run directory:
#     bash run_yambda.sh listen
#     bash run_yambda.sh like
set -eo pipefail
cd "$(dirname "$0")"
TASK=$1
ROOT=$PWD
mkdir -p "runs/yambda_$TASK/logs"
(cd "runs/yambda_$TASK" && PYTHONPATH="$ROOT/0_shared" python3 "$ROOT/1_data/prep_yambda.py" "$TASK" | tee logs/prep.log)
bash run_comparison.sh "runs/yambda_$TASK"

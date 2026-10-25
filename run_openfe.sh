#!/bin/bash
# OpenFE selection, then CatBoost YetiRank on all + selected grammar features, in the current run directory:
#     tmux new-session -d -s openfe_kion -c "$PWD" "bash run_openfe.sh"
#     tmux new-session -d -s openfe_listen -c "$PWD/runs/yambda_listen" "bash ../../run_openfe.sh"
set -eo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$ROOT/0_shared"
python3 "$ROOT/3_boosting/select_openfe.py" | tee logs/openfe_select.log
python3 "$ROOT/3_boosting/train_boosting.py" yetirank openfe | tee logs/boosting_yetirank_openfe.log

#!/bin/bash
# KION from the raw files: 1_data/prep.py draws the 100k users with random_state=0, test_leak.py checks the
# split, then run_comparison.sh trains every model; every script takes the split and pool from stage.prepare.
#     tmux new-session -d -s recsys -c "$PWD" "bash run.sh"
set -eo pipefail
# every step imports kion, stage, candidates, features, metrics from 0_shared
export PYTHONPATH="$PWD/0_shared"
mkdir -p logs/catboost models reports

python3 1_data/prep.py | tee logs/prep.log
python3 1_data/test_leak.py

bash run_comparison.sh

#!/bin/bash
# Rerun every experiment on the same sample: 1_data/prep.py draws the 100k users with random_state=0,
# and every script takes the split and candidate pool from stage.prepare.
#     tmux new-session -d -s recsys -c "$PWD" "bash run.sh"
set -eo pipefail
# every step imports kion, stage, candidates, features, metrics from 0_shared
export PYTHONPATH="$PWD/0_shared"
mkdir -p logs/catboost models reports

python3 1_data/prep.py | tee logs/prep.log
python3 1_data/test_leak.py

python3 2_baselines/baseline_popularity.py | tee logs/popularity.log
python3 2_baselines/baseline_ease.py | tee logs/ease.log

for features in all hand within_user; do
    python3 3_boosting/train_boosting.py yetirank "$features" | tee "logs/boosting_yetirank_$features.log"
done
python3 3_boosting/train_boosting.py pointwise all | tee logs/boosting_pointwise_all.log
python3 3_boosting/diag_importance.py | tee logs/importance.log

for encoder in autoencoder bag lstm transformer; do
    python3 4_neural/train_neural.py "$encoder" | tee "logs/neural_$encoder.log"
done

python3 5_report/report.py | tee logs/report.log

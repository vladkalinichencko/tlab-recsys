#!/bin/bash
# One Yambda task end to end, in its own run directory with data/, logs/, models/, reports/:
#     tmux new-session -d -s yambda_listen -c "$PWD" "bash run_yambda.sh listen"
#     tmux new-session -d -s yambda_like   -c "$PWD" "bash run_yambda.sh like"
set -eo pipefail
TASK=$1
ROOT=$PWD
export PYTHONPATH="$ROOT/0_shared:$ROOT/3_boosting"
# the nets read the last 512 events of each user, the boosting aggregates the whole history
export MAX_EVENTS=512
mkdir -p "runs/yambda_$TASK" && cd "runs/yambda_$TASK"
mkdir -p logs/catboost models reports

python3 "$ROOT/1_data/prep_yambda.py" "$TASK" | tee logs/prep.log

python3 "$ROOT/2_baselines/baseline_popularity.py" | tee logs/popularity.log
python3 "$ROOT/2_baselines/baseline_ease.py" | tee logs/ease.log

for encoder in autoencoder bag lstm transformer; do
    python3 "$ROOT/4_neural/train_neural.py" "$encoder" | tee "logs/neural_$encoder.log"
done

python3 "$ROOT/4_neural/train_tabm.py" | tee logs/tabm_all.log

for features in all grammar hand within_user; do
    python3 "$ROOT/3_boosting/train_boosting.py" yetirank "$features" | tee "logs/boosting_yetirank_$features.log"
done
python3 "$ROOT/3_boosting/train_boosting.py" pointwise all | tee logs/boosting_pointwise_all.log

python3 "$ROOT/5_report/report.py" | tee logs/report.log

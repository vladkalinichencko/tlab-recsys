#!/bin/bash
# KION models compared on the same 200 random test users (metrics.eval_users, subset_* in every report).
#     tmux new-session -d -s kion_trees  -c "$PWD" "bash run_subset.sh trees"
#     tmux new-session -d -s kion_nets   -c "$PWD" "bash run_subset.sh nets"
# trees run on the CPU, nets share the GPU one after another
set -eo pipefail
export PYTHONPATH="$PWD/0_shared:$PWD/3_boosting:$PWD/4_neural"
run() { python3 -u "$@" 2>&1 | grep --line-buffered -v Warning; }
if [ "$1" = trees ]; then
    run 2_baselines/baseline_popularity.py | tee logs/popularity.log
    run 2_baselines/baseline_ease.py | tee logs/ease.log
    for features in hand auto; do
        run 3_boosting/train_boosting.py yetirank "$features" | tee "logs/boosting_yetirank_$features.log"
    done
else
    for variant in softmax counts; do
        run 4_neural/train_relational.py "$variant" | tee "logs/tabicl_$variant.log"
    done
    for features in all auto; do
        run 4_neural/train_tabm.py "$features" | tee "logs/tabm_$features.log"
    done
fi

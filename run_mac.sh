#!/bin/bash
# The fast half of the comparison, on the Mac CPU: popularity, EASE, boosting on all features, on the
# hand features and on the automatically generated ones (grammar.py), and the permutation importance,
# on KION (.) and on both Yambda tasks (runs/yambda_listen, runs/yambda_like) by default, or on the run
# directories given as arguments. One tmux session runs everything one after another;
# logs in <run directory>/logs/, reports in <run directory>/reports/.
set -eo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
PYTHONPATH="$ROOT/0_shared:$ROOT/3_boosting:$ROOT/4_neural"
if [ $# -eq 0 ]; then set -- . runs/yambda_listen runs/yambda_like; fi

job() {
    local dir=$1 name=$2 script=$3
    shift 3
    echo "cd '$ROOT/$dir' && mkdir -p logs models reports && PYTHONPATH='$PYTHONPATH' python3 -u '$ROOT/$script' $* 2>&1 | grep --line-buffered -v Warning | tee logs/$name.log"
}

commands=""
for dir in "$@"; do
    commands="$commands${commands:+; }$(job "$dir" popularity 2_baselines/baseline_popularity.py); $(job "$dir" ease 2_baselines/baseline_ease.py)"
    for features in all hand auto; do
        commands="$commands; $(job "$dir" "boosting_yetirank_$features" 3_boosting/train_boosting.py "$features")"
    done
    commands="$commands; $(job "$dir" importance 3_boosting/diag_importance.py)"
done

tmux new-session -d -c "$ROOT" -s mac_trees "$commands"
tmux ls

#!/bin/bash
# The fast half of the KION comparison, on the Mac CPU: popularity, EASE, boosting on the hand features
# and on the automatically generated ones (grammar.py), one after another, about 2 h; logs in logs/, reports in reports/.
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs models
export PYTHONPATH="$PWD/0_shared:$PWD/3_boosting:$PWD/4_neural"
job() { echo "PYTHONPATH='$PYTHONPATH' python3 -u $1 2>&1 | grep --line-buffered -v Warning | tee logs/$2.log"; }

trees="$(job 2_baselines/baseline_popularity.py popularity); $(job 2_baselines/baseline_ease.py ease)"
for features in hand auto; do
    trees="$trees; $(job "3_boosting/train_boosting.py yetirank $features" "boosting_yetirank_$features")"
done

tmux new-session -d -c "$PWD" -s kion_mac -n trees "$trees"
tmux ls

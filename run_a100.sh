#!/bin/bash
# The slow half of the comparison, on one A100: four TabICL variants, TabM and the autoregressive
# transformer, on KION (.) and on both Yambda tasks (runs/yambda_listen, runs/yambda_like) by default,
# or on the run directories given as arguments.
# Each model is one background process that goes through the datasets one after another, so six
# processes share the card at any time: a TabICL run holds 27.6M parameters and batches of 16 tables
# of 65 rows, a few GB each; the Yambda transformer (512 events, 20K tracks) peaks near 30 GB.
# The runs are detached with setsid, so they outlive the terminal (no tmux needed).
# On the machine (the prepared tables data/*.parquet and runs/yambda_*/data/*.parquet are in the repository):
#     git clone https://github.com/vladkalinichencko/tlab-recsys.git recsys
#     cd recsys && pip install -r requirements.txt && bash run_comparison.sh
#     bash run_a100.sh runs/yambda_listen runs/yambda_like   # Yambda only
#     tail -f logs/tabicl_softmax.log                        # progress of one run
#     grep -h "val ndcg" logs/*.log runs/*/logs/*.log | tail # last val numbers of all runs
set -eo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
export PYTHONPATH="$ROOT/0_shared:$ROOT/3_boosting:$ROOT/4_neural"
python3 -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; print(torch.cuda.get_device_name())"
if [ $# -eq 0 ]; then set -- . runs/yambda_listen runs/yambda_like; fi

run() { setsid nohup bash -c "$1" > /dev/null 2>&1 & }
job() {
    local dir=$1 name=$2 script=$3
    shift 3
    # Yambda has its own schema (0_shared/yambda.py); a Yambda user has a median of 1.7K events, the nets read the last 512
    local yambda=""
    [[ $dir == runs/yambda_* ]] && yambda="DATASET=yambda MAX_EVENTS=512 "
    echo "cd '$ROOT/$dir' && mkdir -p logs models reports && ${yambda}python3 -u '$ROOT/$script' $* 2>&1 | grep --line-buffered -v Warning > logs/$name.log"
}
chain() {
    local commands=""
    for dir in "${DATASETS[@]}"; do commands="$commands${commands:+; }$(job "$dir" "$@")"; done
    echo "$commands"
}
DATASETS=("$@")

run "$(chain tabm_all 4_neural/train_tabm.py all); $(chain tabm_auto 4_neural/train_tabm.py auto)"
for variant in softmax counts gates full; do
    run "$(chain "tabicl_$variant" 4_neural/train_relational.py "$variant")"
done
run "$(chain neural_transformer 4_neural/train_neural.py transformer)"

sleep 2
pgrep -af "train_" || true

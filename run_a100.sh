#!/bin/bash
# The slow half of the KION comparison, on one A100: three TabICL variants, TabM and the autoregressive
# transformer, all at once.
# Each TabICL run holds 27.6M parameters and batches of 16 tables of 65 rows, a few GB of the
# 64 GB budget, so the five runs share the card in parallel (one background process each, about 1.5 h).
# The runs are detached with setsid, so they outlive the terminal (no tmux needed).
# On the machine (the prepared KION tables data/*.parquet are in the repository):
#     git clone https://github.com/vladkalinichencko/tlab-recsys.git recsys
#     cd recsys && pip install -r requirements.txt && bash run_comparison.sh
#     tail -f logs/tabicl_softmax.log          # progress of one run
#     grep -h "val ndcg" logs/*.log | tail     # last val numbers of all runs
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs models
export PYTHONPATH="$PWD/0_shared:$PWD/3_boosting:$PWD/4_neural"
python3 -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; print(torch.cuda.get_device_name())"

run() { setsid nohup bash -c "$1" > /dev/null 2>&1 & }
job() { echo "python3 -u $1 2>&1 | grep --line-buffered -v Warning > logs/$2.log"; }

run "$(job '4_neural/train_tabm.py all' tabm_all); $(job '4_neural/train_tabm.py auto' tabm_auto)"
for variant in softmax counts gates; do
    run "$(job "4_neural/train_relational.py $variant" "tabicl_$variant")"
done
run "$(job '4_neural/train_neural.py transformer' neural_transformer)"

sleep 2
pgrep -af "train_" || true

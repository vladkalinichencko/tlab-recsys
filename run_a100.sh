#!/bin/bash
# The slow half of the KION comparison, on one A100: three TabICL variants, TabM and the autoregressive
# transformer, all at once.
# Each TabICL run holds 27.6M parameters and batches of 16 tables of 65 rows, a few GB of the
# 64 GB budget, so the five runs share the card in parallel (one tmux window each, about 1.5 h).
# On the machine (the prepared KION tables data/*.parquet are in the repository):
#     git clone https://github.com/vladkalinichencko/tlab-recsys.git recsys
#     cd recsys && pip install -r requirements.txt && bash run_comparison.sh
#     tmux attach -t kion_a100          # ctrl-b n for the next window
# Back on the Mac:
#     rsync -a HOST:recsys/logs/ logs/ && rsync -a HOST:recsys/reports/ reports/ && rsync -a HOST:recsys/models/ models/
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs models
export PYTHONPATH="$PWD/0_shared:$PWD/3_boosting:$PWD/4_neural"
job() { echo "PYTHONPATH='$PYTHONPATH' python3 -u $1 2>&1 | grep --line-buffered -v Warning | tee logs/$2.log"; }

tmux new-session -d -c "$PWD" -s kion_a100 -n tabm "$(job '4_neural/train_tabm.py all' tabm_all); $(job '4_neural/train_tabm.py auto' tabm_auto)"
for variant in softmax counts gates; do
    tmux new-window -c "$PWD" -t kion_a100 -n "tabicl_$variant" "$(job "4_neural/train_relational.py $variant" "tabicl_$variant")"
done
tmux new-window -c "$PWD" -t kion_a100 -n neural_transformer "$(job '4_neural/train_neural.py transformer' neural_transformer)"
tmux ls

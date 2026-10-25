"""Pretrained TabICL v2 (Qu et al. 2025) over the relational context of each pair, then fine-tuned on KION.

    python3 4_neural/train_relational.py {softmax,counts}

softmax keeps TabICL as pretrained; counts swaps its attention between rows for SiLU / n
(counting_attention.py), so the number of matching events reaches the output, and fine-tunes from
the same weights. The two differ only in that one block.

TabICL reads a table column by column (distribution of each feature), then row by row (between
the features of a row), then across rows (between events). Here the table of a pair is built by
contexts.py: the user's last events, the film's last events by all users, the pair as the last row.
No hand features. Epoch 0 is the pretrained model as is (zero-shot), every epoch after it is
STEPS fine-tuning steps; the epoch with the best val ndcg@10 on VAL_USERS users is scored on the
test users of metrics.eval_users, the same users every other model reports as subset_*.

Batches hold half positive pairs and half negative ones, since one random batch has 0.1 positive
on average; the score adds back the log odds of the real positive rate, so log loss stays honest.
"""
import copy
import sys

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tabicl._model.tabicl import TabICL

import contexts
import counting_attention
import metrics

VARIANT = sys.argv[1]
NAME = f"tabicl_{VARIANT}"
EPOCHS = 3
STEPS = 50
BATCH = 16
LR = 1e-5
VAL_USERS = 20
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
rng = np.random.default_rng(0)


def load_model():
    checkpoint = torch.load(
        hf_hub_download("jingang/TabICL", "tabicl-classifier-v2-20260212.ckpt"), map_location="cpu", weights_only=True
    )
    model = TabICL(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(DEVICE)


def logit(model, stage, pairs):
    X_BTH, y_BT, label_B = stage.batch(pairs)
    out_B1C = model(X_BTH.to(DEVICE), y_BT.to(DEVICE))
    return out_B1C[:, -1, 1] - out_B1C[:, -1, 0], label_B.to(DEVICE)


def score(model, stage, users):
    """probability of watched for every pair of `users`, the balanced-batch prior taken back out."""
    pairs = np.flatnonzero(np.isin(stage.pair_user, users))
    with torch.no_grad():
        parts = [logit(model, stage, pairs[i : i + BATCH])[0].cpu() for i in range(0, len(pairs), BATCH)]
    frame = stage.frame.iloc[pairs].copy()
    frame["score"] = torch.sigmoid(torch.cat(parts) + shift).numpy()
    return frame


fit, val, test = contexts.Stage("fit"), contexts.Stage("val"), contexts.Stage("test")
val_users = np.random.default_rng(0).choice(np.unique(val.pair_user), VAL_USERS, replace=False)
test_users = metrics.eval_users(test.frame)
positive, negative = np.flatnonzero(fit.label == 1), np.flatnonzero(fit.label == 0)
base_rate = fit.label.mean()
shift = 0.0  # zero-shot reads the watched rate of past events, not of the window, so its log loss is not calibrated

model = load_model()
if VARIANT == "counts":
    counting_attention.use(model)
model.train()  # TabICL has no dropout, train mode only switches to the plain forward
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

best, best_state, best_shift, curve = -1.0, None, 0.0, []
for epoch in range(EPOCHS + 1):
    if epoch > 0:
        shift = float(np.log(base_rate / (1 - base_rate)))  # log odds of the real rate minus those of 50/50
        for step in range(STEPS):
            pairs = np.concatenate([rng.choice(positive, BATCH // 2), rng.choice(negative, BATCH // 2)])
            logit_B, label_B = logit(model, fit, pairs)
            loss = F.binary_cross_entropy_with_logits(logit_B, label_B.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch {epoch}  step {step + 1:3d}  loss {loss.item():.4f}", flush=True)

    ndcg = metrics.ranking(score(model, val, val_users), "score")["ndcg@10"]
    curve.append(ndcg)
    print(f"epoch {epoch}  val ndcg@10 {ndcg:.4f}", flush=True)
    if ndcg > best:
        best, best_state, best_shift = ndcg, copy.deepcopy(model.state_dict()), shift

model.load_state_dict(best_state)
shift = best_shift
metrics.report(
    NAME,
    score(model, test, test_users),
    "score",
    {"context_rows": contexts.ROWS, "val_users": VAL_USERS, "val_ndcg@10_curve": curve, "best_epoch": int(np.argmax(curve))},
)

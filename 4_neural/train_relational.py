"""Pretrained TabICL v2 (Qu et al. 2025) over the relational context of each pair, then fine-tuned on KION.

    python3 4_neural/train_relational.py {softmax,counts,gates}

The table of a pair (u, p) is built by contexts.py: labeled pairs of the window before, (u, f) for
films f and (v, p) for users v, each labeled with whether it was watched in that window, then the
pair itself as the last row. The context asks the same question the pair asks. No hand features.

TabICL reads a table column by column (distribution of each feature), then row by row (between
the features of a row), then across rows (between pairs). softmax keeps TabICL as pretrained;
counts swaps its attention between rows for SiLU / n (counting_attention.py), so the number of
matching rows reaches the output; gates is softmax with a stochastic gate on every column
(stochastic_gates.py), the penalty LAMBDA · open gates in the loss, so the report says which
columns the model needs. All three fine-tune from the same weights.

Epoch 0 is the pretrained model as is (zero-shot), every epoch after it is STEPS fine-tuning
steps; the epoch with the best val ndcg@10 on VAL_USERS users is scored on the test users of
metrics.eval_users, the same users every other model reports as subset_*.

Batches hold half positive pairs and half negative ones, since one random batch has 0.1 positive
on average; the score adds back the log odds of the real positive rate, so log loss stays honest.
TabICL stays in train mode throughout (it has no dropout; its eval mode switches to a cached
inference path), only the gates switch their noise off for scoring.
After training, probes.r2 reads the state of the pair row (the input of TabICL's decoder) of the
best epoch and of the pretrained model, so the report shows which hand aggregates fine-tuning taught.
"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tabicl._model.tabicl import TabICL

import contexts
import counting_attention
import metrics
import probes
import stochastic_gates

VARIANT = sys.argv[1]
NAME = f"tabicl_{VARIANT}"
EPOCHS = int(os.environ.get("EPOCHS", 20))
STEPS = int(os.environ.get("STEPS", 200))
VAL_USERS = int(os.environ.get("VAL_USERS", 50))
BATCH = 16
SCORE_BATCH = 128
LR = 1e-5
GATE_LR = 1e-2
LAMBDA = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
rng = np.random.default_rng(0)


def pretrained(variant):
    checkpoint = torch.load(
        hf_hub_download("jingang/TabICL", "tabicl-classifier-v2-20260212.ckpt"), map_location="cpu", weights_only=True
    )
    model = TabICL(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])

    if variant == "counts":
        counting_attention.use(model)
    model.gates = stochastic_gates.Gates(len(contexts.COLUMNS)) if variant == "gates" else torch.nn.Identity()
    return model.to(DEVICE).train()


def logit(model, stage, pairs):
    X_BTH, y_BT, label_B = stage.batch(pairs)
    out_B1C = model(model.gates(X_BTH.to(DEVICE)), y_BT.to(DEVICE))
    return out_B1C[:, -1, 1] - out_B1C[:, -1, 0], label_B.to(DEVICE)


def score(model, stage, users, shift):
    """probability of watched for every pair of `users`, the balanced-batch prior taken back out."""
    pairs = np.flatnonzero(np.isin(stage.pair_user, users))

    model.gates.eval()
    with torch.no_grad():
        parts = [logit(model, stage, pairs[i : i + SCORE_BATCH])[0].cpu() for i in range(0, len(pairs), SCORE_BATCH)]
    model.gates.train()

    frame = stage.frame.iloc[pairs].copy()
    frame["score"] = torch.sigmoid(torch.cat(parts) + shift).numpy()
    return frame


def pair_state(model, stage, rows):
    """state of the pair row after the in-context blocks, the input of TabICL's decoder."""
    captured = []
    hook = model.icl_predictor.decoder.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0][:, -1].cpu()))
    model.gates.eval()
    with torch.no_grad():
        for i in range(0, len(rows), SCORE_BATCH):
            logit(model, stage, rows[i : i + SCORE_BATCH])
    model.gates.train()
    hook.remove()
    return torch.cat(captured).numpy()


fit, val, test = contexts.Stage("fit"), contexts.Stage("val"), contexts.Stage("test")
val_users = np.random.default_rng(0).choice(np.unique(val.pair_user), VAL_USERS, replace=False)
test_users = metrics.eval_users(test.frame)
positive, negative = np.flatnonzero(fit.label == 1), np.flatnonzero(fit.label == 0)
prior_shift = float(np.log(fit.label.mean() / (1 - fit.label.mean())))

model = pretrained(VARIANT)
gate_parameters = list(model.gates.parameters())
tabicl_parameters = [p for name, p in model.named_parameters() if not name.startswith("gates.")]
optimizer = torch.optim.AdamW(
    [{"params": tabicl_parameters, "lr": LR}, {"params": gate_parameters, "lr": GATE_LR, "weight_decay": 0.0}], weight_decay=0.01
)

best, best_state, best_shift, curve = -1.0, None, 0.0, []
for epoch in range(EPOCHS + 1):
    shift = prior_shift if epoch > 0 else 0.0
    for step in range(STEPS if epoch > 0 else 0):
        pairs = np.concatenate([rng.choice(positive, BATCH // 2), rng.choice(negative, BATCH // 2)])
        logit_B, label_B = logit(model, fit, pairs)
        loss = F.binary_cross_entropy_with_logits(logit_B, label_B.float())
        if VARIANT == "gates":
            loss = loss + LAMBDA * model.gates.open()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch}  step {step + 1:3d}  loss {loss.item():.4f}", flush=True)

    ndcg = metrics.ranking(score(model, val, val_users, shift), "score")["ndcg@10"]
    curve.append(ndcg)
    print(f"epoch {epoch}  val ndcg@10 {ndcg:.4f}", flush=True)
    if ndcg > best:
        best, best_state, best_shift = ndcg, copy.deepcopy(model.state_dict()), shift

model.load_state_dict(best_state)
probe_rows, probe_table = probes.pairs()
extras = {
    "context_rows": contexts.ROWS,
    "val_users": VAL_USERS,
    "val_ndcg@10_curve": curve,
    "best_epoch": int(np.argmax(curve)),
    "parameters": sum(p.numel() for p in model.parameters()),
    "probe_r2": probes.r2(pair_state(model, val, probe_rows), probe_table),
    "probe_r2_pretrained": probes.r2(pair_state(pretrained("softmax" if VARIANT == "gates" else VARIANT), val, probe_rows), probe_table),
}
if VARIANT == "gates":
    extras["gates"] = dict(zip(contexts.COLUMNS, model.gates.mu.clamp(0, 1).tolist()))
torch.save(best_state, f"models/{NAME}.pt")
metrics.report(NAME, score(model, test, test_users, best_shift), "score", extras)

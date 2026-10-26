"""Autoregressive transformer over the user's history, with the boosting's split, pool and label.

    python3 4_neural/train_neural.py transformer

Trained on the fit window (input: history before it, target: watched pool items in it),
training stops PATIENCE epochs after the best val ndcg@10 (at most EPOCHS), the val ndcg@10 curve
goes into the report, and the epoch with the best val ndcg@10 is scored on test. No hand features.
The transformer is autoregressive (SASRec, Kang & McAuley 2018):
besides the window label its loss holds the cross-entropy of the next event's item at every slot
of the history, scored against the same item vectors that score the pool.
"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import metrics
from encoders import real_slots
import probes
import recommender
import stage

ENCODER = sys.argv[1]
EPOCHS = int(os.environ.get("EPOCHS", 30))
PATIENCE = 3
BATCH = 256
LR = 1e-3
SCORE_CHUNK = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ITEMS = stage.SHAPE[1]

torch.manual_seed(0)
rng = np.random.default_rng(0)


def load(name):
    start, history, scored_users, pool_items, frame = stage.prepare(name)
    inputs = recommender.histories(history, start, scored_users, N_ITEMS)

    label_UP = torch.tensor(
        frame["label"].to_numpy().reshape(len(scored_users), len(pool_items)), dtype=torch.float32
    )

    inputs = {name: value.to(DEVICE) for name, value in inputs.items()}
    return inputs, torch.tensor(pool_items, dtype=torch.long, device=DEVICE), label_UP.to(DEVICE), frame


def take(inputs, index):
    return {name: value[index] for name, value in inputs.items()}


def score(model, inputs, pool_P):
    model.eval()
    users = len(inputs["length"])

    with torch.no_grad():
        chunks = [
            torch.sigmoid(model(take(inputs, slice(i, i + SCORE_CHUNK)), pool_P))
            for i in range(0, users, SCORE_CHUNK)
        ]
    return torch.cat(chunks).cpu().numpy().ravel()


def pair_state(model, inputs, pool_P, rows):
    """state of the val pairs `rows` (frame order: user-major, pool-minor) that the last layer turns into the logit."""
    pool = len(pool_P)
    user_N, item_N = rows // pool, rows % pool

    users = torch.from_numpy(np.unique(user_N)).to(DEVICE)

    model.eval()
    with torch.no_grad():
        state_UPH = torch.cat([model.pair_state(take(inputs, users[i : i + BATCH]), pool_P) for i in range(0, len(users), BATCH)])
    return state_UPH[np.searchsorted(users.cpu().numpy(), user_N), item_N].cpu().numpy()


def next_event_loss(model, batch):
    """cross-entropy of the item of the next event at every slot of the history."""
    hidden_ULD = model.encoder.sequence(batch)[:, :-1]
    next_UL = batch["item"][:, 1:]
    real_UL = real_slots(batch)[:, 1:]

    logit_RI = hidden_ULD[real_UL] @ model.item.weight.T
    return F.cross_entropy(logit_RI, next_UL[real_UL])


fit_inputs, fit_pool_P, fit_label_UP, _ = load("fit")
val_inputs, val_pool_P, _, val_frame = load("val")
test_inputs, test_pool_P, _, test_frame = load("test")

model = recommender.Recommender(N_ITEMS, ENCODER).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_ndcg, best_epoch, best_state = -1.0, 0, None
val_curve = []

for epoch in range(EPOCHS):
    model.train()

    losses = []

    for batch in np.array_split(rng.permutation(len(fit_label_UP)), max(1, len(fit_label_UP) // BATCH)):

        batch = torch.from_numpy(batch).to(DEVICE)
        logit_UP = model(take(fit_inputs, batch), fit_pool_P)

        loss = F.binary_cross_entropy_with_logits(logit_UP, fit_label_UP[batch])

        if ENCODER == "transformer":
            loss = loss + next_event_loss(model, take(fit_inputs, batch))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    val_frame["score"] = score(model, val_inputs, val_pool_P)
    ndcg = metrics.ranking(val_frame, "score")[f"ndcg@{metrics.K}"]
    val_curve.append(ndcg)
    print(f"epoch {epoch + 1:3}  loss {np.mean(losses):.5f}  val ndcg@10 {ndcg:.4f}", flush=True)

    if ndcg > best_ndcg:
        best_ndcg, best_epoch, best_state = ndcg, epoch + 1, copy.deepcopy(model.state_dict())
    if epoch + 1 - best_epoch >= PATIENCE:
        break

model.load_state_dict(best_state)

probe_rows, probe_table = probes.pairs()
probe_r2 = probes.r2(pair_state(model, val_inputs, val_pool_P, probe_rows), probe_table)

test_frame["score"] = score(model, test_inputs, test_pool_P)

metrics.report(
    f"neural_{ENCODER}",
    test_frame,
    "score",
    {
        "best_epoch": best_epoch,
        f"val_ndcg@{metrics.K}_curve": val_curve,
        f"val_ndcg@{metrics.K}": best_ndcg,
        "parameters": sum(p.numel() for p in model.parameters()),
        "probe_r2": probe_r2,
    },
)

torch.save(best_state, f"models/neural_{ENCODER}.pt")

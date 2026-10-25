"""Neural baselines with the boosting's split, pool and label, one training loop for all.

    python3 4_neural/train_neural.py {autoencoder,bag,lstm,transformer,cross}

Trained on the fit window (input: history before it, target: watched pool items in it),
every encoder trains for the same EPOCHS, the val ndcg@10 curve goes into the report,
and the epoch with the best val ndcg@10 is scored on test. No hand features.
autoencoder and bag see no order of events, so lstm or transformer minus bag is
what the order gives on its own. cross is the main model of notes §6 (recommender.Cross).
"""
import copy
import sys

import numpy as np
import torch
import torch.nn.functional as F

import metrics
import recommender
import stage

ENCODER = sys.argv[1]
EPOCHS = 30
BATCH = 256
LR = 1e-3
SCORE_CHUNK = 4096
N_ITEMS = stage.SHAPE[1]

torch.manual_seed(0)
rng = np.random.default_rng(0)


def load(name):
    start, history, scored_users, pool_items, frame = stage.prepare(name)
    inputs = recommender.histories(history, start, scored_users, N_ITEMS)
    label_UP = torch.tensor(
        frame["label"].to_numpy().reshape(len(scored_users), len(pool_items)), dtype=torch.float32
    )
    stats_PS = recommender.item_stats(history, start, pool_items)
    return inputs, (torch.tensor(pool_items, dtype=torch.long), stats_PS), label_UP, frame


def take(inputs, index):
    return {name: value[index] for name, value in inputs.items()}


def score(model, inputs, pool_P):
    model.eval()
    users = len(inputs["length"])
    with torch.no_grad():
        chunks = [
            torch.sigmoid(model(take(inputs, slice(i, i + SCORE_CHUNK)), *pool_P))
            for i in range(0, users, SCORE_CHUNK)
        ]
    return torch.cat(chunks).numpy().ravel()


fit_inputs, fit_pool_P, fit_label_UP, _ = load("fit")
val_inputs, val_pool_P, _, val_frame = load("val")
test_inputs, test_pool_P, _, test_frame = load("test")

if ENCODER == "cross":
    codes = lambda column: torch.tensor(stage.items.set_index("item")[column].reindex(range(N_ITEMS)).factorize()[0])
    model = recommender.Cross(N_ITEMS, codes("genre"), codes("country"))
else:
    model = recommender.Recommender(N_ITEMS, ENCODER)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_ndcg, best_epoch, best_state = -1.0, 0, None
val_curve = []

for epoch in range(EPOCHS):
    model.train()
    losses = []
    for batch in np.array_split(rng.permutation(len(fit_label_UP)), max(1, len(fit_label_UP) // BATCH)):
        batch = torch.from_numpy(batch)
        logit_UP = model(take(fit_inputs, batch), *fit_pool_P)
        loss = F.binary_cross_entropy_with_logits(logit_UP, fit_label_UP[batch])
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

model.load_state_dict(best_state)
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
    },
)

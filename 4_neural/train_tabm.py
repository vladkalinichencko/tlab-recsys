"""TabM (Gorishniy et al. 2025) on the same rows and features as boosting `all`.

    python3 4_neural/train_tabm.py [all|auto]

One row = user x pool item, label = watched in the window, binary cross-entropy on each of the K
ensemble members, the score is the mean probability. Numbers go through a quantile transform fitted
on fit, NaN becomes 0, categories become indices with one extra index for unseen values.
The epoch with the best val ndcg@10 is kept, as the tree with the best val ndcg@10 in boosting.
"""
import copy
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from tabm import TabM

import boosting
import grammar
import metrics
import stage

FEATURES = sys.argv[1] if len(sys.argv) > 1 else "all"
NAME = f"tabm_{FEATURES}"
K = 32
EPOCHS = 30
BATCH = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

specs = list(grammar.all_specs()) if FEATURES == "auto" else ()
fit, validation, test = (stage.table(name, specs) for name in ("fit", "val", "test"))
columns = boosting.feature_set(FEATURES, fit)
categorical = [column for column in boosting.CATEGORICAL if column in columns]
numeric = [column for column in columns if column not in categorical]

quantiles = QuantileTransformer(output_distribution="normal", subsample=1_000_000, random_state=0)
quantiles.fit(fit[numeric].to_numpy(np.float32))
vocabulary = {column: {value: i for i, value in enumerate(fit[column].astype(str).unique())} for column in categorical}


def tensors(table):
    x_num = np.nan_to_num(quantiles.transform(table[numeric].to_numpy(np.float32))).astype(np.float32)
    x_cat = np.stack(
        [table[column].astype(str).map(vocabulary[column]).fillna(len(vocabulary[column])).to_numpy(np.int64) for column in categorical],
        axis=1,
    )
    return torch.from_numpy(x_num), torch.from_numpy(x_cat)


def score(model, x_num, x_cat):
    model.eval()
    with torch.no_grad():
        parts = [
            torch.sigmoid(model(x_num[i : i + BATCH].to(DEVICE), x_cat[i : i + BATCH].to(DEVICE))).mean(1).squeeze(-1).cpu()
            for i in range(0, len(x_num), BATCH)
        ]
    return torch.cat(parts).numpy()


fit_num, fit_cat = tensors(fit)
val_num, val_cat = tensors(validation)
test_num, test_cat = tensors(test)
label = torch.from_numpy(fit["label"].to_numpy(np.float32))

model = TabM.make(
    n_num_features=len(numeric),
    cat_cardinalities=[len(vocabulary[column]) + 1 for column in categorical],
    d_out=1,
    k=K,
).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)

best, best_state, curve = -1.0, None, []
for epoch in range(EPOCHS):
    model.train()
    order = torch.randperm(len(label))
    for i in range(0, len(order), BATCH):
        rows = order[i : i + BATCH]
        logits = model(fit_num[rows].to(DEVICE), fit_cat[rows].to(DEVICE)).squeeze(-1)  # (B, K)
        loss = F.binary_cross_entropy_with_logits(logits, label[rows, None].expand(-1, K).to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    validation["score"] = score(model, val_num, val_cat)
    ndcg = metrics.ranking(validation, "score")["ndcg@10"]
    curve.append(ndcg)
    print(f"epoch {epoch + 1:3d}  loss {loss.item():.5f}  val ndcg@10 {ndcg:.4f}", flush=True)
    if ndcg > best:
        best, best_state = ndcg, copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
validation["score"] = score(model, val_num, val_cat)
test["score"] = score(model, test_num, test_cat)
metrics.report(
    NAME,
    test,
    "score",
    {
        "features": len(columns),
        "val_ndcg@10_curve": curve,
        **{f"val_{key}": value for key, value in metrics.ranking(validation, "score").items()},
    },
)

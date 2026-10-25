"""OpenFE (Zhang et al. 2023) on the grammar features: which relational aggregates to add to `all`.

    python3 3_boosting/select_openfe.py   ->  logs/openfe_selected.json

Every candidate from grammar.all_specs is scored by how much it improves the base model alone.
The base model is LightGBM on the `all` features; out-of-fold on fit, averaged over the folds on val.
Stage 1  one LightGBM per candidate on that single column, starting from the base predictions;
         the score is the drop of val log loss. Successive halving: the round with 2^r user blocks
         keeps the better half, the last round sees all blocks, a candidate needs a positive score.
Stage 2  one LightGBM on `all` plus the survivors from the base predictions, the survivors ranked
         by split gain, the top TOP are kept.
Selection runs on USERS random users of fit and val, the test window is never read.
"""
import json

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

import boosting
import grammar
import stage

USERS = 2000
BLOCKS = 8
FOLDS = 5
TOP = 50
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=16, verbose=-1, n_jobs=4, seed=0)


def to_lightgbm(table, columns):
    frame = table[columns].copy()
    for column in boosting.CATEGORICAL:
        if column in columns:
            frame[column] = frame[column].astype(str).astype("category")
    return frame


def boost(columns, fit, val, init_fit, init_val, patience):
    train = lgb.Dataset(to_lightgbm(fit, columns), fit["label"], init_score=init_fit)
    valid = lgb.Dataset(to_lightgbm(val, columns), val["label"], init_score=init_val, reference=train)
    return lgb.train(PARAMS, train, 1000, valid_sets=[valid], callbacks=[lgb.early_stopping(patience, verbose=False)])


def logloss(label, raw):
    probability = np.clip(1 / (1 + np.exp(-raw)), 1e-7, 1 - 1e-7)
    return -np.mean(label * np.log(probability) + (1 - label) * np.log(1 - probability))


specs = list(grammar.all_specs())
fit = stage.table("fit", specs, USERS)
val = stage.table("val", specs, USERS)
names = [grammar.name(spec) for spec in specs]
base = [column for column in boosting.feature_set("all", fit) if column not in names]

# base predictions: out-of-fold on fit, mean of the fold models on val
fit["base"], val["base"] = 0.0, 0.0
for train_rows, held_rows in GroupKFold(FOLDS).split(fit, groups=fit["user"]):
    model = boost(base, fit.iloc[train_rows], fit.iloc[held_rows], None, None, 50)
    fit.loc[fit.index[held_rows], "base"] = model.predict(to_lightgbm(fit.iloc[held_rows], base), raw_score=True)
    val["base"] += model.predict(to_lightgbm(val, base), raw_score=True) / FOLDS
print(f"base val logloss {logloss(val['label'], val['base']):.5f}  candidates {len(names)}")

# stage 1: successive featurewise halving over user blocks
block_of = {user: i % BLOCKS for i, user in enumerate(np.random.default_rng(0).permutation(fit["user"].unique()))}
block_of |= {user: i % BLOCKS for i, user in enumerate(np.random.default_rng(1).permutation(val["user"].unique()))}
survivors, blocks = names, 1
while True:
    fit_part = fit[fit["user"].map(block_of) < blocks]
    val_part = val[val["user"].map(block_of) < blocks]
    before = logloss(val_part["label"], val_part["base"])
    score = {}
    for column in survivors:
        model = boost([column], fit_part, val_part, fit_part["base"], val_part["base"], 3)
        score[column] = before - model.best_score["valid_0"]["binary_logloss"]
    survivors = sorted(survivors, key=score.get, reverse=True)
    print(f"stage 1  blocks {blocks}/{BLOCKS}  candidates {len(survivors)}  best {survivors[0]} {score[survivors[0]]:.6f}")
    if blocks == BLOCKS:
        survivors = [column for column in survivors if score[column] > 0]
        break
    survivors = survivors[: max(len(survivors) // 2, TOP)]
    blocks *= 2

# stage 2: gain of the survivors next to the base features
model = boost(base + survivors, fit, val, fit["base"], val["base"], 50)
gain = dict(zip(model.feature_name(), model.feature_importance("gain")))
selected = [column for column in sorted(survivors, key=gain.get, reverse=True)[:TOP] if gain[column] > 0]
for column in selected:
    print(f"{column:40s} stage1 {score[column]:.6f}  gain {gain[column]:.1f}")

by_name = dict(zip(names, specs))
json.dump([by_name[column] for column in selected], open("logs/openfe_selected.json", "w"), indent=1)
print(f"selected {len(selected)} of {len(names)}")

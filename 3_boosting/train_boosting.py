"""Gradient boosting on hand features: which loss, which feature set.

    python3 3_boosting/train_boosting.py {pointwise,yetirank} {all,hand,within_user,grammar,auto,openfe}

pointwise  binary log loss on each pair, the score is a probability
yetirank   listwise loss inside each user's group
Feature sets are in boosting.py. Every run is the same CatBoost with the same number of trees,
the tree with the best val ndcg@10 is kept, and the val curve is in logs/catboost/<name>,
so two runs differ by the loss or by the feature set alone.
"""
import sys

import numpy as np
from catboost import CatBoost

import boosting
import grammar
import metrics
import stage

LOSS, FEATURES = sys.argv[1], sys.argv[2]
NAME = f"boosting_{LOSS}_{FEATURES}"
LOSSES = {"pointwise": "Logloss", "yetirank": "YetiRank"}
PARAMS = dict(
    loss_function=LOSSES[LOSS],
    eval_metric="NDCG:top=10;type=Exp",
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    use_best_model=True,
    random_seed=0,
    verbose=50,
    train_dir=f"logs/catboost/{NAME}",
)


def score(model, table):
    raw = model.predict(boosting.pool(table, columns))
    return 1.0 / (1.0 + np.exp(-raw)) if LOSS == "pointwise" else raw


specs = {"grammar": list(grammar.all_specs()), "auto": list(grammar.all_specs()), "openfe": grammar.load("logs/openfe_selected.json") if FEATURES == "openfe" else []}.get(FEATURES, ())
fit, validation, test = (stage.table(name, specs) for name in ("fit", "val", "test"))
columns = boosting.feature_set(FEATURES, fit)

model = CatBoost(PARAMS).fit(boosting.pool(fit, columns), eval_set=boosting.pool(validation, columns))
model.save_model(f"models/{NAME}.cbm")

validation["score"] = score(model, validation)
test["score"] = score(model, test)
metrics.report(
    NAME,
    test,
    "score",
    {
        "trees": model.tree_count_,
        "features": len(columns),
        **{f"val_{key}": value for key, value in metrics.ranking(validation, "score").items()},
    },
)

"""Gradient boosting with the listwise YetiRank loss inside each user's group: which feature set.

    python3 3_boosting/train_boosting.py {all,hand,auto}

Feature sets are in boosting.py. Every run is the same CatBoost with the same number of trees,
the tree with the best val ndcg@10 is kept, and the val curve is in logs/catboost/<name>,
so two runs differ by the feature set alone.
"""
import sys

from catboost import CatBoost

import boosting
import grammar
import metrics
import stage

FEATURES = sys.argv[1]
NAME = f"boosting_yetirank_{FEATURES}"
PARAMS = dict(
    loss_function="YetiRank",
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
    return model.predict(boosting.pool(table, columns))


specs = list(grammar.all_specs()) if FEATURES == "auto" else ()
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

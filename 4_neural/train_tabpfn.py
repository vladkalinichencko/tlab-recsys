"""TabPFN in context on automatic features, as RDBLearn and TabPFN-Rel do with DFS aggregates.

    python3 4_neural/train_tabpfn.py [auto|all]

No training: CONTEXT random rows of fit are the context, every val and test row is classified against them,
the score is the probability of the positive class.
"""
import sys

import numpy as np
from tabpfn import TabPFNClassifier

import boosting
import grammar
import metrics
import stage

FEATURES = sys.argv[1] if len(sys.argv) > 1 else "auto"
NAME = f"tabpfn_{FEATURES}"
CONTEXT = 10_000
BATCH = 5_000

specs = list(grammar.all_specs()) if FEATURES == "auto" else ()
fit, validation, test = (stage.table(name, specs) for name in ("fit", "val", "test"))
columns = boosting.feature_set(FEATURES, fit)


def matrix(table):
    frame = table[columns].copy()
    for column in boosting.CATEGORICAL:
        if column in columns:
            frame[column] = frame[column].astype(str).astype("category").cat.codes
    return frame.to_numpy(np.float32)


context = fit.sample(min(CONTEXT, len(fit)), random_state=0)
model = TabPFNClassifier(ignore_pretraining_limits=True, random_state=0).fit(matrix(context), context["label"].to_numpy())


def score(table):
    rows = matrix(table)
    return np.concatenate([model.predict_proba(rows[i : i + BATCH])[:, 1] for i in range(0, len(rows), BATCH)])


validation["score"] = score(validation)
test["score"] = score(test)
metrics.report(
    NAME,
    test,
    "score",
    {"features": len(columns), "context": len(context), **{f"val_{k}": v for k, v in metrics.ranking(validation, "score").items()}},
)

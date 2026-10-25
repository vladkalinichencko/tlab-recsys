"""Which hand features carry the ranking: permutation importance in val ndcg@10.

Loads models/boosting_yetirank_hand.cbm (python3 3_boosting/train_boosting.py yetirank hand),
shuffles one column at a time over all val users and records how much ndcg@10 drops.
Writes reports/importance.json.
"""
import json

import numpy as np
from catboost import CatBoost

import boosting
import metrics
import stage

model = CatBoost().load_model("models/boosting_yetirank_hand.cbm")
columns = model.feature_names_
validation = stage.table("val")
rng = np.random.default_rng(0)


def ndcg(table):
    table = table.assign(score=model.predict(boosting.pool(table, columns)))
    return metrics.ranking(table, "score")[f"ndcg@{metrics.K}"]


baseline = ndcg(validation)
drops = {}
for column in columns:
    shuffled = validation.copy()
    shuffled[column] = rng.permutation(shuffled[column].to_numpy())
    drops[column] = baseline - ndcg(shuffled)
    print(f"{drops[column]:9.5f}  {column}", flush=True)

with open(f"{metrics.REPORTS}/importance.json", "w") as file:
    json.dump(
        {"baseline": baseline, "users": int(validation["user"].nunique()), "drops": drops},
        file,
        indent=2,
    )

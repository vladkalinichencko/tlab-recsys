import json

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

K = 10
REPORTS = "reports"
# the same random test users for every model, so models too slow for the full test (TabICL) compare with the rest
EVAL_USERS = 200


def eval_users(candidates):
    users = np.sort(candidates["user"].unique())
    return np.random.default_rng(0).choice(users, min(EVAL_USERS, len(users)), replace=False)


def ranking(candidates, score, k=K):
    ranked = candidates[["user", "label", score]].sort_values(
        ["user", score], ascending=[True, False]
    )
    ranked["rank"] = ranked.groupby("user").cumcount() + 1

    top = ranked[ranked["rank"] <= k].copy()
    top["gain"] = top["label"] / np.log2(top["rank"] + 1)
    top["hit_precision"] = (
        top.groupby("user")["label"].cumsum() / top["rank"] * top["label"]
    )
    per_user = top.groupby("user").agg(
        dcg=("gain", "sum"), hits=("label", "sum"), ap=("hit_precision", "sum")
    )

    positive = ranked.groupby("user")["label"].sum()
    per_user = per_user.reindex(positive.index).fillna(0)
    reachable = positive.clip(upper=k).to_numpy().astype(int)
    ideal = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(1, k + 1) + 1))])

    return {
        f"ndcg@{k}": float(np.mean(per_user["dcg"].to_numpy() / ideal[reachable])),
        f"recall@{k}": float(np.mean(per_user["hits"].to_numpy() / positive.to_numpy())),
        f"map@{k}": float(np.mean(per_user["ap"].to_numpy() / reachable)),
    }


def scores(candidates, score):
    label = candidates["label"].to_numpy()
    value = candidates[score].to_numpy().astype(np.float64)
    calibrated = value.min() >= 0.0 and value.max() <= 1.0
    return {
        "users": int(candidates["user"].nunique()),
        "candidates": len(candidates),
        "positive_rate": float(label.mean()),
        "roc_auc": float(roc_auc_score(label, value)),
        "logloss": float(log_loss(label, value)) if calibrated else None,
        **ranking(candidates, score),
    }


def report(name, candidates, score, extra=None):
    subset = candidates[candidates["user"].isin(eval_users(candidates))]
    result = {
        "model": name,
        **scores(candidates, score),
        **{f"subset_{key}": value for key, value in scores(subset, score).items()},
        **(extra or {}),
    }
    candidates[["user", "item", "label", score]].to_parquet(f"{REPORTS}/{name}_scores.parquet")
    with open(f"{REPORTS}/{name}.json", "w") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))
    return result

import numpy as np
import pandas as pd
import scipy.sparse as sp

POOL = 300
EASE_REG = 500.0


def matrix(events, shape):
    watched = events[events["watched"]]
    return sp.csr_matrix(
        (
            np.ones(len(watched), dtype=np.float32),
            (watched["user"].to_numpy(), watched["item"].to_numpy()),
        ),
        shape=shape,
    )


def popular(history, k=POOL):
    watched = history[history["watched"]]
    return (
        watched.groupby("item")["user"]
        .nunique()
        .sort_values(ascending=False)
        .head(k)
        .rename("pop")
    )


def select(history, target, k=POOL):
    items = popular(history, k)
    reachable = target[target["item"].isin(items.index)]
    return np.sort(reachable["user"].unique()), items.index.to_numpy()


def frame(users, items, target, shape):
    labels = matrix(target, shape)[users][:, items]
    return pd.DataFrame(
        {
            "user": np.repeat(users, len(items)),
            "item": np.tile(items, len(users)),
            "label": np.asarray(labels.todense()).ravel().clip(0, 1).astype(np.int8),
        }
    )


def ease(X, reg=EASE_REG):
    gram = np.asarray((X.T @ X).todense(), dtype=np.float64)
    gram[np.diag_indices_from(gram)] += reg
    inverse = np.linalg.inv(gram)
    weights = -inverse / np.diag(inverse)
    weights[np.diag_indices_from(weights)] = 0.0
    return weights.astype(np.float32)


def item_knn(X):
    counts = np.asarray((X.T @ X).todense(), dtype=np.float32)
    norm = np.sqrt(np.diag(counts)) + 1e-6
    similarity = counts / np.outer(norm, norm)
    similarity[np.diag_indices_from(similarity)] = 0.0
    return similarity


def scores(users, items, X, weights):
    return np.asarray(X[users] @ weights[:, items]).ravel()

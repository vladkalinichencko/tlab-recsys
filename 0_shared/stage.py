import numpy as np

import candidates
import features
import grammar
import kion

events, users, items = kion.load()
SHAPE = (len(users), len(items))
STAGES = kion.stages(events)


def prepare(name):
    start, end = STAGES[name]
    history = kion.history(events, start)
    target = kion.target(events, start, end)
    scored_users, pool_items = candidates.select(history, target)
    frame = candidates.frame(scored_users, pool_items, target, SHAPE)
    print(
        f"{name}: days [{start},{end})  "
        f"users {len(scored_users)}/{target['user'].nunique()} reachable  "
        f"pool {len(pool_items)}  rows {len(frame)}"
    )
    return start, history, scored_users, pool_items, frame


def table(name, specs=(), max_users=None):
    """specs adds grammar features, max_users keeps a random subset of users (for feature selection)."""
    start, history, scored_users, pool_items, frame = prepare(name)
    if max_users is not None and len(scored_users) > max_users:
        scored_users = np.sort(np.random.default_rng(0).choice(scored_users, max_users, replace=False))
        frame = frame[frame["user"].isin(scored_users)].reset_index(drop=True)
    interactions = candidates.matrix(history, SHAPE)
    frame["ease"] = candidates.scores(
        scored_users, pool_items, interactions, candidates.ease(interactions)
    )
    frame["knn"] = candidates.scores(
        scored_users, pool_items, interactions, candidates.item_knn(interactions)
    )
    if specs:
        frame = grammar.build(frame, history, start, specs, items)
    return features.build(frame, history, start, users, items)


import candidates
import features
import grammar
import windows

events, users, items = windows.load()
SHAPE = (len(users), len(items))
STAGES = windows.stages(events)


def prepare(name):
    start, end = STAGES[name]
    history = windows.history(events, start)
    target = windows.target(events, start, end)
    scored_users, pool_items = candidates.select(history, target)

    frame = candidates.frame(scored_users, pool_items, target, SHAPE)

    print(
        f"{name}: days [{start},{end})  "
        f"users {len(scored_users)}/{target['user'].nunique()} reachable  "
        f"pool {len(pool_items)}  rows {len(frame)}"
    )
    return start, history, scored_users, pool_items, frame


def table(name, specs=()):
    """specs adds grammar features."""
    start, history, scored_users, pool_items, frame = prepare(name)
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

"""Features see only events before the window, the target only events inside it, metrics are sane."""
import pandas as pd

import features
import kion
import metrics

START = 3

EVENTS = pd.DataFrame(
    {
        "user": [0, 0, 1, 1],
        "item": [0, 1, 0, 1],
        "day": [1, 2, 2, 5],
        "watched_pct": [100.0, 100.0, 100.0, 100.0],
        "watched": [True, True, True, True],
    }
)
USERS = pd.DataFrame(
    {
        "user": [0, 1],
        "age": ["a", "b"],
        "income": ["a", "b"],
        "sex": ["m", "f"],
        "kids_flg": [0, 0],
    }
)
ITEMS = pd.DataFrame(
    {
        "item": [0, 1],
        "content_type": ["film", "film"],
        "release_year": [2000, 2001],
        "age_rating": [0, 0],
        "genre": ["drama", "comedy"],
        "country": ["ru", "ru"],
    }
)
CANDIDATES = pd.DataFrame({"user": [1], "item": [1], "label": [1]})

past = features.build(CANDIDATES, kion.history(EVENTS, START), START, USERS, ITEMS)
dropped = features.build(
    CANDIDATES, kion.history(EVENTS[EVENTS["day"] != 5], START), START, USERS, ITEMS
)
whole = features.build(CANDIDATES, EVENTS, START, USERS, ITEMS)

assert past.equals(dropped)
assert not past.equals(whole)
assert past["ui_watched_all"].item() == 0
assert whole["ui_watched_all"].item() == 1
assert kion.target(EVENTS, START, 6)["day"].to_list() == [5]

perfect = pd.DataFrame(
    {"user": [0, 0, 0], "item": [0, 1, 2], "label": [1, 1, 0], "score": [0.9, 0.8, 0.1]}
)
assert metrics.ranking(perfect, "score")["ndcg@10"] == 1.0
assert metrics.ranking(perfect, "score")["recall@10"] == 1.0

reversed_order = perfect.assign(score=[0.1, 0.2, 0.9])
assert metrics.ranking(reversed_order, "score")["ndcg@10"] < 1.0

print("future events excluded, target window isolated, ranking metrics calibrated")

"""Relational feature grammar: entity x event column x aggregate x window, one aggregate per spec.

A spec is (entity, column, aggregate, window). Window None is the whole history,
age is days from the event to the window start, count ignores the column.
"""
import itertools
import json

import numpy as np
import pandas as pd

ENTITIES = {
    "user": ["user"],
    "item": ["item"],
    "pair": ["user", "item"],
    "user_genre": ["user", "genre"],
    "user_country": ["user", "country"],
}
COLUMNS = ("watched", "watched_pct", "age")
AGGREGATES = ("sum", "mean", "max", "min", "std")
WINDOWS = (1, 3, 7, 14, 30, None)


def name(spec):
    return "g." + ".".join(map(str, spec))


def all_specs():
    for entity, window in itertools.product(ENTITIES, WINDOWS):
        yield (entity, "events", "count", window)
        for column, aggregate in itertools.product(COLUMNS, AGGREGATES):
            yield (entity, column, aggregate, window)


def load(path):
    return [tuple(spec) for spec in json.load(open(path))]


def build(frame, history, start, specs, items):
    attributes = items[["item", "genre", "country"]]
    history = history.merge(attributes, on="item")
    history["watched"] = history["watched"].astype(np.float32)
    history["age"] = (start - history["day"]).astype(np.float32)
    table = frame.merge(attributes, on="item", how="left")

    blocks = {}
    for spec in specs:
        blocks.setdefault((spec[0], spec[3]), []).append(spec)
    for (entity, window), block in blocks.items():
        keys = ENTITIES[entity]
        recent = history if window is None else history[history["day"] >= start - window]
        grouped = recent.groupby(keys)
        stats = pd.DataFrame(
            {name(spec): grouped.size() if spec[2] == "count" else grouped[spec[1]].agg(spec[2]) for spec in block}
        ).astype(np.float32)
        table = table.merge(stats.reset_index(), on=keys, how="left")
        # no events in the window: zero count and sum, the other aggregates stay NaN
        empty = [name(spec) for spec in block if spec[2] in ("count", "sum")]
        table[empty] = table[empty].fillna(0)

    return table.drop(columns=["genre", "country"])

"""One small table per (user, item) pair, the context TabICL reads in place of hand features.

The pair (u, p) asks: does u watch p in the window [start, end). The context rows ask the same
question one window earlier, [start - DAYS, start), where the answer is known:

    CONTEXT rows (u, f)   items f of the earlier pool, label: u watched f in the earlier window
    CONTEXT rows (v, p)   users v active in the earlier window, label: v watched p in it

Half of each block are positives where there are enough of them (positives are rare), the rest
are random. Every row describes the pair (user e, item e) as of the start of its own window:

    same_user   e is u                    same_item     e is p
    same_<g>    group g of e is group g of p, for every group of the schema
    seen        user e watched item e before the window of the row (the cell of the interaction matrix)
    user_events log(1 + events of user e)     item_events   log(1 + events of item e) before that window
    then the user columns of e and the item columns of e of the schema, numbers for CONTEXT_NUMERIC, category codes for the rest

Axes: B pairs, T rows of a table (T = ROWS + 1), R context rows, C rows of one block, H columns.
"""
import numpy as np
import torch

import candidates
import stage
import windows
from schema import dataset

CONTEXT = 32
ROWS = 2 * CONTEXT
DAYS = windows.VAL_DAYS
GROUPS = list(dataset.GROUPS)
ATTRIBUTES = [*dataset.USER_COLUMNS, *dataset.ITEM_COLUMNS]
COLUMNS = ["same_user", "same_item", *[f"same_{group}" for group in GROUPS], "seen", "user_events", "item_events", *ATTRIBUTES]


def codes(table, column):
    return table[column].astype(str).factorize()[0].astype(np.float32)


users = stage.users.set_index("user").reindex(range(stage.SHAPE[0]))
items = stage.items.set_index("item").reindex(range(stage.SHAPE[1]))


def column(table, name):
    if name in dataset.CONTEXT_NUMERIC:
        return table[name].fillna(table[name].median()).to_numpy(np.float32)
    return codes(table, name)


USER_COLUMNS = np.array([column(users, name) for name in dataset.USER_COLUMNS], np.float32).reshape(-1, len(users)).T
ITEM_COLUMNS = np.array([column(items, name) for name in dataset.ITEM_COLUMNS], np.float32).reshape(-1, len(items)).T
GROUP_CODES = [codes(items, group) for group in GROUPS]


def relation(event_user, event_item, user, item):
    """the columns that say how the row (event_user, event_item) relates to the pair (user, item)."""
    return [event_user == user, event_item == item, *[group[event_item] == group[item] for group in GROUP_CODES]]


def standardize(X_BTH):
    """each column of each table to zero mean, unit variance over its context rows, as the TabICL wrapper does.

    a column constant over the context keeps scale 1 (as sklearn's StandardScaler), so the pair row
    differing from it reads 1, not 1e6."""
    context = X_BTH[:, :-1]
    std = context.std(1, keepdim=True)
    return (X_BTH - context.mean(1, keepdim=True)) / torch.where(std > 1e-6, std, 1.0)


def binary(events):
    matrix = candidates.matrix(events, stage.SHAPE)
    matrix.data[:] = 1
    return matrix


class Past:
    """what is known at `start`: who watched what before it, how many events each user and item has, the pool."""

    def __init__(self, start):
        history = windows.history(stage.events, start)
        self.seen = binary(history)
        self.user_events = np.log1p(np.bincount(history["user"], minlength=stage.SHAPE[0])).astype(np.float32)
        self.item_events = np.log1p(np.bincount(history["item"], minlength=stage.SHAPE[1])).astype(np.float32)
        self.pool = candidates.popular(history).index.to_numpy()

    def rows(self, event_user, event_item, user, item):
        seen = np.asarray(self.seen[event_user.ravel(), event_item.ravel()]).reshape(event_user.shape)
        return np.concatenate(
            [
                np.stack(
                    [*relation(event_user, event_item, user, item), seen, self.user_events[event_user], self.item_events[event_item]], -1
                ).astype(np.float32),
                USER_COLUMNS[event_user],
                ITEM_COLUMNS[event_item],
            ],
            -1,
        )


class Stage:
    def __init__(self, name):
        start, _, _, _, self.frame = stage.prepare(name)
        self.now, self.before = Past(start), Past(start - DAYS)
        earlier = windows.target(stage.events, start - DAYS, start)
        self.answer = binary(earlier)
        self.by_user, self.by_item = self.answer.tocsr(), self.answer.tocsc()
        self.active = np.unique(earlier["user"])
        self.pair_user = self.frame["user"].to_numpy()
        self.pair_item = self.frame["item"].to_numpy()
        self.label = self.frame["label"].to_numpy(np.int64)
        self.rng = np.random.default_rng(0)

    def half_positive(self, positives, others):
        """CONTEXT ids: up to CONTEXT / 2 of `positives`, the rest drawn from `others`."""
        positives = self.rng.permutation(positives)[: CONTEXT // 2]
        return np.concatenate([positives, self.rng.choice(others, CONTEXT - len(positives))])

    def context_pairs(self, user_B, item_B):
        """ROWS (user, item) pairs of the earlier window per pair: u with pool items, then active users with p."""
        items_BC = np.stack([self.half_positive(np.intersect1d(self.by_user[u].indices, self.before.pool), self.before.pool) for u in user_B])
        users_BC = np.stack([self.half_positive(self.by_item[:, f].indices, self.active) for f in item_B])
        user_BR = np.concatenate([np.repeat(user_B[:, None], CONTEXT, 1), users_BC], 1)
        item_BR = np.concatenate([items_BC, np.repeat(item_B[:, None], CONTEXT, 1)], 1)
        return user_BR, item_BR

    def batch(self, pairs):
        """tables X_BTH, context labels y_BR and the pair labels for the rows `pairs` of the frame."""
        user_B, item_B = self.pair_user[pairs], self.pair_item[pairs]
        user_BR, item_BR = self.context_pairs(user_B, item_B)

        context_BRH = self.before.rows(user_BR, item_BR, user_B[:, None], item_B[:, None])
        pair_BH = self.now.rows(user_B, item_B, user_B, item_B)
        X_BTH = torch.from_numpy(np.concatenate([context_BRH, pair_BH[:, None]], 1))
        y_BR = np.asarray(self.answer[user_BR.ravel(), item_BR.ravel()]).reshape(user_BR.shape).astype(np.float32)
        return standardize(X_BTH), torch.from_numpy(y_BR), torch.from_numpy(self.label[pairs])

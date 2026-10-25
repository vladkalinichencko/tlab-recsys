"""One small table per (user, film) pair, the context TabICL reads in place of hand features.

The pair (u, p) asks: does u watch p in the window [start, end). The context rows ask the same
question one window earlier, [start - DAYS, start), where the answer is known:

    CONTEXT rows (u, f)   films f of the earlier pool, label: u watched f in the earlier window
    CONTEXT rows (v, p)   users v active in the earlier window, label: v watched p in it

Half of each block are positives where there are enough of them (positives are rare), the rest
are random. Every row describes the pair (user e, film e) as of the start of its own window:

    same_user   e is u          same_film     e is p
    same_genre  genre of e is genre of p      same_country  country of e is country of p
    seen        user e watched film e before the window of the row (the cell of the interaction matrix)
    user_events log(1 + events of user e)     film_events   log(1 + events of film e) before that window
    age, income, sex, kids                    user columns of e
    content_type, release_year, age_rating    film columns of e

Axes: B pairs, T rows of a table (T = ROWS + 1), R context rows, C rows of one block, H columns.
"""
import numpy as np
import torch

import candidates
import kion
import stage

CONTEXT = 32
ROWS = 2 * CONTEXT
DAYS = kion.VAL_DAYS
COLUMNS = [
    "same_user", "same_film", "same_genre", "same_country", "seen", "user_events", "film_events",
    "age", "income", "sex", "kids", "content_type", "release_year", "age_rating",
]


def codes(table, column):
    return table[column].astype(str).factorize()[0].astype(np.float32)


users = stage.users.set_index("user").reindex(range(stage.SHAPE[0]))
films = stage.items.set_index("item").reindex(range(stage.SHAPE[1]))
USER_COLUMNS = np.stack([codes(users, c) for c in ("age", "income", "sex", "kids_flg")], 1)
FILM_COLUMNS = np.stack(
    [codes(films, "content_type"), films["release_year"].fillna(films["release_year"].median()).to_numpy(np.float32), codes(films, "age_rating")], 1
)
GENRE, COUNTRY = codes(films, "genre"), codes(films, "country")


def relation(event_user, event_film, user, film):
    """the columns that say how the row (event_user, event_film) relates to the pair (user, film)."""
    return [
        event_user == user,
        event_film == film,
        GENRE[event_film] == GENRE[film],
        COUNTRY[event_film] == COUNTRY[film],
    ]


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
    """what is known at `start`: who watched what before it, how many events each user and film has, the pool."""

    def __init__(self, start):
        history = kion.history(stage.events, start)
        self.seen = binary(history)
        self.user_events = np.log1p(np.bincount(history["user"], minlength=stage.SHAPE[0])).astype(np.float32)
        self.film_events = np.log1p(np.bincount(history["item"], minlength=stage.SHAPE[1])).astype(np.float32)
        self.pool = candidates.popular(history).index.to_numpy()

    def rows(self, event_user, event_film, user, film):
        seen = np.asarray(self.seen[event_user.ravel(), event_film.ravel()]).reshape(event_user.shape)
        return np.concatenate(
            [
                np.stack(
                    [*relation(event_user, event_film, user, film), seen, self.user_events[event_user], self.film_events[event_film]], -1
                ).astype(np.float32),
                USER_COLUMNS[event_user],
                FILM_COLUMNS[event_film],
            ],
            -1,
        )


class Stage:
    def __init__(self, name):
        start, _, _, _, self.frame = stage.prepare(name)
        self.now, self.before = Past(start), Past(start - DAYS)
        earlier = kion.target(stage.events, start - DAYS, start)
        self.answer = binary(earlier)
        self.by_user, self.by_film = self.answer.tocsr(), self.answer.tocsc()
        self.active = np.unique(earlier["user"])
        self.pair_user = self.frame["user"].to_numpy()
        self.pair_film = self.frame["item"].to_numpy()
        self.label = self.frame["label"].to_numpy(np.int64)
        self.rng = np.random.default_rng(0)

    def half_positive(self, positives, others):
        """CONTEXT ids: up to CONTEXT / 2 of `positives`, the rest drawn from `others`."""
        positives = self.rng.permutation(positives)[: CONTEXT // 2]
        return np.concatenate([positives, self.rng.choice(others, CONTEXT - len(positives))])

    def context_pairs(self, user_B, film_B):
        """ROWS (user, film) pairs of the earlier window per pair: u with pool films, then active users with p."""
        films_BC = np.stack([self.half_positive(np.intersect1d(self.by_user[u].indices, self.before.pool), self.before.pool) for u in user_B])
        users_BC = np.stack([self.half_positive(self.by_film[:, f].indices, self.active) for f in film_B])
        user_BR = np.concatenate([np.repeat(user_B[:, None], CONTEXT, 1), users_BC], 1)
        film_BR = np.concatenate([films_BC, np.repeat(film_B[:, None], CONTEXT, 1)], 1)
        return user_BR, film_BR

    def batch(self, pairs):
        """tables X_BTH, context labels y_BR and the pair labels for the rows `pairs` of the frame."""
        user_B, film_B = self.pair_user[pairs], self.pair_film[pairs]
        user_BR, film_BR = self.context_pairs(user_B, film_B)

        context_BRH = self.before.rows(user_BR, film_BR, user_B[:, None], film_B[:, None])
        pair_BH = self.now.rows(user_B, film_B, user_B, film_B)
        X_BTH = torch.from_numpy(np.concatenate([context_BRH, pair_BH[:, None]], 1))
        y_BR = np.asarray(self.answer[user_BR.ravel(), film_BR.ravel()]).reshape(user_BR.shape).astype(np.float32)
        return standardize(X_BTH), torch.from_numpy(y_BR), torch.from_numpy(self.label[pairs])

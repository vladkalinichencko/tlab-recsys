"""One small table per (user, film) pair, the context TabICL reads in place of hand features.

Rows: the last CONTEXT events of the user before the window (fewer if the user has fewer),
topped up with the last events of the film by all users, then the pair itself as the last row.
Label of a context row: the event was watched. Label of the pair row: watched in the window.
Every row is an event (user e, film e) seen from the pair (u, p):

    days        days from the event to the window start (0 for the pair row)
    same_user   e is u          same_film     e is p
    same_genre  genre of e is genre of p      same_country  country of e is country of p
    user_events log(1 + events of user e)     film_events   log(1 + events of film e), the node degree (PNA)
    age, income, sex, kids                    user columns of e, category codes
    content_type, release_year, age_rating    film columns of e, codes, year as is

Axes: B pairs, T rows of a table (T = 2 * CONTEXT + 1), H columns.
"""
import numpy as np
import torch

import stage

CONTEXT = 32
ROWS = 2 * CONTEXT
COLUMNS = [
    "days", "same_user", "same_film", "same_genre", "same_country", "user_events", "film_events",
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


def last_events(owner, n_owners, depth):
    """index of the last `depth` events of every owner, newest first, -1 where the owner has fewer."""
    order = np.lexsort((-np.arange(len(owner)), owner))  # by owner, newest (latest position) first
    sorted_owner = owner[order]
    first = np.searchsorted(sorted_owner, np.arange(n_owners))
    rank = np.arange(len(order)) - first[sorted_owner]
    keep = rank < depth
    table = np.full((n_owners, depth), -1)
    table[sorted_owner[keep], rank[keep]] = order[keep]
    return table


class Stage:
    def __init__(self, name):
        self.start, history, scored_users, pool_items, self.frame = stage.prepare(name)
        history = history.sort_values("day", kind="stable")
        self.user, self.film = history["user"].to_numpy(), history["item"].to_numpy()
        self.days = (self.start - history["day"].to_numpy()).astype(np.float32)
        self.watched = history["watched"].to_numpy(np.float32)
        self.user_events = np.log1p(np.bincount(self.user, minlength=stage.SHAPE[0])).astype(np.float32)
        self.film_events = np.log1p(np.bincount(self.film, minlength=stage.SHAPE[1])).astype(np.float32)
        self.by_user = last_events(self.user, stage.SHAPE[0], CONTEXT)
        self.by_film = last_events(self.film, stage.SHAPE[1], ROWS)
        assert (self.by_film[pool_items] >= 0).all(), "a pool film has fewer than ROWS events"
        self.pair_user = self.frame["user"].to_numpy()
        self.pair_film = self.frame["item"].to_numpy()
        self.label = self.frame["label"].to_numpy(np.int64)

    def rows(self, event_user, event_film, user, film, days, user_events, film_events):
        return np.concatenate(
            [
                np.stack(
                    [days, event_user == user, event_film == film, GENRE[event_film] == GENRE[film],
                     COUNTRY[event_film] == COUNTRY[film], user_events, film_events], -1
                ).astype(np.float32),
                USER_COLUMNS[event_user],
                FILM_COLUMNS[event_film],
            ],
            -1,
        )

    def batch(self, pairs):
        """tables X_BTH, context labels y_BT (T - 1 rows) and the pair labels for the rows `pairs` of the frame."""
        user_B, film_B = self.pair_user[pairs], self.pair_film[pairs]
        own_BC = self.by_user[user_B]
        n_B = (own_BC >= 0).sum(1)
        slot = np.arange(ROWS)[None, :]
        from_film = np.take_along_axis(self.by_film[film_B], np.clip(slot - n_B[:, None], 0, None), 1)
        event_BR = np.where(slot < n_B[:, None], np.pad(own_BC, ((0, 0), (0, ROWS - CONTEXT))), from_film)

        u, f = user_B[:, None], film_B[:, None]
        context_BRH = self.rows(
            self.user[event_BR], self.film[event_BR], u, f, self.days[event_BR],
            self.user_events[self.user[event_BR]], self.film_events[self.film[event_BR]],
        )
        pair_BH = self.rows(
            user_B, film_B, user_B, film_B, np.zeros(len(pairs), np.float32),
            self.user_events[user_B], self.film_events[film_B],
        )
        X_BTH = torch.from_numpy(np.concatenate([context_BRH, pair_BH[:, None]], 1))
        return standardize(X_BTH), torch.from_numpy(self.watched[event_BR]), torch.from_numpy(self.label[pairs])


def standardize(X_BTH):
    """each column of each table to zero mean, unit variance over its context rows, as the TabICL wrapper does."""
    context = X_BTH[:, :-1]
    return (X_BTH - context.mean(1, keepdim=True)) / (context.std(1, keepdim=True) + 1e-6)

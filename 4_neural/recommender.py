"""Neural recommender: user history -> encoder -> user vector -> logits over the pool.

Axes: U users, L event slots (slot 0 is BOS), D model width, P pool items.
The encoder is swapped by name (encoders.ENCODERS); the history batch and the head stay the same.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from encoders import ENCODERS

# 64 covers 99% of KION users; a Yambda user has a median of 1.7K events after the catalog cut, run_yambda.sh sets 512
MAX_EVENTS = int(os.environ.get("MAX_EVENTS", 64))
DIM = 64
INIT_STD = 0.02
STATS = 3  # log(1 + watched events of the pool item by all users) over 7 days, 30 days, all history
INIT_BIAS = -5.1  # logit of the 0.6% positive rate, so training starts past the base rate


def item_stats(history, start, pool_items):
    """Notes §6, level 3, "on the side": popularity of each pool item over all users in several windows."""
    watched = history[history["watched"]]
    windows = [watched[watched["day"] >= start - 7], watched[watched["day"] >= start - 30], watched]
    counts = [window["item"].value_counts().reindex(pool_items, fill_value=0).to_numpy() for window in windows]
    return torch.tensor(np.log1p(np.stack(counts, 1)), dtype=torch.float32)


def histories(history, start, users, n_items):
    """Last MAX_EVENTS events of each user before `start`, oldest first, behind a BOS slot."""
    recent = history[history["user"].isin(users)].groupby("user").tail(MAX_EVENTS)
    row = pd.Series(np.arange(len(users)), index=users)[recent["user"]].to_numpy()
    column = 1 + recent.groupby("user").cumcount().to_numpy()

    item_UL = np.full((len(users), MAX_EVENTS + 1), n_items)
    age_UL = np.zeros((len(users), MAX_EVENTS + 1), dtype=np.int64)
    watched_UL = np.zeros((len(users), MAX_EVENTS + 1), dtype=np.int64)
    item_UL[row, column] = recent["item"].to_numpy()
    age_UL[row, column] = np.log2(1 + start - recent["day"].to_numpy()).astype(np.int64)
    watched_UL[row, column] = recent["watched"].to_numpy()
    length_U = 1 + recent.groupby("user").size().reindex(users, fill_value=0).to_numpy()

    return {
        "item": torch.from_numpy(item_UL),
        "age": torch.from_numpy(age_UL),
        "watched": torch.from_numpy(watched_UL),
        "length": torch.from_numpy(length_U),
    }


class Recommender(nn.Module):
    def __init__(self, n_items, encoder):
        super().__init__()
        self.encoder = ENCODERS[encoder](n_items, DIM, MAX_EVENTS)
        self.item = nn.Embedding(n_items, DIM)
        self.item_bias = nn.Parameter(torch.full((n_items,), INIT_BIAS))
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=INIT_STD)  # logits start near zero

    def forward(self, batch, pool_P, stats_PS=None):
        user_UD = self.encoder(batch)
        return user_UD @ self.item(pool_P).T + self.item_bias[pool_P]


class Cross(nn.Module):
    """The main model of notes §6: the pool item looks at the user's history (level 3), item statistics on the side.

    Per (user u, pool item p, event l) one similarity s_upl = q(p)·k(e_l) + match bits (same item, same genre,
    same country) + a learned bias per age bucket. w = sigmoid(s) is a soft filter over the history.
    Pooling heads: log(1 + sum w), log(1 + sum w·watched), max s, the w-weighted mean of the event vectors.
    Then a gated MLP (W1 x) * sigmoid(W2 x) on [pooled, item vector, item stats] -> logit + item bias.
    """

    def __init__(self, n_items, item_genre, item_country):
        super().__init__()
        from encoders import AGE_BUCKETS, Events

        self.events = Events(n_items, DIM)
        self.item = nn.Embedding(n_items, DIM)
        self.query = nn.Linear(DIM, DIM, bias=False)
        self.key = nn.Linear(DIM, DIM, bias=False)
        self.match = nn.Parameter(torch.zeros(3))
        self.age_bias = nn.Parameter(torch.zeros(AGE_BUCKETS))
        width = 3 + 2 * DIM + STATS
        self.gate_in, self.gate = nn.Linear(width, 2 * DIM), nn.Linear(width, 2 * DIM)
        self.out = nn.Linear(2 * DIM, 1)
        self.item_bias = nn.Parameter(torch.full((n_items,), INIT_BIAS))
        self.register_buffer("genre", torch.cat([item_genre, torch.tensor([-1])]))  # BOS has no genre
        self.register_buffer("country", torch.cat([item_country, torch.tensor([-1])]))
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=INIT_STD)

    def forward(self, batch, pool_P, stats_PS):
        event_ULD = self.events(batch)
        item_PD = self.item(pool_P)
        real_UL = torch.arange(event_ULD.shape[1])[None, :] < batch["length"][:, None]
        real_UL[:, 0] = False  # BOS is not an event

        history_UL = batch["item"]
        same_item_UPL = history_UL[:, None, :] == pool_P[None, :, None]
        same_genre_UPL = self.genre[history_UL][:, None, :] == self.genre[pool_P][None, :, None]
        same_country_UPL = self.country[history_UL][:, None, :] == self.country[pool_P][None, :, None]
        similarity_UPL = (
            torch.einsum("pd,uld->upl", self.query(item_PD), self.key(event_ULD)) / DIM**0.5
            + self.match[0] * same_item_UPL
            + self.match[1] * same_genre_UPL
            + self.match[2] * same_country_UPL
            + self.age_bias[batch["age"]][:, None, :]
        )
        weight_UPL = torch.sigmoid(similarity_UPL) * real_UL[:, None, :]

        count_UP = torch.log1p(weight_UPL.sum(-1))
        watched_UP = torch.log1p((weight_UPL * batch["watched"][:, None, :]).sum(-1))
        top_UP = similarity_UPL.masked_fill(~real_UL[:, None, :], -1e4).amax(-1).clamp(min=-10)
        mean_UPD = torch.einsum("upl,uld->upd", weight_UPL, event_ULD) / (weight_UPL.sum(-1, keepdim=True) + 1e-6)

        users, pool = count_UP.shape
        x_UPF = torch.cat(
            [
                torch.stack([count_UP, watched_UP, top_UP], -1),
                mean_UPD,
                item_PD.expand(users, pool, DIM),
                stats_PS.expand(users, pool, STATS),
            ],
            -1,
        )
        hidden_UPH = self.gate_in(x_UPF) * torch.sigmoid(self.gate(x_UPF))
        return self.out(hidden_UPH).squeeze(-1) + self.item_bias[pool_P]

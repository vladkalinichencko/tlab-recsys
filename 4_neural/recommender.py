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
INIT_BIAS = -5.1  # logit of the 0.6% positive rate, so training starts past the base rate


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

    def pair_state(self, batch, pool_P):
        """the terms of the logit of every (user, pool item) pair: user vector times item vector."""
        user_UD, item_PD = self.encoder(batch), self.item(pool_P)
        users, pool = len(user_UD), len(item_PD)
        return user_UD.reshape(users, 1, DIM) * item_PD.reshape(1, pool, DIM)

    def forward(self, batch, pool_P):
        return self.pair_state(batch, pool_P).sum(-1) + self.item_bias[pool_P]

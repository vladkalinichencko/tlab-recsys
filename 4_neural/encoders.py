"""User encoders. Same contract for all: encoder(batch) -> user_UD.

Axes: U users, L event slots (slot 0 is BOS), D model width, I catalog items.
A batch holds the user's last events before the window: item, age bucket, watched bit, length.
    autoencoder  watched-items vector -> MLP, EASE's input with a nonlinearity, order invisible
    bag          masked mean over event embeddings, order invisible
    lstm         one LSTM layer over events, read at the last event
    transformer  two causal self-attention layers with learned positions, read at the last event;
                 slot t sees slots up to t only, so sequence(batch) predicts the next event (SASRec)
"""
import torch
from torch import nn

AGE_BUCKETS = 9
HEADS = 2
LAYERS = 2
AUTOENCODER_HIDDEN = 600
AUTOENCODER_DROPOUT = 0.5


class Events(nn.Module):
    """One event = item embedding + embedding of log2(days before the window) + watched bit."""

    def __init__(self, n_items, dim):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim)  # row n_items is BOS
        self.age = nn.Embedding(AGE_BUCKETS, dim)
        self.watched = nn.Embedding(2, dim)

    def forward(self, batch):
        return self.item(batch["item"]) + self.age(batch["age"]) + self.watched(batch["watched"])


def real_slots(batch):
    position_L = torch.arange(batch["item"].shape[1], device=batch["item"].device)
    return position_L[None, :] < batch["length"][:, None]


def last_slot(hidden_ULD, batch):
    return hidden_ULD[torch.arange(len(batch["length"])), batch["length"] - 1]


class Autoencoder(nn.Module):
    def __init__(self, n_items, dim, max_events):
        super().__init__()
        self.n_items = n_items
        self.mlp = nn.Sequential(
            nn.Dropout(AUTOENCODER_DROPOUT),
            nn.Linear(n_items, AUTOENCODER_HIDDEN),
            nn.Tanh(),
            nn.Linear(AUTOENCODER_HIDDEN, dim),
        )

    def forward(self, batch):
        users = len(batch["length"])
        watched_UI = torch.zeros(users, self.n_items + 1, device=batch["item"].device)
        watched_UI.scatter_add_(1, batch["item"], batch["watched"].float())
        return self.mlp(watched_UI[:, : self.n_items].clamp(max=1.0))


class Bag(nn.Module):
    def __init__(self, n_items, dim, max_events):
        super().__init__()
        self.events = Events(n_items, dim)

    def forward(self, batch):
        event_ULD = self.events(batch) * real_slots(batch)[..., None]
        return event_ULD.sum(dim=1) / batch["length"][:, None]


class LSTM(nn.Module):
    def __init__(self, n_items, dim, max_events):
        super().__init__()
        self.events = Events(n_items, dim)
        self.lstm = nn.LSTM(dim, dim, batch_first=True)

    def forward(self, batch):
        hidden_ULD, _ = self.lstm(self.events(batch))
        return last_slot(hidden_ULD, batch)


class Transformer(nn.Module):
    def __init__(self, n_items, dim, max_events):
        super().__init__()
        self.events = Events(n_items, dim)
        self.position = nn.Embedding(max_events + 1, dim)
        layer = nn.TransformerEncoderLayer(dim, HEADS, 2 * dim, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)

    def sequence(self, batch):
        """hidden state of every slot, each slot computed from the slots up to it."""
        slots = batch["item"].shape[1]
        position_L = torch.arange(slots, device=batch["item"].device)
        future_LL = torch.ones(slots, slots, dtype=torch.bool, device=batch["item"].device).triu(1)
        return self.transformer(
            self.events(batch) + self.position(position_L), mask=future_LL, is_causal=True, src_key_padding_mask=~real_slots(batch)
        )

    def forward(self, batch):
        return last_slot(self.sequence(batch), batch)


ENCODERS = {"autoencoder": Autoencoder, "bag": Bag, "lstm": LSTM, "transformer": Transformer}

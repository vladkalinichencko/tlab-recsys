"""Linear probes (notes §8): which hand aggregates the hidden state of a pair holds.

PAIRS random pairs of val, their hand features from features.build (the ones boosting `hand` reads),
a ridge regression from the hidden state of each pair to each feature, R² on the half of the pairs
the ridge did not fit. R² near 1: the network computes this aggregate on its own; near 0: it does not
hold it linearly. Counts go through log1p, as the networks see them.
"""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

import features
import stage

PAIRS = 4000
TARGETS = [
    "u_events_all", "u_watched_all", "days_since_last",
    "i_watched_7d", "i_watched_all", "i_rate_all", "item_share",
    "ui_events_all", "ug_watched_all", "uc_watched_all", "genre_share", "country_share", "year_gap",
]


def pairs():
    """rows of the val frame to probe and their hand features, in the same order."""
    start, history, _, _, frame = stage.prepare("val")
    rows = np.sort(np.random.default_rng(0).choice(len(frame), PAIRS, replace=False))
    table = features.build(frame.iloc[rows].reset_index(drop=True), history, start, stage.users, stage.items)
    return rows, table


def r2(hidden_ND, table):
    """R² of a ridge from the hidden state to each target, fit on one half, scored on the other."""
    hidden_ND = (hidden_ND - hidden_ND.mean(0)) / (hidden_ND.std(0) + 1e-6)
    half = len(hidden_ND) // 2
    result = {}
    for name in TARGETS:
        target = table[name].to_numpy(np.float64)
        if "_events_" in name or "_watched_" in name:
            target = np.log1p(target)
        known = ~np.isnan(target)
        if not known.any():  # year_gap on Yambda, which has no release years
            continue
        fit, held = known.copy(), known.copy()
        fit[half:], held[:half] = False, False
        ridge = RidgeCV(alphas=np.logspace(-2, 3, 6)).fit(hidden_ND[fit], target[fit])
        result[name] = float(r2_score(target[held], ridge.predict(hidden_ND[held])))
    return result

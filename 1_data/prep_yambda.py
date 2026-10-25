"""Yambda-50M -> data/*.parquet in the KION schema, one task per run directory.

    cd runs/yambda_listen && PYTHONPATH=../../0_shared python3 ../../1_data/prep_yambda.py listen
    cd runs/yambda_like   && PYTHONPATH=../../0_shared python3 ../../1_data/prep_yambda.py like

Both tasks see the same history: every listen and every like of all 10k users.
The tasks differ only by which event is positive (the `watched` column):
listen  a listen with played_ratio_pct >= 50 (Listen+ of the Yambda paper)
like    a like
The catalog is cut to the CATALOG tracks with most listeners before the test window;
events on other tracks are dropped. Only the USERS users with most events before the test window are kept,
so a whole run fits in an hour. Artist and album go into the KION columns genre and country,
users have no attributes. Artist and album stay integer ids, so the 30M-row history
does not carry Python strings; boosting.pool turns them into categories.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kion import PREP, TEST_DAYS

TASK = sys.argv[1]
RAW = Path(__file__).resolve().parents[1] / "data/yambda"
LISTEN_PCT = 50
CATALOG = 20_000
USERS = 1_000
SECONDS_PER_DAY = 86_400

listens = pd.read_parquet(RAW / "flat/50m/listens.parquet", columns=["uid", "timestamp", "item_id", "played_ratio_pct"])
likes = pd.read_parquet(RAW / "flat/50m/likes.parquet", columns=["uid", "timestamp", "item_id"])
listens["kind"] = "listen"
likes["kind"] = "like"
events = pd.concat([listens, likes], ignore_index=True)
del listens, likes
events["day"] = events["timestamp"] // SECONDS_PER_DAY

test_from = events["day"].max() + 1 - TEST_DAYS
listeners = events[(events["kind"] == "listen") & (events["day"] < test_from)].groupby("item_id")["uid"].nunique()
events = events[events["item_id"].isin(listeners.nlargest(CATALOG).index)]
history_length = events[events["day"] < test_from]["uid"].value_counts()
events = events[events["uid"].isin(history_length.nlargest(USERS).index)]

if TASK == "listen":
    events["watched"] = (events["kind"] == "listen") & (events["played_ratio_pct"] >= LISTEN_PCT)
elif TASK == "like":
    events["watched"] = events["kind"] == "like"
else:
    raise ValueError(TASK)

user_ids = np.sort(events["uid"].unique())
item_ids = np.sort(events["item_id"].unique())
user_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
item_index = pd.Series(np.arange(len(item_ids)), index=item_ids)

events = (
    pd.DataFrame(
        {
            "user": events["uid"].map(user_index).to_numpy(np.int32),
            "item": events["item_id"].map(item_index).to_numpy(np.int32),
            "timestamp": events["timestamp"].to_numpy(),
            "day": events["day"].to_numpy(np.int16),
            "watched_pct": events["played_ratio_pct"].to_numpy(np.float32),  # NaN for likes
            "watched": events["watched"].to_numpy(),
        }
    )
    .sort_values(["user", "timestamp"], ignore_index=True)
    .drop(columns="timestamp")
)

users = pd.DataFrame(
    {
        "user": np.arange(len(user_ids), dtype=np.int32),
        "age": "unknown",
        "income": "unknown",
        "sex": "unknown",
        "kids_flg": np.int8(0),
    }
)

# a track with several artists or albums keeps the first one
artist = pd.read_parquet(RAW / "artist_item_mapping.parquet").drop_duplicates("item_id").set_index("item_id")["artist_id"]
album = pd.read_parquet(RAW / "album_item_mapping.parquet").drop_duplicates("item_id").set_index("item_id")["album_id"]
items = pd.DataFrame(
    {
        "item": np.arange(len(item_ids), dtype=np.int32),
        "content_type": "track",
        "release_year": np.float32(np.nan),
        "age_rating": np.float32(np.nan),
        "genre": artist.reindex(item_ids).fillna(-1).to_numpy(np.int64),
        "country": album.reindex(item_ids).fillna(-1).to_numpy(np.int64),
    }
)

Path(PREP).mkdir(exist_ok=True)
events.to_parquet(f"{PREP}/events.parquet", index=False)
users.to_parquet(f"{PREP}/users.parquet", index=False)
items.to_parquet(f"{PREP}/items.parquet", index=False)

print(f"task {TASK}  events {len(events)}  users {len(users)}  items {len(items)}")
print(f"days 0..{events['day'].max()}  positive rate {events['watched'].mean():.4f}")

"""KION raw csv -> data/*.parquet: a fixed 100k-user sample (random_state=0), day index, watched label.

Missing user attributes become the category "unknown"; missing release year and age rating stay NaN.
"""
import numpy as np
import pandas as pd

from kion import WATCHED_PCT, read_raw
from windows import PREP

N_USERS = 100_000
MIN_ITEM_EVENTS = 5

events, users, items = read_raw()

watch_date = pd.to_datetime(events["last_watch_dt"])
events["day"] = (watch_date - watch_date.min()).dt.days
events["watched_pct"] = events["watched_pct"].fillna(0)
events["watched"] = events["watched_pct"] >= WATCHED_PCT

sample = pd.Series(events["user_id"].unique()).sample(N_USERS, random_state=0)
events = events[events["user_id"].isin(set(sample))]

frequency = events["item_id"].value_counts()
events = events[events["item_id"].isin(frequency[frequency >= MIN_ITEM_EVENTS].index)]

user_ids = np.sort(events["user_id"].unique())
item_ids = np.sort(events["item_id"].unique())
user_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
item_index = pd.Series(np.arange(len(item_ids)), index=item_ids)

events = pd.DataFrame(
    {
        "user": events["user_id"].map(user_index).to_numpy(np.int32),
        "item": events["item_id"].map(item_index).to_numpy(np.int32),
        "day": events["day"].to_numpy(np.int16),
        "watched_pct": events["watched_pct"].to_numpy(np.float32),
        "watched": events["watched"].to_numpy(),
    }
).sort_values(["user", "day"], ignore_index=True)

users = (
    pd.DataFrame({"user": np.arange(len(user_ids), dtype=np.int32), "user_id": user_ids})
    .merge(users, on="user_id", how="left")
    .assign(
        age=lambda frame: frame["age"].fillna("unknown"),
        income=lambda frame: frame["income"].fillna("unknown"),
        sex=lambda frame: frame["sex"].fillna("unknown"),
        kids_flg=lambda frame: frame["kids_flg"].fillna(0).astype(np.int8),
    )[["user", "age", "income", "sex", "kids_flg"]]
)

items = (
    pd.DataFrame({"item": np.arange(len(item_ids), dtype=np.int32), "item_id": item_ids})
    .merge(items, on="item_id", how="left")
    .assign(
        content_type=lambda frame: frame["content_type"].fillna("unknown"),
        release_year=lambda frame: frame["release_year"].astype(np.float32),
        age_rating=lambda frame: frame["age_rating"].astype(np.float32),
        genre=lambda frame: frame["genres"].fillna("unknown").str.split(",").str[0].str.strip(),
        country=lambda frame: frame["countries"].fillna("unknown").str.split(",").str[0].str.strip(),
    )[["item", "content_type", "release_year", "age_rating", "genre", "country"]]
)

events.to_parquet(f"{PREP}/events.parquet", index=False)
users.to_parquet(f"{PREP}/users.parquet", index=False)
items.to_parquet(f"{PREP}/items.parquet", index=False)

print(f"events {len(events)}  users {len(users)}  items {len(items)}")
print(f"days 0..{events['day'].max()}  watched rate {events['watched'].mean():.3f}")

"""KION: the raw csv and the schema of data/*.parquet that prep.py writes.

    USER_COLUMNS, ITEM_COLUMNS   attributes in users.parquet and items.parquet
    CATEGORICAL                  the attributes that are categories, for CatBoost and TabM
    CONTEXT_NUMERIC              the attributes TabICL reads as numbers, the rest as category codes
    GROUPS                       item columns a user's taste is counted over: group -> aggregates of (user, group)
    USER_AGGREGATES              per-user aggregates over item attributes, next to the counts
    GAPS                         item attribute minus the user's aggregate of it
    SHARES                       the user's watched count in the item's group over all the user's watched
    PROBES                       hand features the linear probes read back from a hidden state
"""
import pandas as pd

RAW = "data/raw"
WATCHED_PCT = 10

USER_COLUMNS = ["age", "income", "sex", "kids_flg"]
ITEM_COLUMNS = ["content_type", "release_year", "age_rating"]
CATEGORICAL = ["age", "income", "sex", "content_type", "genre", "country"]
CONTEXT_NUMERIC = ["release_year"]
GROUPS = {
    "genre": {"ug_events": ("day", "size"), "ug_watched": ("watched", "sum"), "ug_pct": ("watched_pct", "mean")},
    "country": {"uc_events": ("day", "size"), "uc_watched": ("watched", "sum")},
}
USER_AGGREGATES = {"u_year": ("release_year", "mean")}
GAPS = {"year_gap": ("release_year", "u_year_all")}
SHARES = {"genre_share": "ug_watched_all", "country_share": "uc_watched_all"}
PROBES = [
    "u_events_all", "u_watched_all", "days_since_last",
    "i_watched_7d", "i_watched_all", "i_rate_all", "item_share",
    "ui_events_all", "ug_watched_all", "uc_watched_all", "genre_share", "country_share", "year_gap",
]


def read_raw():
    return (
        pd.read_csv(f"{RAW}/interactions.csv"),
        pd.read_csv(f"{RAW}/users.csv"),
        pd.read_csv(f"{RAW}/items.csv"),
    )

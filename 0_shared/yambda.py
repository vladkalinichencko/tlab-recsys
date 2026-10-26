"""Yambda: the schema of runs/yambda_*/data/*.parquet that prep_yambda.py writes, the same names as kion.py.

Users have no attributes, a track has its artist and album, both integer ids.
"""
USER_COLUMNS = []
ITEM_COLUMNS = []
CATEGORICAL = ["artist", "album"]
CONTEXT_NUMERIC = []
GROUPS = {
    "artist": {"uartist_events": ("day", "size"), "uartist_watched": ("watched", "sum"), "uartist_pct": ("watched_pct", "mean")},
    "album": {"ualbum_events": ("day", "size"), "ualbum_watched": ("watched", "sum"), "ualbum_pct": ("watched_pct", "mean")},
}
USER_AGGREGATES = {}
GAPS = {}
SHARES = {"artist_share": "uartist_watched_all", "album_share": "ualbum_watched_all"}
PROBES = [
    "u_events_all", "u_watched_all", "days_since_last",
    "i_watched_7d", "i_watched_all", "i_rate_all", "item_share",
    "ui_events_all", "uartist_watched_all", "ualbum_watched_all", "artist_share", "album_share",
]

import pandas as pd

WINDOWS = {"7d": 7, "30d": 30, "all": None}


def recent(history, start, days):
    return history if days is None else history[history["day"] >= start - days]


def blocks(history, start, keys, aggregations):
    windows = [
        recent(history, start, days)
        .groupby(keys)
        .agg(**{f"{name}_{tag}": column for name, column in aggregations.items()})
        for tag, days in WINDOWS.items()
    ]
    return pd.concat(windows, axis=1).reset_index()


def user_stats(history, start):
    return blocks(
        history,
        start,
        ["user"],
        {
            "u_events": ("day", "size"),
            "u_watched": ("watched", "sum"),
            "u_pct": ("watched_pct", "mean"),
            "u_items": ("item", "nunique"),
            "u_year": ("release_year", "mean"),
            "u_last": ("day", "max"),
        },
    )


def item_stats(history, start):
    return blocks(
        history,
        start,
        ["item"],
        {
            "i_events": ("day", "size"),
            "i_watched": ("watched", "sum"),
            "i_rate": ("watched", "mean"),
            "i_users": ("user", "nunique"),
            "i_pct": ("watched_pct", "mean"),
        },
    )


def pair_stats(history, start):
    return blocks(
        history,
        start,
        ["user", "item"],
        {
            "ui_events": ("day", "size"),
            "ui_watched": ("watched", "sum"),
            "ui_pct": ("watched_pct", "max"),
        },
    )


def genre_stats(history, start):
    return blocks(
        history,
        start,
        ["user", "genre"],
        {
            "ug_events": ("day", "size"),
            "ug_watched": ("watched", "sum"),
            "ug_pct": ("watched_pct", "mean"),
        },
    )


def country_stats(history, start):
    return blocks(
        history,
        start,
        ["user", "country"],
        {
            "uc_events": ("day", "size"),
            "uc_watched": ("watched", "sum"),
        },
    )


def build(candidates, history, start, users, items):
    history = history.merge(items, on="item")
    table = (
        candidates.merge(users, on="user", how="left")
        .merge(items, on="item", how="left")
        .merge(user_stats(history, start), on="user", how="left")
        .merge(item_stats(history, start), on="item", how="left")
        .merge(pair_stats(history, start), on=["user", "item"], how="left")
        .merge(genre_stats(history, start), on=["user", "genre"], how="left")
        .merge(country_stats(history, start), on=["user", "country"], how="left")
    )

    # no events in a window means zero counts; means and last day over no events stay NaN
    keys = ["user", "item", "label"]
    numeric = [name for name in table.select_dtypes("number") if name not in keys]
    counts = [
        name for name in numeric if any(f"_{kind}_" in name for kind in ("events", "watched", "items", "users"))
    ]
    table[numeric] = table[numeric].astype("float32")
    table[counts] = table[counts].fillna(0)
    table["days_since_last"] = start - table["u_last_all"]
    table["year_gap"] = table["release_year"] - table["u_year_all"]
    table["genre_share"] = table["ug_watched_all"] / (table["u_watched_all"] + 1)
    table["country_share"] = table["uc_watched_all"] / (table["u_watched_all"] + 1)
    table["item_share"] = table["i_watched_7d"] / (table["i_watched_all"] + 1)
    return table

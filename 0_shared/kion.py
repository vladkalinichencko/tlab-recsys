import pandas as pd

RAW = "data/raw"
PREP = "data"
WATCHED_PCT = 10
FIT_DAYS = 14
VAL_DAYS = 14
TEST_DAYS = 14


def read_raw():
    return (
        pd.read_csv(f"{RAW}/interactions.csv"),
        pd.read_csv(f"{RAW}/users.csv"),
        pd.read_csv(f"{RAW}/items.csv"),
    )


def load():
    return (
        pd.read_parquet(f"{PREP}/events.parquet"),
        pd.read_parquet(f"{PREP}/users.parquet"),
        pd.read_parquet(f"{PREP}/items.parquet"),
    )


def stages(events):
    end = events["day"].max() + 1
    test_from = end - TEST_DAYS
    val_from = test_from - VAL_DAYS
    fit_from = val_from - FIT_DAYS
    return {
        "fit": (fit_from, val_from),
        "val": (val_from, test_from),
        "test": (test_from, end),
    }


def history(events, start):
    return events[events["day"] < start]


def target(events, start, end):
    return events[(events["day"] >= start) & (events["day"] < end) & events["watched"]]

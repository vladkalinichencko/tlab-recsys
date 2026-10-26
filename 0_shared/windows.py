import pandas as pd

PREP = "data"
FIT_DAYS = 14
VAL_DAYS = 14
TEST_DAYS = 14


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

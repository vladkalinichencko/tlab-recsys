"""Feature sets and CatBoost pools shared by train_boosting.py and diag_importance.py.

    all          hand features + EASE and kNN scores
    hand         hand features only
    auto         no hand features: the raw user and item columns + every grammar feature
"""
from catboost import Pool

CATEGORICAL = ["age", "income", "sex", "content_type", "genre", "country"]
CF = ("ease", "knn")
RAW = ["age", "income", "sex", "kids_flg", "content_type", "release_year", "age_rating", "genre", "country"]


def feature_set(name, table):
    everything = [column for column in table.columns if column not in ("user", "item", "label")]
    hand = [column for column in everything if column not in CF]
    if name == "auto":
        return [column for column in everything if column.startswith("g.") or column in RAW]
    if name == "all":
        return everything
    if name == "hand":
        return hand
    raise ValueError(name)


def pool(table, columns):
    frame = table[columns].copy()
    categorical = [column for column in CATEGORICAL if column in columns]
    for column in categorical:
        frame[column] = frame[column].astype(str)
    return Pool(
        frame,
        table["label"].to_numpy(),
        group_id=table["user"].to_numpy(),
        cat_features=categorical,
    )

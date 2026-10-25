"""Figures for report.md from reports/*.json, logs/catboost/*/test_error.tsv and logs/neural_*.log (KION).

    python3 5_report/figures.py   # from recsys/, writes 5_report/figures/*.png
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("5_report/figures")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

NAMES = {
    "popularity": "популярность",
    "ease": "EASE",
    "boosting_yetirank_within_user": "CatBoost, только история юзера (55)",
    "boosting_yetirank_hand": "CatBoost, ручные без CF (71)",
    "boosting_yetirank_all": "CatBoost YetiRank, все (73)",
    "boosting_pointwise_all": "CatBoost Logloss, все (73)",
    "neural_autoencoder": "нелинейный автоэнкодер",
    "neural_bag": "среднее по истории",
    "neural_lstm": "LSTM",
    "neural_transformer": "трансформер",
}
FAMILY = {"popularity": "#9aa0a6", "ease": "#9aa0a6", "boosting": "#d9822b", "neural": "#3b7dd8"}


def color(model):
    return next(c for prefix, c in FAMILY.items() if model.startswith(prefix))


# 1. test metrics, one bar per model
reports = pd.DataFrame([json.load(open(f"reports/{m}.json")) for m in NAMES]).set_index("model")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
order = reports.sort_values("ndcg@10").index
for ax, metric in zip(axes, ["ndcg@10", "recall@10", "roc_auc"]):
    values = reports.loc[order, metric]
    bars = ax.barh([NAMES[m] for m in order], values, color=[color(m) for m in order])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlim(values.min() * 0.85, values.max() * 1.08)
    ax.set_title(metric)
fig.suptitle("KION, test: 15 318 юзеров × 300 кандидатов")
fig.tight_layout()
fig.savefig(OUT / "test_metrics.png", dpi=160)

# 2. val ndcg@10 along training: trees of CatBoost, epochs of the nets
fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for run, style in [
    ("boosting_yetirank_all", "-"),
    ("boosting_yetirank_hand", "-"),
    ("boosting_yetirank_within_user", "-"),
    ("boosting_pointwise_all", "-"),
    ("boosting_yetirank_openfe", "--"),
]:
    curve = pd.read_csv(f"logs/catboost/{run}/test_error.tsv", sep="\t")
    label = NAMES.get(run, "CatBoost, все + 4 из перебора (не дошёл до теста)")
    left.plot(curve["iter"] + 1, curve["NDCG:top=10;type=Exp"], style, label=label, lw=1.4)
left.set_xscale("log")
left.set_xlabel("число деревьев")
left.set_ylabel("val ndcg@10")
left.set_ylim(0.18, 0.35)
left.legend(fontsize=8, loc="lower right")
left.set_title("бустинг")
for encoder in ["autoencoder", "bag", "lstm", "transformer"]:
    epochs = re.findall(r"epoch\s+(\d+)\s+loss ([\d.]+)\s+val ndcg@10 ([\d.]+)", open(f"logs/neural_{encoder}.log").read())
    right.plot([int(e) for e, _, _ in epochs], [float(v) for _, _, v in epochs], marker="o", ms=3, label=NAMES[f"neural_{encoder}"])
right.set_xlabel("эпоха")
right.legend(fontsize=8, loc="lower right")
right.set_title("нейросети над историей")
fig.suptitle("KION, val ndcg@10 по ходу обучения")
fig.tight_layout()
fig.savefig(OUT / "val_curves.png", dpi=160)

# 3. train log loss of the nets: the transformer keeps fitting fit while val drops
fig, ax = plt.subplots(figsize=(6.5, 4))
for encoder in ["autoencoder", "bag", "lstm", "transformer"]:
    epochs = re.findall(r"epoch\s+(\d+)\s+loss ([\d.]+)", open(f"logs/neural_{encoder}.log").read())
    ax.plot([int(e) for e, _ in epochs], [float(l) for _, l in epochs], marker="o", ms=3, label=NAMES[f"neural_{encoder}"])
ax.set_xlabel("эпоха")
ax.set_ylabel("log loss на fit")
ax.legend(fontsize=8)
ax.set_title("KION, лосс обучения нейросетей")
fig.tight_layout()
fig.savefig(OUT / "train_loss.png", dpi=160)

# 4. permutation importance of the boosting without CF, summed by block
BLOCKS = [("счётчики фильма", "i_"), ("счётчики фильма", "item_"), ("история пользователя", "days_since"), ("история пользователя", "genre_share"), ("история пользователя", "country_share"), ("пара пользователь–фильм", "ui_"), ("история пользователя", "u")]
drops = pd.Series(json.load(open("reports/importance.json"))["drops"])


def block(feature):
    return next((name for name, prefix in BLOCKS if feature.startswith(prefix)), "атрибуты")


by_block = drops.clip(lower=0).groupby(drops.index.map(block)).sum().sort_values()
fig, ax = plt.subplots(figsize=(6.5, 3.6))
bars = ax.barh(by_block.index, by_block.values, color="#d9822b")
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
ax.set_xlabel("сумма просадки val ndcg@10 при перемешивании колонки")
ax.set_title("KION, CatBoost без CF: какие блоки признаков работают")
fig.tight_layout()
fig.savefig(OUT / "importance.png", dpi=160)
print(by_block.round(3).to_string())

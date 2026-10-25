"""Figures for report.md from reports/*.json and logs/catboost/*/test_error.tsv (KION).

    python3 5_report/figures.py   # from recsys/, writes 5_report/figures/*.png

A model enters a figure once its report exists, so the script runs before every model has finished.
Every model is compared on the same 200 test users (subset_* of metrics.eval_users), the only test
TabICL is scored on. A net's val curve is drawn up to its best epoch plus PATIENCE epochs, the point
where early stopping ends the run.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("5_report/figures")
OUT.mkdir(exist_ok=True)
PATIENCE = 3
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

NAMES = {
    "popularity": "популярность",
    "ease": "EASE",
    "boosting_yetirank_all": "CatBoost, ручные + CF (73)",
    "boosting_yetirank_hand": "CatBoost, ручные без CF (71)",
    "boosting_yetirank_auto": "CatBoost, автоматически сгенерированный набор (489)",
    "tabm_all": "TabM, ручные + CF",
    "tabm_auto": "TabM, автоматически сгенерированный набор",
    "neural_transformer": "авторегрессионный трансформер",
    "tabicl_softmax": "TabICL, softmax",
    "tabicl_counts": "TabICL, SiLU / n",
    "tabicl_gates": "TabICL, softmax + STG",
}
FAMILY = {"popularity": "#9aa0a6", "ease": "#9aa0a6", "boosting": "#d9822b", "tabm": "#7b4fd8", "neural": "#3b7dd8", "tabicl": "#2a9d5c"}
NAMES = {model: name for model, name in NAMES.items() if Path(f"reports/{model}.json").exists()}
reports = {model: json.load(open(f"reports/{model}.json")) for model in NAMES}


def color(model):
    return next(c for prefix, c in FAMILY.items() if model.startswith(prefix))


# 1. test on the same 200 users, one bar per model
subset = pd.DataFrame({model: {metric: report.get(f"subset_{metric}") for metric in ["ndcg@10", "recall@10", "roc_auc"]} for model, report in reports.items()}).T.dropna(subset=["ndcg@10"])
order = subset.sort_values("ndcg@10").index
fig, axes = plt.subplots(1, 3, figsize=(15, 0.42 * len(order) + 1.6), sharey=True)
for ax, metric in zip(axes, subset.columns):
    values = subset.loc[order, metric].astype(float)
    bars = ax.barh([NAMES[m] for m in order], values, color=[color(m) for m in order])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlim(values.min() * 0.85, values.max() * 1.08)
    ax.set_title(metric)
fig.suptitle("KION, test: одни и те же 200 юзеров × 300 кандидатов")
fig.tight_layout()
fig.savefig(OUT / "test_metrics.png", dpi=160)

# 2. val ndcg@10 along training: trees of CatBoost, epochs of the nets up to early stopping
fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for model, style in [("boosting_yetirank_all", "-"), ("boosting_yetirank_hand", "--"), ("boosting_yetirank_auto", ":")]:
    if model in NAMES:
        curve = pd.read_csv(f"logs/catboost/{model}/test_error.tsv", sep="\t")
        left.plot(curve["iter"] + 1, curve["NDCG:top=10;type=Exp"], style, color=color(model), lw=1.4, label=NAMES[model])
left.set_xscale("log")
left.set_xlabel("число деревьев")
left.set_ylabel("val ndcg@10")
left.legend(fontsize=8, loc="lower right")
left.set_title("бустинг")
for model, report in reports.items():
    curve = report.get("val_ndcg@10_curve")
    if curve:
        first = 0 if model.startswith("tabicl") else 1
        best = curve.index(max(curve))
        shown = curve[: best + PATIENCE + 1]
        right.plot(range(first, first + len(shown)), shown, marker="o", ms=3, color=color(model), label=NAMES[model])
        right.plot(first + best, curve[best], "*", ms=11, color=color(model))
right.set_xlabel("эпоха (0 у TabICL: предобученная модель без дообучения)")
right.legend(fontsize=8, loc="lower right")
right.set_title("нейросети, звезда: эпоха, которая идёт на test")
fig.suptitle("KION, val ndcg@10 по ходу обучения")
fig.tight_layout()
fig.savefig(OUT / "val_curves.png", dpi=160)

# 3. linear probes: R² of RidgeCV from the pair state onto each hand aggregate
probed = {}
for model, report in reports.items():
    if "probe_r2_pretrained" in report:
        probed[f"{NAMES[model]}, до дообучения"] = report["probe_r2_pretrained"]
    if "probe_r2" in report:
        probed[NAMES[model]] = report["probe_r2"]
if probed:
    r2 = pd.DataFrame(probed).T.clip(lower=0)
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(r2) + 2))
    image = ax.imshow(r2.values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(r2.shape[1]), r2.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(r2)), r2.index)
    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            ax.text(j, i, f"{r2.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if r2.iloc[i, j] > 0.6 else "black")
    fig.colorbar(image, ax=ax, label="R² на отложенной половине")
    ax.set_title("KION, линейные пробы: какие ручные агрегаты читаются из состояния пары")
    fig.tight_layout()
    fig.savefig(OUT / "probes.png", dpi=160)
    print(r2.round(2).to_string())

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

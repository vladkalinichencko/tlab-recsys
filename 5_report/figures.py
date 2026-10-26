"""Figures for report.md from <run>/reports/*.json and <run>/logs/catboost/*/test_error.tsv, per dataset.

    python3 5_report/figures.py   # from recsys/, writes 5_report/figures/<run>_*.png

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
DEPTH = 6
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

RUNS = {
    "kion": (Path("."), "KION"),
    "yambda_listen": (Path("runs/yambda_listen"), "Yambda, прослушивание"),
    "yambda_like": (Path("runs/yambda_like"), "Yambda, лайк"),
}
NAMES = {
    "popularity": "популярность",
    "ease": "EASE",
    "boosting_yetirank_all": "CatBoost, ручные + CF",
    "boosting_yetirank_hand": "CatBoost, ручные без CF",
    "boosting_yetirank_auto": "CatBoost, автоматически сгенерированный набор",
    "tabm_all": "TabM, ручные + CF",
    "tabm_auto": "TabM, автоматически сгенерированный набор",
    "neural_transformer": "авторегрессионный трансформер",
    "tabicl_softmax": "TabICL, softmax",
    "tabicl_counts": "TabICL, SiLU / n",
    "tabicl_gates": "TabICL, softmax + STG",
}
TITLES = {"ndcg@10": "NDCG@10", "recall@10": "Recall@10", "roc_auc": "ROC AUC"}
MARKERS = ["o", "s", "o", "s", "^", "o", "^", "o", "o", "s", "^"]
FAMILY = {"popularity": "#9aa0a6", "ease": "#9aa0a6", "boosting": "#d9822b", "tabm": "#7b4fd8", "neural": "#3b7dd8", "tabicl": "#2a9d5c"}


def color(model):
    return next(c for prefix, c in FAMILY.items() if model.startswith(prefix))


def label(model, report):
    return f"{NAMES[model]} ({report['features']})" if "features" in report and not model.startswith("tabicl") else NAMES[model]


def test_metrics(reports, title, out):
    """test on the same 200 users, one bar per model"""
    subset = pd.DataFrame({model: {metric: report.get(f"subset_{metric}") for metric in ["ndcg@10", "recall@10", "roc_auc"]} for model, report in reports.items()}).T.dropna(subset=["ndcg@10"])
    order = subset.sort_values("ndcg@10").index
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.42 * len(order) + 1.6), sharey=True)
    for ax, metric in zip(axes, subset.columns):
        values = subset.loc[order, metric].astype(float)
        bars = ax.barh([label(m, reports[m]) for m in order], values, color=[color(m) for m in order])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xlim(values.min() * 0.85, values.max() * 1.08)
        ax.set_title(TITLES[metric])
    fig.suptitle(f"{title}, test: одни и те же 200 юзеров × 300 кандидатов")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def val_curves(root, reports, title, out):
    """val ndcg@10 along training: trees of CatBoost, epochs of the nets up to early stopping"""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for model, style in [("boosting_yetirank_all", "-"), ("boosting_yetirank_hand", "--"), ("boosting_yetirank_auto", ":")]:
        path = root / f"logs/catboost/{model}/test_error.tsv"
        if model in reports and path.exists():
            curve = pd.read_csv(path, sep="\t")
            left.plot(curve["iter"] + 1, curve["NDCG:top=10;type=Exp"], style, color=color(model), lw=1.4, label=NAMES[model])
    left.set_xscale("log")
    left.set_xlabel("число деревьев")
    left.set_ylabel("NDCG@10 на валидации")
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
    right.set_xlabel("эпоха (0 у TabICL: предобученная модель без дообучения)" if any(m.startswith("tabicl") for m in reports) else "эпоха")
    right.legend(fontsize=8, loc="lower right")
    right.set_title("нейросети, звезда: эпоха, которая идёт на test")
    fig.suptitle(f"{title}, NDCG@10 на валидации по ходу обучения")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def probes(reports, title, out):
    """linear probes: R² of RidgeCV from the pair state onto each hand aggregate"""
    probed = {}
    for model, report in reports.items():
        if "probe_r2_pretrained" in report:
            probed[f"{NAMES[model]}, до дообучения"] = report["probe_r2_pretrained"]
        if "probe_r2" in report:
            probed[NAMES[model]] = report["probe_r2"]
    if not probed:
        return
    r2 = pd.DataFrame(probed).T.clip(lower=0)
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(r2) + 2))
    image = ax.imshow(r2.values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(r2.shape[1]), r2.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(r2)), r2.index)
    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            ax.text(j, i, f"{r2.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if r2.iloc[i, j] > 0.6 else "black")
    fig.colorbar(image, ax=ax, label="R² на отложенной половине")
    ax.set_title(f"{title}, линейные пробы: какие ручные агрегаты читаются из состояния пары")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(r2.round(2).to_string())


def block(feature):
    if feature.startswith(("i_", "item_")):
        return "счётчики товара"
    if feature.startswith("ui_"):
        return "пара пользователь–товар"
    if feature.startswith("u") or feature.startswith("days_since") or feature.endswith("_share"):
        return "история пользователя"
    return "атрибуты"


def importance(root, title, out):
    """permutation importance of the boosting without CF, summed by block"""
    path = root / "reports/importance.json"
    if not path.exists():
        return
    drops = pd.Series(json.load(open(path))["drops"])
    by_block = drops.clip(lower=0).groupby(drops.index.map(block)).sum().sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.barh(by_block.index, by_block.values, color="#d9822b")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("сумма просадки NDCG@10 на валидации при перемешивании колонки")
    ax.set_title(f"{title}, CatBoost без CF: какие блоки признаков работают")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(by_block.round(3).to_string())


def size(model, report, items):
    """learned numbers: a net's parameters, a CatBoost tree's 2^DEPTH leaves and DEPTH thresholds, EASE's item-item matrix"""
    if "parameters" in report:
        return report["parameters"]
    if model.startswith("boosting"):
        return report["trees"] * (2**DEPTH + DEPTH)
    if model == "ease":
        return items**2
    if model == "popularity":
        return items


def sizes(runs, out):
    """test ndcg@10 of the 200 users against the number of learned numbers, one panel per dataset"""
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 5.2))
    handles = {}
    for ax, (title, reports, items) in zip(axes, runs):
        for model, report in reports.items():
            n = size(model, report, items)
            if n is None or "subset_ndcg@10" not in report:
                continue
            handles[model] = ax.scatter(n, report["subset_ndcg@10"], s=55, color=color(model), marker=MARKERS[list(NAMES).index(model)], edgecolor="black", lw=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("число обучаемых чисел")
        ax.set_title(title)
    axes[0].set_ylabel("NDCG@10 на тесте, 200 пользователей")
    order = [model for model in NAMES if model in handles]
    fig.legend([handles[m] for m in order], [NAMES[m] for m in order], loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Размер модели и качество")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


runs = []
for tag, (root, title) in RUNS.items():
    reports = {model: json.load(open(root / f"reports/{model}.json")) for model in NAMES if (root / f"reports/{model}.json").exists()}
    print(f"== {title}: {', '.join(reports)}")
    test_metrics(reports, title, OUT / f"{tag}_test_metrics.png")
    val_curves(root, reports, title, OUT / f"{tag}_val_curves.png")
    probes(reports, title, OUT / f"{tag}_probes.png")
    importance(root, title, OUT / f"{tag}_importance.png")
    runs.append((title, reports, len(pd.read_parquet(root / "data/items.parquet"))))
sizes(runs, OUT / "sizes.png")

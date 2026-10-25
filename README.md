# tlab-recsys

Отчёт лежит в [report.md](report.md), здесь только запуск. Все команды запускай из корня репозитория.

```bash
pip install -r requirements.txt
```

Готовые таблицы KION (`data/*.parquet`) и Yambda (`runs/yambda_listen/data`, `runs/yambda_like/data`) лежат в репозитории, так что исходники качать не нужно. Сравнение моделей на всех трёх наборах запускает один скрипт. На Маке он вызывает `run_mac.sh` с популярностью, EASE и CatBoost, на машине с картой NVIDIA вызывает `run_a100.sh` с TabM, тремя вариантами TabICL и трансформером:

```bash
bash run_comparison.sh
```

Чтобы посчитать только часть наборов, передай их папки, например `bash run_comparison.sh .` для KION или `bash run_comparison.sh runs/yambda_like`. На A100 модели идут в фоне, логи пишутся в `logs/`.

Если хочешь собрать таблицы заново из исходников, KION берётся из [KION_DATASET](https://github.com/irsafilo/KION_DATASET) в `data/raw/`, Yambda берётся с [Hugging Face](https://huggingface.co/datasets/yandex/yambda) в `data/yambda/`. Дальше `run.sh` готовит KION, проверяет утечку и запускает сравнение, а `run_yambda.sh` делает то же для одной задачи Yambda:

```bash
bash run.sh
```

```bash
bash run_yambda.sh listen
```

Графики для отчёта рисуются из `reports/` и `logs/` в `5_report/figures/`:

```bash
python3 5_report/figures.py
```

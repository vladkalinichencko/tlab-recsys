# tlab-recsys

Отчёт лежит в [report.md](report.md), здесь только как всё запустить.

Зависимости ставятся одной командой:

```bash
pip install -r requirements.txt
```

Датасетов в репозитории нет. KION берётся из [KION_DATASET](https://github.com/irsafilo/KION_DATASET), и `interactions.csv`, `users.csv`, `items.csv` кладутся в `data/raw/`. Yambda берётся с [Hugging Face](https://huggingface.co/datasets/yandex/yambda): нужны `flat/50m/listens.parquet` и `flat/50m/likes.parquet` в `data/yambda/`.

Весь KION от сэмпла до итоговых метрик запускает один скрипт. Он готовит данные, проверяет, что нет утечки, и по очереди обучает популярность, EASE, CatBoost и нейросети:

```bash
bash run.sh
```

Yambda идёт двумя задачами, прослушивание и лайк, и у каждой своя папка `runs/yambda_<задача>`:

```bash
bash run_yambda.sh listen
```

Перебор агрегатов с отбором как в OpenFE запускается из папки прогона, после основного скрипта:

```bash
bash run_openfe.sh
```

Графики для отчёта рисует отдельный скрипт, он читает `reports/` и `logs/` и пишет картинки в `5_report/figures/`:

```bash
python3 5_report/figures.py
```

Код разложен по шагам: `0_shared` содержит общее (окна, кандидаты, признаки, метрики), `1_data` готовит данные, `2_baselines` хранит популярность и EASE, `3_boosting` хранит CatBoost и перебор, `4_neural` хранит нейросети, `5_report` хранит отчёт и графики. Все скрипты, кроме перебора, запускаются из корня репозитория.

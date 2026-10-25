# tlab-recsys

Отчёт лежит в [report.md](report.md), здесь только как всё запустить.

Зависимости ставятся одной командой:

```bash
pip install -r requirements.txt
```

Датасетов в репозитории нет. KION берётся из [KION_DATASET](https://github.com/irsafilo/KION_DATASET), и `interactions.csv`, `users.csv`, `items.csv` кладутся в `data/raw/`. Yambda берётся с [Hugging Face](https://huggingface.co/datasets/yandex/yambda): нужны `flat/50m/listens.parquet` и `flat/50m/likes.parquet` в `data/yambda/`.

Весь KION от сэмпла до итоговых метрик запускает один скрипт. Он готовит данные, проверяет, что нет утечки, и запускает `run_comparison.sh`: на Маке популярность, EASE и CatBoost, на машине с картой NVIDIA TabM, три варианта TabICL и авторегрессионный трансформер:

```bash
bash run.sh
```

Yambda идёт двумя задачами, прослушивание и лайк, и у каждой своя папка `runs/yambda_<задача>`:

```bash
bash run_yambda.sh listen
```

Yambda я гонял локально на Маке, поэтому `1_data/prep_yambda.py` режет данные сильнее, чем на KION. Он оставляет 20 тысяч треков с самым большим числом слушателей и тысячу пользователей с самой длинной историей, иначе EASE и прогон целиком не влезают в память и в час. Атрибутов пользователя в Yambda нет, а у трека вместо жанра и страны идут артист и альбом. Метку скрипт берёт как в статье Yambda: прослушивание хотя бы половины трека для `listen` и лайк для `like`. Дальше `run_comparison.sh` обучает в папке задачи те же модели, что и на KION. Готовые таблицы обеих задач лежат в репозитории, так что `bash run_comparison.sh` без аргументов сразу считает KION и обе задачи Yambda.

Графики для отчёта рисует отдельный скрипт, он читает `reports/` и `logs/` и пишет картинки в `5_report/figures/`:

```bash
python3 5_report/figures.py
```

Код разложен по шагам: `0_shared` содержит общее (окна, кандидаты, признаки, метрики), `1_data` готовит данные, `2_baselines` хранит популярность и EASE, `3_boosting` хранит CatBoost, `4_neural` хранит нейросети, `5_report` хранит отчёт и графики. Любой скрипт запускай из корня репозитория.

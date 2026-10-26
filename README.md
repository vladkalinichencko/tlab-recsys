# tlab-recsys

Запуск:

```bash
pip install -r requirements.txt
```

Готовые таблицы KION (`data/*.parquet`) и Yambda (`runs/yambda_listen/data`, `runs/yambda_like/data`) лежат в репозитории. Сравнение моделей на всех трёх наборах запускает один скрипт. На Маке он вызывает `run_mac.sh` с популярностью, EASE и CatBoost, а на NVIDIA вызывает `run_a100.sh` с TabM, тремя вариантами TabICL и трансформером:

```bash
bash run_comparison.sh
```

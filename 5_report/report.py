"""One table of every model's test metrics from reports/*.json."""
import glob
import json

import pandas as pd

reports = [json.load(open(path)) for path in sorted(glob.glob("reports/*.json"))]
rows = [report for report in reports if "model" in report]

print(
    pd.DataFrame(rows)
    .set_index("model")[["roc_auc", "ndcg@10", "recall@10", "map@10", "logloss"]]
    .sort_values("ndcg@10", ascending=False)
    .round(4)
    .to_string()
)

import datetime
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.train import get_models

ROOT = Path(__file__).resolve().parents[1]


def main():
    th = json.loads((ROOT / "models" / "thresholds.json").read_text())
    tr = json.loads((ROOT / "reports" / "test_results.json").read_text())
    win = th["winner"]

    df = pd.read_csv(ROOT / "data" / "processed" / "train.csv").dropna(
        subset=["text", "label"])
    pipe = get_models()[win].fit(df["text"].astype(str).tolist(),
                                 df["label"].astype(int).values)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(pipe, ROOT / "models" / "spam_pipeline.joblib")

    raw = ROOT / "data" / "raw"
    meta = {
        "model_name": win,
        "model_version": "1.0.0",
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
        "sklearn_version": sklearn.__version__,
        "selection_criterion": "highest CV F1 on spam (train only), ties by precision",
        "spam_threshold": th["spam_threshold"],
        "uncertain_low": th["uncertain_low"],
        "spam_min_precision": th["spam_min_precision"],
        "min_precision_reached": th["min_precision_reached"],
        "test_metrics": tr["winner_tuned"],
        "dataset": {
            "files": sorted(p.name for p in raw.iterdir()) if raw.exists() else [],
            "source_url": "unverified, please confirm",
            "license": "unverified, please confirm",
        },
        "trained_on_old_corpus": True,
    }
    (ROOT / "models" / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("saved model + metadata for", win)


if __name__ == "__main__":
    main()
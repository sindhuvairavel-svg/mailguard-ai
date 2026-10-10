import json
import warnings
from pathlib import Path

import joblib
import sklearn

ROOT = Path(__file__).resolve().parents[1]


class Predictor:
    def __init__(self):
        self.meta = json.loads((ROOT / "models" / "metadata.json").read_text())
        if self.meta["sklearn_version"] != sklearn.__version__:
            warnings.warn(
                f"sklearn mismatch: trained {self.meta['sklearn_version']}, "
                f"installed {sklearn.__version__}")
        self.pipe = joblib.load(ROOT / "models" / "spam_pipeline.joblib")

    def predict(self, text):
        p = float(self.pipe.predict_proba([text])[0, 1])
        thr, low = self.meta["spam_threshold"], self.meta["uncertain_low"]
        label = "spam" if p >= thr else ("uncertain" if p >= low else "ham")
        return {"label": label, "spam_probability": p}


_instance = None


def get_predictor():
    global _instance
    if _instance is None:
        _instance = Predictor()
    return _instance


def predict(text):
    return get_predictor().predict(text)
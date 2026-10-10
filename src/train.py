import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

import src.preprocessing as prep

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "train.csv"
OUT = ROOT / "reports" / "model_comparison.json"


def _find_preprocess():
    for name in ("preprocess", "preprocess_text", "clean_text", "preprocess_email"):
        fn = getattr(prep, name, None)
        if callable(fn):
            return fn
    raise ImportError("No preprocessing function found in src/preprocessing.py")


def _clean_many(texts):
    fn = _find_preprocess()
    return [fn(str(t)) for t in texts]


def make_pipe(clf):
    return Pipeline([
        ("prep", FunctionTransformer(_clean_many, validate=False)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, max_features=50000)),
        ("clf", clf),
    ])


def get_models():
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    svm = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)
    sgd = SGDClassifier(loss="modified_huber", class_weight="balanced",
                        random_state=42)
    cnb = ComplementNB()
    models = {
        "MultinomialNB": MultinomialNB(),
        "ComplementNB": cnb,
        "LogisticRegression": lr,
        "LinearSVM_calibrated": svm,
        "SGD": sgd,
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=1),
        "SoftVoting": VotingClassifier(
            [("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
             ("svm", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
             ("sgd", SGDClassifier(loss="modified_huber",
                                   class_weight="balanced", random_state=42)),
             ("cnb", ComplementNB())],
            voting="soft"),
    }
    return {name: make_pipe(clf) for name, clf in models.items()}


def main():
    df = pd.read_csv(TRAIN).dropna(subset=["text", "label"])
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(int).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"precision": "precision", "recall": "recall", "f1": "f1"}

    results = {}
    for name, pipe in get_models().items():
        print("training", name, flush=True)
        r = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
        results[name] = {
            m: {"mean": float(np.mean(r["test_" + m])),
                "std": float(np.std(r["test_" + m]))}
            for m in scoring}
        print(name, {m: round(v["mean"], 4) for m, v in results[name].items()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
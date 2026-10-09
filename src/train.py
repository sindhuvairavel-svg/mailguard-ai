import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from src.preprocessing import preprocess

TRAIN = Path("data/processed/train.csv")
OUT = Path("reports/model_comparison.json")


def tfidf():
    return TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2),
                           min_df=2, sublinear_tf=True)


def get_models():
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    svm = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced"), cv=3)
    sgd = SGDClassifier(loss="modified_huber", class_weight="balanced",
                        random_state=42)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=1)
    models = {
        "MultinomialNB": MultinomialNB(),
        "ComplementNB": ComplementNB(),
        "LogisticRegression": lr,
        "LinearSVM_calibrated": svm,
        "SGD": sgd,
        "RandomForest": rf,
        "SoftVoting": VotingClassifier(
            [("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
             ("cnb", ComplementNB()),
             ("sgd", SGDClassifier(loss="modified_huber",
                                   class_weight="balanced",
                                   random_state=42))],
            voting="soft"),
    }
    return {k: Pipeline([("tfidf", tfidf()), ("clf", v)])
            for k, v in models.items()}


def main():
    df = pd.read_csv(TRAIN)
    X, y = df["text"].astype(str), df["label"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
    }
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
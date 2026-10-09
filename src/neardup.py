from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import preprocess

P = Path("data/processed")
THRESH = 0.95


def near_dup_mask(train_text, other_text):
    vec = TfidfVectorizer()
    X = vec.fit_transform([preprocess(t) for t in train_text])
    Y = vec.transform([preprocess(t) for t in other_text])
    sims = cosine_similarity(Y, X)
    return sims.max(axis=1) >= THRESH


if __name__ == "__main__":
    train = pd.read_csv(P / "train.csv")
    for name in ("val", "test"):
        df = pd.read_csv(P / (name + ".csv"))
        mask = near_dup_mask(train["text"], df["text"])
        print(name, "rows:", len(df), "near-dups removed:", int(mask.sum()))
        df[~mask].to_csv(P / (name + ".csv"), index=False)
        print(name, "after:", int((~mask).sum()))
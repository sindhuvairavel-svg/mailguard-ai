from pathlib import Path
import pandas as pd

RAW = Path("data/raw")


def load_sms(path):
    try:
        df = pd.read_csv(path, sep="\t", header=None,
                         names=["label", "text"], encoding="utf-8",
                         quoting=3, on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        raise ValueError("Empty file: %s" % path)
    if df.empty:
        raise ValueError("No rows in file: %s" % path)
    df["label"] = df["label"].str.strip().str.lower().map({"ham": 0, "spam": 1})
    df["source"] = "sms_uci"
    return df


def load_all():
    frames = []
    for f in sorted(RAW.iterdir()):
        if f.name == "SMSSpamCollection":
            frames.append(load_sms(f))
    if not frames:
        raise ValueError("No supported dataset found in data/raw")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset="text")
    df["label"] = df["label"].astype(int)
    for c in (0, 1):
        if (df["label"] == c).sum() < 10:
            raise ValueError("Need at least 10 rows of each class")
    return df[["text", "label", "source"]].reset_index(drop=True)


if __name__ == "__main__":
    df = load_all()
    print("Total rows:", len(df))
    for s, g in df.groupby("source"):
        print(s, "rows:", len(g),
              "spam ratio:", round(g.label.mean(), 3),
              "avg len:", round(g.text.str.len().mean(), 1))
    print("trained_on_old_corpus = True (SMS only, 2011, English)")
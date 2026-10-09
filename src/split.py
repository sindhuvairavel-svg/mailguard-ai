from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data import load_all

OUT = Path("data/processed")


def make_split(df):
    train, rest = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42)
    val, test = train_test_split(
        rest, test_size=0.50, stratify=rest["label"], random_state=42)
    return train, val, test


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    train, val, test = make_split(load_all())
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(OUT / (name + ".csv"), index=False)
        print(name, len(part), "spam ratio:", round(part.label.mean(), 3))
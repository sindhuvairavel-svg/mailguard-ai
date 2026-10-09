import pandas as pd
import pytest
from src.data import load_sms
from src.split import make_split


def _frame(n=200):
    return pd.DataFrame({
        "text": ["msg %d" % i for i in range(n)],
        "label": [1 if i % 8 == 0 else 0 for i in range(n)],
        "source": "t",
    })


def test_no_overlap():
    train, val, test = make_split(_frame())
    a, b, c = set(train.text), set(val.text), set(test.text)
    assert not (a & b) and not (a & c) and not (b & c)


def test_sizes_and_ratio():
    df = _frame(400)
    train, val, test = make_split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert abs(len(train) / len(df) - 0.70) < 0.02
    base = df.label.mean()
    for part in (train, val, test):
        assert abs(part.label.mean() - base) < 0.02


def test_loader_empty_file(tmp_path):
    p = tmp_path / "empty"
    p.write_text("")
    with pytest.raises(Exception):
        load_sms(p)
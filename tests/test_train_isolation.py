import inspect
import src.train as t


def test_only_train_csv_is_read():
    src = inspect.getsource(t)
    assert "train.csv" in src
    assert "val.csv" not in src
    assert "test.csv" not in src
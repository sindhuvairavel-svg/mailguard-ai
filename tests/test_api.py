import io
import logging

import pytest

from app import create_app

SPAM = "Your parcel is on hold. Pay a delivery fee of $2.99 to release it: http://track-parcel.example/pay"
HAM = "Hi team, the weekly sync is moved to Thursday 3pm. Please update your calendars. Thanks, Priya"
HIGH = {"PREDICT_LIMIT": "1000/minute", "BATCH_LIMIT": "1000/minute"}


@pytest.fixture
def client():
    return create_app(HIGH).test_client()


def csv_file(content):
    return {"file": (io.BytesIO(content), "t.csv")}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.get_json()["model_loaded"] is True


def test_model_info(client):
    j = client.get("/api/model-info").get_json()
    assert j["model_name"] and j["limitations"] and j["metrics"]


def test_predict_spam(client):
    j = client.post("/api/predict", json={"text": SPAM}).get_json()
    assert j["label"] == "spam"
    assert j["top_terms"]["note"] == "model-derived, not proof"
    assert "warnings" in j and "request_id" in j


def test_predict_ham_subject_body(client):
    j = client.post("/api/predict", json={"subject": "Weekly sync", "body": HAM}).get_json()
    assert j["label"] == "ham"


def test_predict_empty_400(client):
    assert client.post("/api/predict", json={"text": "   "}).status_code == 400


def test_predict_too_long_400():
    c = create_app(dict(HIGH, MAX_TEXT_LEN=50)).test_client()
    assert c.post("/api/predict", json={"text": "a" * 51}).status_code == 400


def test_predict_not_json_400(client):
    assert client.post("/api/predict", data="hello").status_code == 400


def test_404_json_no_path_leak(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert "Users" not in r.get_data(as_text=True)


def test_security_headers(client):
    h = client.get("/api/health").headers
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in h


def test_batch_ok(client):
    data = b"text\nhello team meeting at 3pm\nPay delivery fee now http://x.example\n"
    r = client.post("/api/batch-predict", data=csv_file(data),
                    content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200 and j["count"] == 2
    assert j["results"][0]["label"] in ("ham", "uncertain", "spam")


def test_batch_missing_column_400(client):
    r = client.post("/api/batch-predict", data=csv_file(b"body\nhello\n"),
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_batch_bad_csv_400(client):
    r = client.post("/api/batch-predict", data=csv_file(b""),
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_batch_too_many_rows_400():
    c = create_app(dict(HIGH, MAX_BATCH_ROWS=2)).test_client()
    r = c.post("/api/batch-predict", data=csv_file(b"text\na b c\nd e f\ng h i\n"),
               content_type="multipart/form-data")
    assert r.status_code == 400


def test_batch_big_file_413():
    c = create_app(dict(HIGH, MAX_CONTENT_LENGTH=200)).test_client()
    big = b"text\n" + b"hello world\n" * 500
    r = c.post("/api/batch-predict", data=csv_file(big),
               content_type="multipart/form-data")
    assert r.status_code == 413


def test_rate_limit_429():
    c = create_app({"PREDICT_LIMIT": "3/minute"}).test_client()
    codes = [c.post("/api/predict", json={"text": HAM}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429


def test_email_text_not_logged(client, caplog):
    caplog.set_level(logging.DEBUG)
    client.post("/api/predict", json={"text": "SECRETTOKEN98765 please wire the money"})
    assert "SECRETTOKEN98765" not in caplog.text
    assert "length=" in caplog.text


def test_cors_allowed_and_blocked():
    c = create_app(dict(HIGH, ALLOWED_ORIGINS="https://good.example")).test_client()
    ok = c.get("/api/health", headers={"Origin": "https://good.example"})
    bad = c.get("/api/health", headers={"Origin": "https://evil.example"})
    assert ok.headers.get("Access-Control-Allow-Origin") == "https://good.example"
    assert "Access-Control-Allow-Origin" not in bad.headers
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:5000"


def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def show(title, res):
    print("==", title, "->", res[0])
    print(json.dumps(res[1], indent=1)[:700])


show("health", call("GET", "/api/health"))
show("model-info", call("GET", "/api/model-info"))
show("predict spam", call("POST", "/api/predict", {
    "subject": "Parcel",
    "body": "Your parcel is on hold. Pay a delivery fee of $2.99 to release it"}))
show("predict ham", call("POST", "/api/predict", {
    "text": "Hi team, the weekly sync is moved to Thursday 3pm. Thanks, Priya"}))
show("predict empty", call("POST", "/api/predict", {"text": ""}))
codes = [call("POST", "/api/predict", {"text": "meeting moved to 3pm today please confirm"})[0]
         for _ in range(32)]
print("== rate limit codes:", codes)
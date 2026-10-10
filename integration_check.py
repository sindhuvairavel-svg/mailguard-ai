import json
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:5000"
SPAM = [
    "Your bank KYC has expired. Update within 24 hours or your account will be suspended: http://secure-kyc-update.example/login",
    "Your parcel is on hold. Pay a delivery fee of $2.99 to release it: http://track-parcel.example/pay",
    "Security alert: we sent an OTP to your phone. Reply with the OTP now to verify your account.",
    "Congratulations! You won $1,000,000 in our lottery. Send your bank details to claim your prize.",
    "Cheap meds online, no prescription needed. Buy now with 80% discount, limited offer!!!",
]
HAM = [
    "Hi team, the weekly sync is moved to Thursday 3pm. Please update your calendars. Thanks, Priya",
    "IT Helpdesk: as requested, here is the link to reset your password for the company portal. It expires in one hour.",
    "Congratulations on your promotion to Senior Analyst! HR will share the updated contract details next week.",
    "Can you send me the notes from yesterday's lecture? I missed the second half. Thanks!",
    "Your October invoice is attached. Let me know if you have questions about the payment schedule.",
]


def post_json(payload):
    req = urllib.request.Request(
        BASE + "/api/predict", data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def upload(data):
    b = uuid.uuid4().hex
    head = ('--%s\r\nContent-Disposition: form-data; name="file"; '
            'filename="t.csv"\r\nContent-Type: text/csv\r\n\r\n' % b).encode()
    body = head + data + ("\r\n--%s--\r\n" % b).encode()
    req = urllib.request.Request(
        BASE + "/api/batch-predict", data=body, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + b})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


print("TRUTH  LABEL      PROB    TEXT")
for truth, texts in (("spam", SPAM), ("ham", HAM)):
    for t in texts:
        code, j = post_json({"text": t})
        print("%-5s  %-9s  %.3f   %s" % (truth, j.get("label"), j.get("spam_probability", -1), t[:45]))

print("--- error cases (expected 400, 400, 400, 413) ---")
print("empty input   :", post_json({"text": ""})[0])
print("huge input    :", post_json({"text": "a " * 15000})[0])
print("bad csv       :", upload(b"body\nhello\n"))
print("big file      :", upload(b"text\n" + b"hello world\n" * 120000))
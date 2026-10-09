import html
import re
import unicodedata

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MONEY_RE = re.compile(r"[$£€₹]\s?\d[\d,]*(\.\d+)?|\b\d[\d,]*(\.\d+)?\s?(usd|gbp|eur|inr|rs)\b", re.I)
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-\s()]{7,}\d)\b")
NUM_RE = re.compile(r"\b\d+\b")
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def preprocess(text):
    """Shared by training and inference. English-oriented:
    combining marks are stripped, which affects non-Latin scripts."""
    if text is None:
        return ""
    t = str(text)
    t = TAG_RE.sub(" ", t)
    t = html.unescape(t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = URL_RE.sub(" urltoken ", t)
    t = EMAIL_RE.sub(" emailtoken ", t)
    t = MONEY_RE.sub(" moneytoken ", t)
    t = PHONE_RE.sub(" phonetoken ", t)
    t = NUM_RE.sub(" numtoken ", t)
    return SPACE_RE.sub(" ", t).strip()
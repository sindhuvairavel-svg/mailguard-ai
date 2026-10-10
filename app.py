import logging
import os
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, g, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException

from src.explain import NOTE, explain
from src.predictor import get_predictor
from src.reliability import get_warnings

ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mailguard")


def _defaults():
    return {
        "MAX_TEXT_LEN": int(os.getenv("MAX_TEXT_LEN", "20000")),
        "MAX_CONTENT_LENGTH": int(os.getenv("MAX_UPLOAD_BYTES", "1000000")),
        "MAX_BATCH_ROWS": int(os.getenv("MAX_BATCH_ROWS", "500")),
        "PREDICT_LIMIT": os.getenv("PREDICT_LIMIT", "30/minute"),
        "BATCH_LIMIT": os.getenv("BATCH_LIMIT", "5/minute"),
        "ALLOWED_ORIGINS": os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"),
    }


def create_app(overrides=None):
    app = Flask(__name__)
    cfg = _defaults()
    cfg.update(overrides or {})
    app.config.update(cfg)

    origins = [o.strip() for o in app.config["ALLOWED_ORIGINS"].split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": origins}})
    limiter = Limiter(key_func=get_remote_address, app=app, storage_uri="memory://")

    @app.before_request
    def _request_id():
        g.request_id = uuid.uuid4().hex[:8]

    @app.after_request
    def _security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'")
        resp.headers["X-Request-ID"] = getattr(g, "request_id", "-")
        return resp

    def err(msg, code):
        return jsonify(error=msg, request_id=getattr(g, "request_id", "-")), code

    messages = {400: "bad request", 404: "not found", 405: "method not allowed",
                413: "payload too large",
                429: "rate limit exceeded, please wait and retry"}
    for code, msg in messages.items():
        app.register_error_handler(code, lambda e, c=code, m=msg: err(m, c))

    @app.errorhandler(Exception)
    def _unhandled(e):
        if isinstance(e, HTTPException):
            return err(e.name.lower(), e.code)
        log.error("unhandled %s request_id=%s", type(e).__name__,
                  getattr(g, "request_id", "-"))
        return err("internal error", 500)

    @app.route("/")
    def index():
        if (Path(app.static_folder) / "index.html").exists():
            return send_from_directory(app.static_folder, "index.html")
        return jsonify(service="MailGuard AI", status="ok")

    @app.route("/api/health")
    @limiter.exempt
    def health():
        try:
            get_predictor()
            return jsonify(status="ok", model_loaded=True)
        except Exception:
            return jsonify(status="degraded", model_loaded=False), 503

    @app.route("/api/model-info")
    def model_info():
        m = get_predictor().meta
        limits = [
            "Works best on English text; other languages are less reliable.",
            "Very short messages give less reliable results.",
            "Results are a probability from a statistical model, not proof.",
            "Emails you submit are not stored.",
        ]
        if m.get("trained_on_old_corpus"):
            limits.insert(0, "Trained on an older corpus; new phishing styles may be missed.")
        return jsonify(
            model_name=m["model_name"], model_version=m["model_version"],
            trained_at=m["trained_at"], sklearn_version=m["sklearn_version"],
            metrics=m["test_metrics"], dataset=m["dataset"],
            thresholds={"spam": m["spam_threshold"], "uncertain_low": m["uncertain_low"],
                        "spam_min_precision": m["spam_min_precision"],
                        "min_precision_reached": m["min_precision_reached"]},
            limitations=limits)

    @app.route("/api/predict", methods=["POST"])
    @limiter.limit(app.config["PREDICT_LIMIT"])
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return err("send JSON with 'text', or 'subject' and 'body'", 400)
        subject = data.get("subject") or ""
        body = data.get("body") or data.get("text") or ""
        if not isinstance(subject, str) or not isinstance(body, str):
            return err("'subject', 'body' and 'text' must be strings", 400)
        text = (subject + "\n" + body).strip()
        if not text:
            return err("text is empty", 400)
        if len(text) > app.config["MAX_TEXT_LEN"]:
            return err("text too long (max %d characters)" % app.config["MAX_TEXT_LEN"], 400)
        log.info("predict request_id=%s length=%d", g.request_id, len(text))

        pred = get_predictor()
        res = pred.predict(text)
        p = res["spam_probability"]
        warnings = get_warnings(text, p, pred.meta, pred.pipe)
        terms, ewarn = explain(pred.pipe, text)
        if ewarn:
            warnings.append("explanation_unavailable")
        return jsonify(
            label=res["label"], spam_probability=p, confidence=max(p, 1 - p),
            top_terms=({"note": NOTE, "terms": terms} if terms is not None else None),
            warnings=warnings, model_version=pred.meta["model_version"],
            request_id=g.request_id)

    @app.route("/api/batch-predict", methods=["POST"])
    @limiter.limit(app.config["BATCH_LIMIT"])
    def batch_predict():
        f = request.files.get("file")
        if f is None:
            return err("upload a CSV as form field 'file'", 400)
        try:
            df = pd.read_csv(f.stream)
        except Exception:
            return err("could not read CSV", 400)
        if "text" not in df.columns:
            return err("CSV must have a 'text' column", 400)
        if len(df) == 0:
            return err("CSV has no rows", 400)
        if len(df) > app.config["MAX_BATCH_ROWS"]:
            return err("too many rows (max %d)" % app.config["MAX_BATCH_ROWS"], 400)
        log.info("batch request_id=%s rows=%d", g.request_id, len(df))

        pred = get_predictor()
        thr, low = pred.meta["spam_threshold"], pred.meta["uncertain_low"]
        texts = df["text"].fillna("").astype(str).str.slice(0, app.config["MAX_TEXT_LEN"]).tolist()
        idx = [i for i, t in enumerate(texts) if t.strip()]
        probs = {}
        if idx:
            ps = pred.pipe.predict_proba([texts[i] for i in idx])[:, 1]
            probs = {i: float(p) for i, p in zip(idx, ps)}
        results = []
        for i in range(len(texts)):
            if i in probs:
                p = probs[i]
                lab = "spam" if p >= thr else ("uncertain" if p >= low else "ham")
                results.append({"row": i, "label": lab, "spam_probability": p})
            else:
                results.append({"row": i, "label": None, "spam_probability": None})
        return jsonify(count=len(results), results=results, request_id=g.request_id)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
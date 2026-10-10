import json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from src.train import get_models

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed"
CMP = ROOT / "reports" / "model_comparison.json"
MIN_P = float(os.getenv("SPAM_MIN_PRECISION", "0.98"))
UNC_P = 0.90  # looser precision floor = lower edge of uncertain band

SANITY = [
 ("spam", "Your bank KYC has expired. Update within 24 hours or your account will be suspended: http://secure-kyc-update.example/login"),
 ("spam", "Your parcel is on hold. Pay a delivery fee of $2.99 to release it: http://track-parcel.example/pay"),
 ("spam", "Security alert: we sent an OTP to your phone. Reply with the OTP now to verify your account."),
 ("spam", "Congratulations! You won $1,000,000 in our lottery. Send your bank details to claim your prize."),
 ("spam", "Cheap meds online, no prescription needed. Buy now with 80% discount, limited offer!!!"),
 ("ham", "Hi team, the weekly sync is moved to Thursday 3pm. Please update your calendars. Thanks, Priya"),
 ("ham", "IT Helpdesk: as requested, here is the link to reset your password for the company portal. It expires in one hour."),
 ("ham", "Congratulations on your promotion to Senior Analyst! HR will share the updated contract details next week."),
 ("ham", "Can you send me the notes from yesterday's lecture? I missed the second half. Thanks!"),
 ("ham", "Your October invoice is attached. Let me know if you have questions about the payment schedule."),
]


def load(name):
    df = pd.read_csv(D / name).dropna(subset=["text", "label"])
    return df["text"].astype(str).tolist(), df["label"].astype(int).values


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def tune(y, p):
    cand = np.unique(p)

    def prec(t):
        sel = p >= t
        return float((y[sel] == 1).mean()) if sel.any() else 0.0

    ok = [t for t in cand if prec(t) >= MIN_P]
    reached = bool(ok)
    thr = float(min(ok)) if reached else 0.5
    ok2 = [t for t in cand if t <= thr and prec(t) >= UNC_P]
    low = float(min(ok2)) if ok2 else thr
    return thr, low, reached


def main():
    res = json.loads(CMP.read_text())
    ranked = sorted(res, key=lambda k: (round(res[k]["f1"]["mean"], 6),
                                        res[k]["precision"]["mean"]), reverse=True)
    win, runner = ranked[0], ranked[1]
    print("winner:", win, "runner-up:", runner)
    for k in (win, runner):
        print(k, {m: (round(v["mean"], 4), round(v["std"], 4)) for m, v in res[k].items()})

    Xtr, ytr = load("train.csv")
    Xva, yva = load("val.csv")
    fitted = {}
    for n, pipe in get_models().items():
        print("fit", n, flush=True)
        fitted[n] = pipe.fit(Xtr, ytr)

    thr, low, reached = tune(yva, fitted[win].predict_proba(Xva)[:, 1])
    print(f"VAL threshold={thr:.4f} uncertain_low={low:.4f} "
          f"min_precision_reached={reached}")

    Xte, yte = load("test.csv")  # test used once, after selection + tuning
    rows = {n: metrics(yte, (m.predict_proba(Xte)[:, 1] >= 0.5).astype(int))
            for n, m in fitted.items()}
    final = metrics(yte, (fitted[win].predict_proba(Xte)[:, 1] >= thr).astype(int))

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "thresholds.json").write_text(json.dumps(
        {"winner": win, "spam_threshold": thr, "uncertain_low": low,
         "spam_min_precision": MIN_P, "min_precision_reached": reached}, indent=2))
    (ROOT / "reports" / "test_results.json").write_text(
        json.dumps({"all_models_at_0.5": rows, "winner_tuned": final}, indent=2))

    md = ["# Model comparison (TEST set, evaluated once)",
          "Test set did not influence model selection or thresholds.", "",
          "| Model | Acc | Prec | Rec | F1 | FP | FN |", "|---|---|---|---|---|---|---|"]
    for n, r in rows.items():
        md.append(f"| {n} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
                  f"{r['recall']:.4f} | {r['f1']:.4f} | {r['fp']} | {r['fn']} |")
    md += ["", f"**Winner {win}** with tuned threshold {thr:.4f}: {final}"]
    (ROOT / "reports" / "model_comparison.md").write_text("\n".join(md))
    print("TEST all models @0.5:")
    for n, r in rows.items():
        print(n, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    print("TEST winner tuned:", final)

    print("SANITY (not used for tuning):")
    for truth, text in SANITY:
        p = float(fitted[win].predict_proba([text])[0, 1])
        lab = "spam" if p >= thr else ("uncertain" if p >= low else "ham")
        print(f"{truth:5} -> {lab:9} p={p:.3f} | {text[:55]}")


if __name__ == "__main__":
    main()
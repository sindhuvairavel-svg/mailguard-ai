import re


def get_warnings(text, probability, meta, pipe=None):
    w = []
    thr = meta["spam_threshold"]
    if probability is not None and abs(probability - thr) <= 0.10:
        w.append("near_decision_boundary")
    if len(text.split()) < 5:
        w.append("short_input")
    letters = [c for c in text if c.isalpha()]
    if letters and sum(ord(c) > 127 for c in letters) / len(letters) > 0.30:
        w.append("non_english_suspected")
    if pipe is not None:
        try:
            prepped = pipe.named_steps["prep"].transform([text])[0]
            vocab = pipe.named_steps["tfidf"].vocabulary_
            toks = re.findall(r"\w+", prepped)
            if toks and sum(t in vocab for t in toks) / len(toks) < 0.5:
                w.append("low_vocab_coverage")
        except Exception:
            pass
    if meta.get("trained_on_old_corpus"):
        w.append("trained_on_old_corpus")
    return w
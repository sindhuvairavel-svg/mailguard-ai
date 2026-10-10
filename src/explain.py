import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB, MultinomialNB

NOTE = "model-derived, not proof"


def _linear_coef(clf):
    if isinstance(clf, CalibratedClassifierCV):
        cs = []
        for cc in clf.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est is None:
                est = getattr(cc, "base_estimator", None)
            if hasattr(est, "coef_"):
                cs.append(np.ravel(est.coef_))
        return np.mean(cs, axis=0) if cs else None
    if hasattr(clf, "coef_"):
        return np.ravel(clf.coef_)
    return None


def explain(pipe, text, k=8):
    """Returns (terms or None, warning or None). Never invents terms."""
    try:
        prep = pipe.named_steps["prep"]
        tfidf = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["clf"]
        X = tfidf.transform(prep.transform([text])).tocsr()
        if X.nnz == 0:
            return None, "no known terms in input"
        idx = X.indices
        vals = X.data
        names = tfidf.get_feature_names_out()

        coef = _linear_coef(clf)
        if coef is not None:
            contrib = coef[idx] * vals
        elif isinstance(clf, (MultinomialNB, ComplementNB)):
            diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
            contrib = diff[idx]
        else:  # occlusion on top-k tfidf terms
            full = float(clf.predict_proba(X)[0, 1])
            top = np.argsort(-vals)[:k]
            contrib = np.zeros(len(idx))
            for t in top:
                Xo = X.copy()
                Xo.data[t] = 0.0
                contrib[t] = full - float(clf.predict_proba(Xo)[0, 1])

        order = np.argsort(-np.abs(contrib))[:k]
        terms = [{"term": str(names[idx[i]]),
                  "contribution": round(float(contrib[i]), 4),
                  "direction": "spam" if contrib[i] > 0 else "ham"}
                 for i in order if contrib[i] != 0]
        return terms, None
    except Exception as e:
        return None, f"explanation unavailable ({type(e).__name__})"
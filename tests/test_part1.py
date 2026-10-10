from src.explain import explain
from src.predictor import get_predictor
from src.reliability import get_warnings

META = {"spam_threshold": 0.5, "trained_on_old_corpus": True}


def test_short_input():
    assert "short_input" in get_warnings("hi", 0.1, META)


def test_near_boundary():
    assert "near_decision_boundary" in get_warnings("a b c d e f g", 0.52, META)
    assert "near_decision_boundary" not in get_warnings("a b c d e f g", 0.02, META)


def test_non_english():
    assert "non_english_suspected" in get_warnings("வணக்கம் நண்பா எப்படி இருக்கிறீர்கள் இன்று", 0.1, META)


def test_old_corpus_flag():
    assert "trained_on_old_corpus" in get_warnings("a b c d e f g", 0.1, META)


def test_explain_terms_real():
    pipe = get_predictor().pipe
    terms, warn = explain(pipe, "Pay a delivery fee to release your parcel now")
    assert warn is None and terms
    assert all({"term", "contribution", "direction"} <= set(t) for t in terms)


def test_explain_unknown_input_is_none():
    terms, warn = explain(get_predictor().pipe, "zzzqqq xxyyzz")
    assert terms is None and warn
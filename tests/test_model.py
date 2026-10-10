from src.predictor import get_predictor, predict


def test_model_loads():
    assert get_predictor().meta["model_name"]


def test_spam_is_spam():
    assert predict("Your parcel is on hold. Pay a delivery fee of $2.99 to release it: http://track-parcel.example/pay")["label"] == "spam"
    assert predict("Security alert: we sent an OTP to your phone. Reply with the OTP now to verify your account.")["label"] == "spam"


def test_ham_is_ham():
    assert predict("Hi team, the weekly sync is moved to Thursday 3pm. Please update your calendars. Thanks, Priya")["label"] == "ham"
    assert predict("Can you send me the notes from yesterday's lecture? I missed the second half. Thanks!")["label"] == "ham"
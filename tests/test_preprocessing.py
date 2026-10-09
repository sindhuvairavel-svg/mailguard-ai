from src.preprocessing import preprocess


def test_html_removed():
    assert "<b>" not in preprocess("<b>Hello</b>")


def test_entities_unescaped():
    assert preprocess("Tom &amp; Jerry") == "tom & jerry"


def test_lowercase():
    assert preprocess("HELLO") == "hello"


def test_url_token():
    assert "urltoken" in preprocess("visit http://bad.com/x now")


def test_email_token():
    assert "emailtoken" in preprocess("mail me a.b@test.com")


def test_money_token():
    assert "moneytoken" in preprocess("win $1,000 now")


def test_phone_token():
    assert "phonetoken" in preprocess("call +91 98765 43210")


def test_number_token():
    assert "numtoken" in preprocess("you have 5 messages")


def test_none_safe():
    assert preprocess(None) == ""
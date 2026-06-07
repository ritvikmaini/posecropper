from posecropper.metadata import classify_category, parse_filename


def test_classify_top():
    assert classify_category("T-Shirts") == "top"
    assert classify_category("Hoodies") == "top"


def test_classify_bottom():
    assert classify_category("Jeans") == "bottom"
    assert classify_category("Roecke") == "bottom"


def test_classify_review_accessory():
    assert classify_category("Caps") == "review"
    assert classify_category("Sneakers") == "review"


def test_classify_unknown_falls_back_to_review():
    assert classify_category("SomethingNotInAnyList") == "review"
    assert classify_category(None) == "review"


def test_parse_filename():
    d = parse_filename("ABC123_M12-front.jpg")
    assert d["model"] == "ABC123"
    assert d["pose"] == "M12"
    assert d["reading_error"] is False


def test_parse_filename_no_match():
    d = parse_filename("badname.jpg")
    assert d["reading_error"] is True
    assert d["pose"] is None

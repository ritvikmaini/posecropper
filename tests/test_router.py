from posecropper.router import decide


def test_branch1_full_body_pose():
    d = decide("M12", "top")          # pose wins over category
    assert d.strategies == ["full"]
    assert d.review is False


def test_branch2_default_pose():
    d = decide("M20", "bottom")       # M17..M31 -> default
    assert d.strategies == ["default"]
    assert d.review is False


def test_branch3_product_pose():
    d = decide("P3", "top")           # P* -> default
    assert d.strategies == ["default"]
    assert d.review is False


def test_branch4_category_top():
    d = decide("M99", "top")          # unknown pose -> category decides
    assert d.strategies == ["top"]
    assert d.review is False


def test_branch5_category_bottom():
    d = decide("M99", "bottom")
    assert d.strategies == ["bottom"]
    assert d.review is False


def test_branch6_ambiguous_review():
    d = decide("M99", "review")       # unknown pose + non-top/bottom -> both crops, review
    assert d.strategies == ["top", "bottom"]
    assert d.review is True


def test_branch6_none_inputs_review():
    d = decide(None, None)
    assert d.strategies == ["top", "bottom"]
    assert d.review is True

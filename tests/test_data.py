from api.main import CLASS_NAMES


def test_class_names_count():
    assert len(CLASS_NAMES) == 6


def test_class_names_content():
    expected = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
    assert set(CLASS_NAMES) == expected

import shutil

from PIL import Image

# ── _file_hash ────────────────────────────────────────────────────────────────


def test_file_hash_deterministic(tmp_path):
    from garbage_classification.dataset import _file_hash

    f = tmp_path / "img.jpg"
    Image.new("RGB", (10, 10)).save(f)
    assert _file_hash(f) == _file_hash(f)


def test_file_hash_differs_for_different_files(tmp_path):
    from garbage_classification.dataset import _file_hash

    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(a)
    Image.new("RGB", (10, 10), color=(0, 255, 0)).save(b)
    assert _file_hash(a) != _file_hash(b)


# ── prepare() ─────────────────────────────────────────────────────────────────


def test_prepare_keeps_unique_image(tmp_path, monkeypatch):
    """A single unique image must end up in the cleaned dir."""
    import garbage_classification.dataset as ds

    raw = tmp_path / "raw" / "cardboard"
    raw.mkdir(parents=True)
    Image.new("RGB", (10, 10), color=(1, 2, 3)).save(raw / "img1.jpg")

    cleaned = tmp_path / "cleaned"
    monkeypatch.setattr(ds, "RAW_IMAGES_DIR", tmp_path / "raw")
    monkeypatch.setattr(ds, "CLEANED_DIR", cleaned)

    ds.prepare()

    files = list((cleaned / "cardboard").iterdir())
    assert len(files) == 1
    assert files[0].name == "img1.jpg"


def test_prepare_removes_within_class_duplicates(tmp_path, monkeypatch):
    """Two identical images in the same class → only one survives."""
    import garbage_classification.dataset as ds

    raw = tmp_path / "raw" / "cardboard"
    raw.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(raw / "img1.jpg")
    shutil.copy(raw / "img1.jpg", raw / "img2.jpg")  # exact byte-for-byte duplicate

    cleaned = tmp_path / "cleaned"
    monkeypatch.setattr(ds, "RAW_IMAGES_DIR", tmp_path / "raw")
    monkeypatch.setattr(ds, "CLEANED_DIR", cleaned)

    ds.prepare()

    assert len(list((cleaned / "cardboard").iterdir())) == 1


def test_prepare_removes_cross_class_duplicates(tmp_path, monkeypatch):
    """Same image in two different classes → both are dropped (ambiguous label)."""
    import garbage_classification.dataset as ds

    img = Image.new("RGB", (10, 10), color=(42, 42, 42))
    for cls in ("cardboard", "glass"):
        d = tmp_path / "raw" / cls
        d.mkdir(parents=True)
        img.save(d / "same.jpg")

    cleaned = tmp_path / "cleaned"
    monkeypatch.setattr(ds, "RAW_IMAGES_DIR", tmp_path / "raw")
    monkeypatch.setattr(ds, "CLEANED_DIR", cleaned)

    ds.prepare()

    for cls in ("cardboard", "glass"):
        cls_dir = cleaned / cls
        count = len(list(cls_dir.iterdir())) if cls_dir.exists() else 0
        assert count == 0, f"Expected 0 files in {cls}, got {count}"


def test_prepare_multiple_classes_independent(tmp_path, monkeypatch):
    """Unique images in different classes are all kept independently."""
    import garbage_classification.dataset as ds

    for cls, color in [("cardboard", (10, 20, 30)), ("glass", (40, 50, 60))]:
        d = tmp_path / "raw" / cls
        d.mkdir(parents=True)
        Image.new("RGB", (10, 10), color=color).save(d / "img.jpg")

    cleaned = tmp_path / "cleaned"
    monkeypatch.setattr(ds, "RAW_IMAGES_DIR", tmp_path / "raw")
    monkeypatch.setattr(ds, "CLEANED_DIR", cleaned)

    ds.prepare()

    assert len(list((cleaned / "cardboard").iterdir())) == 1
    assert len(list((cleaned / "glass").iterdir())) == 1


# ── split() ───────────────────────────────────────────────────────────────────


def _make_split_env(tmp_path):
    """Helper: build a minimal cleaned dir + split txt files for split() tests."""
    # One cardboard image (label 3) and one glass image (label 1)
    for cls, color in [("cardboard", (10, 20, 30)), ("glass", (40, 50, 60))]:
        d = tmp_path / "cleaned" / cls
        d.mkdir(parents=True)
        Image.new("RGB", (10, 10), color=color).save(d / "img001.jpg")

    for split_name in ("train", "val", "test"):
        txt = tmp_path / f"one-indexed-files-notrash_{split_name}.txt"
        txt.write_text("img001.jpg 3\nimg001.jpg 1\n")  # cardboard + glass

    return tmp_path


def test_split_creates_train_val_test_dirs(tmp_path, monkeypatch):
    """split() must create train/, val/, test/ sub-directories."""
    import garbage_classification.dataset as ds

    _make_split_env(tmp_path)
    processed = tmp_path / "processed"

    monkeypatch.setattr(ds, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(ds, "CLEANED_DIR", tmp_path / "cleaned")
    monkeypatch.setattr(ds, "PROCESSED_DATA_DIR", processed)

    ds.split()

    for split_name in ("train", "val", "test"):
        assert (processed / split_name).is_dir()


def test_split_copies_to_correct_class_dirs(tmp_path, monkeypatch):
    """Files end up in processed/{split}/{class}/ with the right filename."""
    import garbage_classification.dataset as ds

    _make_split_env(tmp_path)
    processed = tmp_path / "processed"

    monkeypatch.setattr(ds, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(ds, "CLEANED_DIR", tmp_path / "cleaned")
    monkeypatch.setattr(ds, "PROCESSED_DATA_DIR", processed)

    ds.split()

    assert (processed / "train" / "cardboard" / "img001.jpg").exists()
    assert (processed / "train" / "glass" / "img001.jpg").exists()


def test_split_handles_missing_source_gracefully(tmp_path, monkeypatch):
    """split() should not crash when a file listed in the txt is absent from cleaned/."""
    import garbage_classification.dataset as ds

    # Split txt references a file that doesn't exist in cleaned/
    (tmp_path / "cleaned" / "cardboard").mkdir(parents=True)
    for split_name in ("train", "val", "test"):
        txt = tmp_path / f"one-indexed-files-notrash_{split_name}.txt"
        txt.write_text("ghost.jpg 3\n")

    processed = tmp_path / "processed"
    monkeypatch.setattr(ds, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(ds, "CLEANED_DIR", tmp_path / "cleaned")
    monkeypatch.setattr(ds, "PROCESSED_DATA_DIR", processed)

    ds.split()  # must not raise

    # Nothing copied — processed dir may not even be created
    assert not (processed / "train" / "cardboard" / "ghost.jpg").exists()

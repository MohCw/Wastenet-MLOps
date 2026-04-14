from pathlib import Path

from garbage_classification.config import (
    DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
)


def test_proj_root_is_path():
    assert isinstance(PROJ_ROOT, Path)


def test_all_dirs_are_paths():
    dirs = [
        DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, METRICS_DIR,
    ]
    for d in dirs:
        assert isinstance(d, Path)


def test_dirs_are_under_proj_root():
    dirs = [
        DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, METRICS_DIR,
    ]
    for d in dirs:
        assert PROJ_ROOT in d.parents

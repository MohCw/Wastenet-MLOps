import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from garbage_classification.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

RAW_IMAGES_DIR = RAW_DATA_DIR / "Garbage classification" / "Garbage classification"
CLEANED_DIR = INTERIM_DATA_DIR / "cleaned"

# 1-indexed label → class directory name (matches raw data folder names)
LABEL_MAP = {1: "glass", 2: "paper", 3: "cardboard", 4: "plastic", 5: "metal", 6: "trash"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _file_hash(path: Path) -> str:
    """MD5 hash of file content for exact-duplicate detection."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _iter_images(directory: Path):
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


@app.command()
def prepare():
    """Copy raw images to data/interim/cleaned/, removing duplicates.

    Two-pass strategy:
      - Within-class duplicates  → keep first occurrence, skip the rest.
      - Cross-class duplicates   → remove ALL occurrences (label is ambiguous).
    """
    logger.info(f"Reading raw images from: {RAW_IMAGES_DIR}")

    # ── Pass 1 : collect hash → [(class_name, Path), ...] ──────────────────
    hash_to_files: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    class_dirs = sorted(d for d in RAW_IMAGES_DIR.iterdir() if d.is_dir())
    for class_dir in tqdm(class_dirs, desc="hashing", unit="class"):
        for img in _iter_images(class_dir):
            h = _file_hash(img)
            hash_to_files[h].append((class_dir.name, img))

    # ── Pass 2 : copy clean images ──────────────────────────────────────────
    copied = within_skipped = cross_skipped = 0

    for h, entries in hash_to_files.items():
        classes_seen = {cls for cls, _ in entries}

        if len(classes_seen) > 1:
            # Cross-class duplicate → ambiguous label, drop all occurrences
            names = [(cls, p.name) for cls, p in entries]
            logger.warning(f"Cross-class duplicate [{h[:8]}…] removed: {names}")
            cross_skipped += len(entries)
            continue

        # Same class → keep first occurrence only
        class_name, img = entries[0]
        out_dir = CLEANED_DIR / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out_dir / img.name)
        copied += 1

        if len(entries) > 1:
            within_skipped += len(entries) - 1
            for _, dup in entries[1:]:
                logger.debug(f"Within-class duplicate removed: {class_name}/{dup.name}")

    logger.success(
        f"Preparation complete: {copied} images kept | "
        f"{within_skipped} within-class duplicates removed | "
        f"{cross_skipped} cross-class duplicates removed"
    )
    logger.info(f"Output: {CLEANED_DIR}")


@app.command()
def split():
    """Organize cleaned images into data/processed/{train,val,test}/{class}/ using predefined splits."""
    splits = {
        "train": RAW_DATA_DIR / "one-indexed-files-notrash_train.txt",
        "val":   RAW_DATA_DIR / "one-indexed-files-notrash_val.txt",
        "test":  RAW_DATA_DIR / "one-indexed-files-notrash_test.txt",
    }

    for split_name, txt_path in splits.items():
        logger.info(f"Processing '{split_name}' split from {txt_path.name}")
        lines = [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]
        copied, missing = 0, 0

        for line in tqdm(lines, desc=split_name, unit="img"):
            parts = line.split()
            if len(parts) < 2:
                continue
            filename, label = parts[0], int(parts[1])
            class_name = LABEL_MAP[label]

            src = CLEANED_DIR / class_name / filename
            if not src.exists():
                logger.warning(f"Missing in cleaned dir: {src}")
                missing += 1
                continue

            dst = PROCESSED_DATA_DIR / split_name / class_name / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

        logger.success(f"  '{split_name}': {copied} images copied, {missing} missing")

    logger.success(f"Split complete → {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    app()

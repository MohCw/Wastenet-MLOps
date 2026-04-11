"""
Pre-compute reference embeddings and image properties from the training set.

Run this once after training to build the reference distribution used by run_drift.py.
Re-run whenever the champion model changes.

Usage:
    poetry run python -m monitoring.build_reference

Output:
    monitoring/reference.parquet  —  one row per training image with columns:
        predicted_class, confidence, brightness, blur_score, r_mean, g_mean, b_mean,
        emb_0 .. emb_N  (ConvNeXt Tiny penultimate-layer features, N=512)
"""
import os
from pathlib import Path

import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from PIL import Image
from timm.data import resolve_data_config, create_transform
from loguru import logger

from garbage_classification.config import PROCESSED_DATA_DIR, PROJ_ROOT

REFERENCE_PATH = PROJ_ROOT / "monitoring" / "reference.parquet"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def _load_champion_model() -> tuple[torch.nn.Module, object, torch.device]:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: torch.nn.Module = mlflow.pytorch.load_model(
        "models:/garbage-classifier@champion", map_location=device
    )
    model = model.to(device).eval()
    data_cfg = resolve_data_config(model.pretrained_cfg, model=model)  # type: ignore[attr-defined]
    transform = create_transform(**data_cfg, is_training=False)
    logger.info(f"Champion model loaded on {device}")
    return model, transform, device


def _process_image(
    img_path: Path,
    model: torch.nn.Module,
    transform,
    device: torch.device,
) -> dict:
    """Extract embedding, class scores, and image properties for one image."""
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Penultimate-layer embedding via timm forward_features
        feats = model.forward_features(tensor)
        if feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])  # global average pool -> (1, C)
        emb: np.ndarray = feats.squeeze().cpu().numpy()

        # Classification probabilities
        probs: np.ndarray = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    # Low-level visual statistics for image property drift
    arr = np.array(image, dtype=np.float32) / 255.0
    gray = np.array(image.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(gray)

    row: dict = {
        "predicted_class": CLASS_NAMES[int(probs.argmax())],
        "confidence": float(probs.max()),
        "brightness": float(arr.mean()),
        "blur_score": float(np.var(gx) + np.var(gy)),
        "r_mean": float(arr[:, :, 0].mean()),
        "g_mean": float(arr[:, :, 1].mean()),
        "b_mean": float(arr[:, :, 2].mean()),
    }
    for i, v in enumerate(emb):
        row[f"emb_{i}"] = float(v)

    return row


def build() -> None:
    model, transform, device = _load_champion_model()

    train_dir = PROCESSED_DATA_DIR / "train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            "Run the DVC pipeline first: dvc repro"
        )

    image_paths = [
        p
        for class_dir in sorted(train_dir.iterdir())
        if class_dir.is_dir()
        for p in class_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    logger.info(f"Processing {len(image_paths)} training images...")

    rows = []
    for idx, img_path in enumerate(image_paths, 1):
        rows.append(_process_image(img_path, model, transform, device))
        if idx % 100 == 0:
            logger.info(f"  {idx}/{len(image_paths)}")

    df = pd.DataFrame(rows)
    REFERENCE_PATH.parent.mkdir(exist_ok=True)
    df.to_parquet(REFERENCE_PATH, index=False)

    n_emb_cols = len([c for c in df.columns if c.startswith("emb_")])
    logger.success(
        f"Reference saved: {len(df)} images, {n_emb_cols} embedding dims -> {REFERENCE_PATH}"
    )


if __name__ == "__main__":
    build()

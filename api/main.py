from contextlib import asynccontextmanager
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow.pytorch
import timm
from timm.data import resolve_data_config, create_transform
import torch
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger

from garbage_classification.config import PROJ_ROOT

# Class names in alphabetical order — matches ImageFolder ordering used during training
# (cardboard=0, glass=1, metal=2, paper=3, plastic=4, trash=5)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

LOGS_DIR = PROJ_ROOT / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

_model: torch.nn.Module | None = None
_transform = None
_device: torch.device | None = None


def _load_model() -> tuple[torch.nn.Module, object, torch.device]:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    model_uri = "models:/garbage-classifier@champion"
    logger.info(f"Loading model from registry: {model_uri}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: torch.nn.Module = mlflow.pytorch.load_model(model_uri, map_location=device)
    model = model.to(device).eval()

    data_cfg = resolve_data_config(model.pretrained_cfg, model=model)  # type: ignore[attr-defined]
    transform = create_transform(**data_cfg, is_training=False)

    logger.info(f"Model loaded on {device}")
    return model, transform, device


def _log_prediction(filename: str, predicted_class: str, confidence: float,
                    scores: dict, inference_ms: float) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "scores": scores,
        "inference_ms": round(inference_ms, 2),
    }
    with PREDICTIONS_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _transform, _device
    _model, _transform, _device = _load_model()
    yield


app = FastAPI(
    title="WasteNet API",
    description=(
        "Garbage image classification API powered by ConvNeXt.\n\n"
        "Upload a photo of waste and get its category: "
        "**cardboard**, **glass**, **metal**, **paper**, **plastic**, **trash**."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", summary="Classify a waste image")
async def predict(file: UploadFile = File(..., description="Image file (jpg, png, webp, ...)")):
    """
    Upload a waste image and receive the predicted category with confidence scores.
    Accepts any image format supported by Pillow (jpg, png, webp, bmp, tiff, ...).

    - **predicted_class**: most likely waste category
    - **confidence**: softmax probability for the top class (0–1)
    - **scores**: softmax probability for each of the 6 waste classes
    - **inference_ms**: model inference time in milliseconds
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, webp, ...)")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    t0 = time.perf_counter()
    tensor = _transform(image).unsqueeze(0).to(_device)  # type: ignore[operator]
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]  # type: ignore[operator]
    inference_ms = (time.perf_counter() - t0) * 1000

    scores = {cls: round(prob.item(), 4) for cls, prob in zip(CLASS_NAMES, probs)}
    predicted_class = max(scores, key=scores.get)  # type: ignore[arg-type]

    _log_prediction(
        filename=file.filename or "unknown",
        predicted_class=predicted_class,
        confidence=scores[predicted_class],
        scores=scores,
        inference_ms=inference_ms,
    )

    return {
        "predicted_class": predicted_class,
        "confidence": scores[predicted_class],
        "scores": scores,
        "inference_ms": round(inference_ms, 2),
    }
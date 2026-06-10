from contextlib import asynccontextmanager
from datetime import datetime, timezone
import io
import json
import os
import time
from typing import Annotated

from fastapi import FastAPI, File, HTTPException
from fastapi import UploadFile as _UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import mlflow.pytorch
import numpy as np
from PIL import Image
from pydantic import WithJsonSchema
from timm.data import create_transform, resolve_data_config
import torch

from garbage_classification.config import PROJ_ROOT

UploadFile = Annotated[_UploadFile, WithJsonSchema({"type": "string", "format": "binary"})]

# Class names in alphabetical order — matches ImageFolder ordering used during training
# (cardboard=0, glass=1, metal=2, paper=3, plastic=4, trash=5)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

LOGS_DIR = PROJ_ROOT / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"
MONITORING_STATIC_DIR = PROJ_ROOT / "monitoring" / "static"

_model: torch.nn.Module | None = None
_transform = None
_device: torch.device | None = None


def _load_model() -> tuple[torch.nn.Module, object, torch.device]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
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


def _extract_image_properties(image: Image.Image) -> dict:
    """Compute low-level visual statistics used for image property drift detection.

    Returns a flat dict of scalar features:
      - brightness : mean pixel intensity (0-1)
      - blur_score : gradient variance proxy for sharpness (higher = sharper)
      - r_mean, g_mean, b_mean : per-channel mean intensities (0-1)
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = np.array(image.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(gray)
    return {
        "brightness": float(arr.mean()),
        "blur_score": float(np.var(gx) + np.var(gy)),
        "r_mean": float(arr[:, :, 0].mean()),
        "g_mean": float(arr[:, :, 1].mean()),
        "b_mean": float(arr[:, :, 2].mean()),
    }


def _log_prediction(
    filename: str,
    predicted_class: str,
    confidence: float,
    scores: dict,
    inference_ms: float,
    embedding: list[float],
    image_props: dict,
) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "scores": scores,
        "inference_ms": round(inference_ms, 2),
        "embedding": embedding,
        **image_props,
    }
    with PREDICTIONS_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _predict_one(image: Image.Image, filename: str) -> dict:
    """Run inference on one decoded image, log it, and return the result dict."""
    t0 = time.perf_counter()
    tensor = _transform(image).unsqueeze(0).to(_device)  # type: ignore[operator]

    with torch.no_grad():
        # Single backbone pass — derive both embedding and logits without recalculation
        features = _model.forward_features(tensor)  # type: ignore[union-attr]  # (1, C, H, W)
        if features.dim() == 4:
            pooled = features.mean(dim=[2, 3])  # global average pool -> (1, C)
        else:
            pooled = features
        embedding: list[float] = pooled.squeeze().cpu().numpy().tolist()
        logits = _model.forward_head(features)  # type: ignore[union-attr]  # pool + drop + FC
        probs = torch.softmax(logits, dim=1)[0]

    inference_ms = (time.perf_counter() - t0) * 1000

    scores = {cls: round(p.item(), 4) for cls, p in zip(CLASS_NAMES, probs)}
    predicted_class = max(scores, key=scores.get)  # type: ignore[arg-type]
    image_props = _extract_image_properties(image)

    _log_prediction(
        filename=filename,
        predicted_class=predicted_class,
        confidence=scores[predicted_class],
        scores=scores,
        inference_ms=inference_ms,
        embedding=embedding,
        image_props=image_props,
    )
    return {
        "filename": filename,
        "predicted_class": predicted_class,
        "confidence": scores[predicted_class],
        "scores": scores,
        "inference_ms": round(inference_ms, 2),
    }


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

@app.get(
    "/monitoring",
    tags=["monitoring"],
    summary="Open the drift monitoring dashboard",
    response_class=RedirectResponse,
    status_code=307,
)
def monitoring_dashboard():
    """Redirect to the Evidently drift monitoring dashboard.

    The dashboard is generated by `monitoring/run_drift.py` and served as static HTML
    under `/monitoring/`. Returns 404 if it has not been generated yet.

    This documented redirect exists so the dashboard is discoverable from Swagger
    (`/docs`) — mounted static files are otherwise absent from the OpenAPI schema.
    """
    if not MONITORING_STATIC_DIR.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Monitoring dashboard not generated yet. Run `python -m monitoring.run_drift`.",
        )
    return RedirectResponse(url="/monitoring/")


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", summary="Classify a waste image")
async def predict(file: UploadFile = File(..., description="Image file (jpg, png, webp, ...)")):
    """
    Upload a waste image and receive the predicted category with confidence scores.
    Accepts any image format supported by Pillow (jpg, png, webp, bmp, tiff, ...).

    - **predicted_class**: most likely waste category
    - **confidence**: softmax probability for the top class (0-1)
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

    result = _predict_one(image, file.filename or "unknown")
    # Drop 'filename' to preserve the original single-image response shape
    return {k: v for k, v in result.items() if k != "filename"}


# Mount the generated Evidently dashboard as static files at /monitoring/.
# Registered AFTER the documented /monitoring redirect route above so the exact-path
# route wins for "/monitoring" while the mount serves "/monitoring/<file>".
if MONITORING_STATIC_DIR.is_dir():
    app.mount(
        "/monitoring",
        StaticFiles(directory=str(MONITORING_STATIC_DIR), html=True),
        name="monitoring",
    )
else:
    logger.warning(
        f"Monitoring static dir not found at {MONITORING_STATIC_DIR}; "
        "/monitoring will return 404 until you run `python -m monitoring.run_drift`."
    )

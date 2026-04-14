# WasteNet MLOps

Garbage image classification pipeline — detects waste type from a photo.
Classes: **cardboard · glass · metal · paper · plastic · trash**

[![CI](https://github.com/MohCw/Wastenet-MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/MohCw/Wastenet-MLOps/actions)
![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-79%25-green)

**Stack:** DVC · MLflow · FastAPI · Docker · GitHub Actions · Evidently AI

---

## Architecture

```
data/raw/
    └─ DVC pipeline ──► data/processed/{train,val,test}/
                               │
                          train.py  ──► MLflow Model Registry (@champion)
                                               │
                                         api/main.py (FastAPI)
                                               │  └─► logs/predictions.jsonl
                                         Docker container       │
                                                          monitoring/run_drift.py
                                                               │
                                                         Evidently UI dashboard
```

---

## Dataset

**Source:** [Garbage Classification — Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

2,527 images across 6 classes:

| Class | Label |
|-------|-------|
| cardboard | 3 |
| glass | 1 |
| metal | 5 |
| paper | 2 |
| plastic | 4 |
| trash | 6 |

Predefined train/val/test splits are provided in `data/raw/` as `.txt` files.
Download manually:

```bash
poetry run python -m garbage_classification.download_dataset
```

---

## Quickstart

```bash
# 1. Install dependencies
poetry install

# 2. Run full pipeline (prepare → split → train → evaluate)
dvc repro

# 3. Start API
poetry run uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

---

## Docker

```bash
docker build -t wastenet-api .

docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
  wastenet-api
```

> **Note:** Pass `MLFLOW_TRACKING_URI` to point the container to your MLflow tracking server.
> The container exposes port `8000` by default; set `PORT` to override (Railway-compatible).

---

## Predict

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg"
# {"predicted_class":"plastic","confidence":0.93,"scores":{...},"inference_ms":14.2}
```

Or use the Swagger UI at `http://localhost:8000/docs`.

---

## MLflow Experiment Tracking

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://localhost:5000
```

Experiment: `garbage-classification/architecture-search`  
Best run promoted automatically to `@champion` alias in the `garbage-classifier` registry.

---

## Monitoring (Evidently AI)

After making predictions via the API (`logs/predictions.jsonl` is populated):

```bash
# 1. Run drift detection (compares prod distribution vs training set)
poetry run python -m monitoring.run_drift

# 2. Launch dashboard
poetry run evidently ui --workspace monitoring/workspace --port 8001
# → http://localhost:8001
```

---

## Port Map

| Service | Port |
|---|---|
| FastAPI (inference API) | `8000` |
| MLflow UI | `5000` |
| Evidently UI (monitoring) | `8001` |

---

## DVC Pipeline

| Stage | Output |
|---|---|
| prepare | `data/interim/cleaned/` |
| split | `data/processed/{train,val,test}/` |
| train | `models/model.pth` + `metrics/train_metrics.json` |
| evaluate | `metrics/eval_metrics.json` + `metrics/confusion_matrix.csv` |

Run a single stage: `dvc repro <stage>`

---

## Project Structure

```
├── garbage_classification/     # Core ML package
│   ├── config.py               # Path constants
│   ├── dataset.py              # DVC prepare + split stages
│   └── modeling/
│       ├── train.py            # Training + MLflow logging
│       └── predict.py          # Evaluation + MLflow logging
├── api/
│   └── main.py                 # FastAPI inference service
├── tests/
│   ├── conftest.py             # FastAPI TestClient fixture (mocks model load)
│   ├── test_api.py             # API endpoint tests (11 tests)
│   ├── test_config.py          # Path constants tests (3 tests)
│   ├── test_data.py            # CLASS_NAMES tests (2 tests)
│   └── test_dataset.py         # Dataset pipeline tests (9 tests)
├── monitoring/
│   └── run_drift.py            # Evidently drift detection
├── logs/
│   └── predictions.jsonl       # Production prediction log (gitignored)
├── dvc.yaml                    # Pipeline definition
├── params.yaml                 # Hyperparameters
├── Dockerfile
└── .github/workflows/ci.yml
```

---

## Code Quality

```bash
make lint      # ruff format --check + ruff check
make format    # ruff check --fix + ruff format
make test      # pytest tests/
```

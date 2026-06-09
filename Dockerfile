FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# CPU PyTorch 
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# API + MLflow + inference deps
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "python-multipart>=0.0.12" \
    "pillow>=11.0.0" \
    "timm>=1.0.26" \
    "mlflow>=3.10.1" \
    "loguru>=0.7.0" \
    "python-dotenv>=1.0.0"

COPY garbage_classification/ ./garbage_classification/
COPY api/ ./api/
COPY monitoring/static/ ./monitoring/static/

EXPOSE 8000

# Shell form required so Railway's $PORT variable is expanded at runtime
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}

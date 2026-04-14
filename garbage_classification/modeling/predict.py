import json
from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import timm
from timm.data import create_transform, resolve_data_config
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import typer
import yaml

from garbage_classification.config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()

PARAMS = yaml.safe_load((PROJ_ROOT / "params.yaml").read_text())["evaluate"]


@app.command()
def main(
    test_dir: Path = PROCESSED_DATA_DIR / "test",
    model_path: Path = MODELS_DIR / "model.pth",
    metrics_path: Path = METRICS_DIR / "eval_metrics.json",
    confusion_matrix_path: Path = METRICS_DIR / "confusion_matrix.csv",
    metadata_path: Path = MODELS_DIR / "model_metadata.json",
):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("garbage-classification/architecture-search")

    logger.info(f"Model   : {model_path}")
    logger.info(f"Test dir: {test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from metadata (arch + num_classes saved by train.py)
    metadata = json.loads(metadata_path.read_text())
    model = timm.create_model(
        metadata["arch"], pretrained=False, num_classes=metadata["num_classes"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    # Correct transforms for this architecture (input_size, mean, std from pretrained_cfg)
    data_cfg = resolve_data_config(model.pretrained_cfg, model=model)
    test_transform = create_transform(**data_cfg, is_training=False)

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=PARAMS["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    CLASS_NAMES = test_dataset.classes

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images.to(device)).argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds)
    test_acc = report["accuracy"]
    test_f1_mac = report["macro avg"]["f1-score"]

    run_id_file = METRICS_DIR / "mlflow_run_id.txt"
    run_id = run_id_file.read_text().strip() if run_id_file.exists() else None
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("stage", "evaluate")
        mlflow.log_metrics({"test_acc": test_acc, "test_f1_macro": test_f1_mac})
        for cls in CLASS_NAMES:
            mlflow.log_metric(f"f1_{cls}", report[cls]["f1-score"])
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(confusion_matrix_path)
        mlflow.log_artifact(str(confusion_matrix_path))

    eval_metrics = {"test_acc": test_acc, "test_f1_macro": test_f1_mac}
    metrics_path.write_text(json.dumps(eval_metrics, indent=2))
    logger.success(f"Evaluation complete — metrics → {metrics_path}")


if __name__ == "__main__":
    app()

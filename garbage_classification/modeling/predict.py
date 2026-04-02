import json
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

import mlflow
import typer
import yaml
from loguru import logger

from garbage_classification.config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()

PARAMS = yaml.safe_load((PROJ_ROOT / "params.yaml").read_text())["evaluate"]


@app.command()
def main(
    test_dir: Path = PROCESSED_DATA_DIR / "test",
    model_path: Path = MODELS_DIR / "model.pth",
    metrics_path: Path = METRICS_DIR / "eval_metrics.json",
    confusion_matrix_path: Path = METRICS_DIR / "confusion_matrix.csv",
):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_PARAMS = yaml.safe_load((PROJ_ROOT / "params.yaml").read_text())["train"]
    strategy = TRAIN_PARAMS.get("strategy", "v1")
    mlflow.set_experiment("garbage-classification")

    logger.info(f"Model   : {model_path}")
    logger.info(f"Test dir: {test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, 
                            batch_size=PARAMS["batch_size"],
                            shuffle=False,
                            num_workers=0)
    CLASS_NAMES = test_dataset.classes

    model = models.resnet18(weights=None)   # weights=None : on charge nos propres poids
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    #
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images.to(device)).argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    test_acc = report["accuracy"]
    test_f1_mac = report["macro avg"]["f1-score"]
    
    with mlflow.start_run(run_name=f"eval_resnet18_{strategy}"):
        mlflow.set_tag("stage", "evaluate")
        mlflow.set_tag("strategy", strategy)
        mlflow.log_metrics({"test_acc": test_acc, "test_f1_macro": test_f1_mac})
        for cls in CLASS_NAMES:
            mlflow.log_metric(f"f1_{cls}", report[cls]["f1-score"])
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(confusion_matrix_path)
        mlflow.log_artifact(str(confusion_matrix_path))
        
    logger.success(f"Evaluation complete — metrics → {metrics_path}")
    
    eval_metrics = {"test_acc": test_acc, "test_f1_macro": test_f1_mac}
    metrics_path.write_text(json.dumps(eval_metrics, indent=2))

if __name__ == "__main__":
    app()

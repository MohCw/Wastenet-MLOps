import json
from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import typer
import yaml
from loguru import logger

from garbage_classification.config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()

PARAMS = yaml.safe_load((PROJ_ROOT / "params.yaml").read_text())["train"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate_loader(model, loader, device):
    model.eval()
    correct = total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            running_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(loader.dataset), correct / total

@app.command()
def main(
    train_dir: Path = PROCESSED_DATA_DIR / "train",
    val_dir: Path = PROCESSED_DATA_DIR / "val",
    model_path: Path = MODELS_DIR / "model.pth",
    metrics_path: Path = METRICS_DIR / "train_metrics.json",
):
    set_seed(PARAMS.get("seed", 42))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    strategy = PARAMS.get("strategy", "linear_probe")
    model_arch = PARAMS.get("model_arch", "resnet18")
    mlflow.set_experiment("garbage-classification/architecture-search")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Model   : {PARAMS['model_arch']}")
    logger.info(f"Epochs  : {PARAMS['epochs']}")
    logger.info(f"Train   : {train_dir}")
    logger.info(f"Val     : {val_dir}")
    logger.info(f"Device : {device}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)

    train_loader  = DataLoader(
        train_dataset, 
        batch_size=PARAMS["batch_size"], 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False)
    val_loader    = DataLoader(
        val_dataset, 
        batch_size=PARAMS["batch_size"], 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False)
    
    NUM_CLASSES   = len(train_dataset.classes)

    with mlflow.start_run(run_name=f"{model_arch}_{strategy}") as run:
        mlflow.log_params(PARAMS)
        mlflow.set_tag("strategy", strategy)
        mlflow.set_tag("model_arch", PARAMS.get("model_arch", "resnet18"))

        # ── TRAINING LOOP  

        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

        if strategy == "linear_probe":
            # linear_probe: freeze backbone, train classification head only
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
            logger.info("Strategy linear_probe: head only")

        elif strategy == "partial_finetune":
            # partial_finetune: unfreeze top layers (layer4) + classification head
            for name, param in model.named_parameters():
                if "layer4" not in name and "fc" not in name:
                    param.requires_grad = False
            optimizer = torch.optim.Adam([
                {"params": filter(lambda p: p.requires_grad, model.layer4.parameters()), "lr": 1e-4},
                {"params": model.fc.parameters(), "lr": 1e-3}
            ])
            logger.info("Strategy partial_finetune: top layers + head")

        elif strategy == "full_finetune":
            # full_finetune: unfreeze all layers with differential learning rates
            backbone_params = []
            for name, param in model.named_parameters():
                if "fc" not in name:
                    backbone_params.append(param)
            optimizer = torch.optim.Adam([
                {"params": backbone_params, "lr": 1e-5},
                {"params": model.fc.parameters(), "lr": 1e-4}
            ])
            logger.info("Strategy full_finetune: all layers")

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_val_acc  = 0.0
        no_improve    = 0
        epochs_run    = 0

        for epoch in range(PARAMS["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate_loader(model, val_loader, device)
            epochs_run = epoch + 1

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
                step=epoch,
            )
            logger.info(
                f"Epoch {epoch+1:>3}/{PARAMS['epochs']}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc  = val_acc
                no_improve    = 0
                torch.save(model.state_dict(), model_path)
                logger.info(f"  → Best checkpoint (val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.4f})")
            else:
                no_improve += 1
                if no_improve >= PARAMS["early_stopping_patience"]:
                    logger.info(f"Early stopping triggered at epoch {epoch+1} (patience={PARAMS['early_stopping_patience']})")
                    break
        
        # Save model + metrics
        mlflow.log_metrics({"best_val_acc": best_val_acc, "best_val_loss": best_val_loss})
        mlflow.log_metric("epochs_run", epochs_run)

        # Infer model signature from a dummy input
        model.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224).to(device)
            dummy_output = model(dummy_input)
        signature = infer_signature(dummy_input.cpu().numpy(), dummy_output.cpu().numpy())

        # Log + register the model in the MLflow Model Registry
        mlflow.pytorch.log_model(model, name="model", registered_model_name="garbage-classifier", signature=signature)

        # Assign the @champion alias to this newly registered version
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if versions:
            latest_version = versions[0].version
            client.set_registered_model_alias("garbage-classifier", "champion", latest_version)
            mlflow.set_tag("registered_model_version", latest_version)
            logger.info(f"Model registered as 'garbage-classifier' v{latest_version} — alias @champion assigned")

        train_metrics = {
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "epochs_run": epochs_run,
        }
        metrics_path.write_text(json.dumps(train_metrics, indent=2))
        mlflow.log_artifact(str(metrics_path))

        # Persist run_id so evaluate stage logs to the same run
        (METRICS_DIR / "mlflow_run_id.txt").write_text(run.info.run_id)

    logger.success(f"Training complete — model → {model_path}")


if __name__ == "__main__":
    app()

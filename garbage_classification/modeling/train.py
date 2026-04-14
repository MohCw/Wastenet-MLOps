import json
from pathlib import Path
import random
import subprocess

from dotenv import load_dotenv
from loguru import logger
import mlflow
from mlflow.models import infer_signature
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import numpy as np
import timm
from timm.data import create_transform, resolve_data_config
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets
import typer
import yaml

from garbage_classification.config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT

load_dotenv()

app = typer.Typer()

PARAMS = yaml.safe_load((PROJ_ROOT / "params.yaml").read_text())["train"]


def get_git_info() -> dict[str, str]:
    """Return current git commit hash and branch, or empty strings if unavailable."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, branch = "", ""
    return {"git_commit": commit, "git_branch": branch}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(arch: str, num_classes: int) -> tuple[nn.Module, dict]:
    """Load pretrained timm model and return it with its data config."""
    model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    data_cfg = resolve_data_config(model.pretrained_cfg, model=model)
    return model, data_cfg


def apply_strategy(model: nn.Module, strategy: str, lr: float) -> torch.optim.Adam:
    """
    Freeze/unfreeze parameters based on strategy and return configured optimizer.
    Position-based splitting — works for any timm architecture (no hardcoded layer names).
    """
    head = model.get_classifier()
    head_param_ids = {id(p) for p in head.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    head_params = list(head.parameters())
    n = len(backbone_params)

    if strategy == "linear_probe":
        for p in backbone_params:
            p.requires_grad = False
        optimizer = torch.optim.Adam(head_params, lr=lr)
        logger.info("Strategy linear_probe: head only")

    elif strategy == "partial_finetune":
        split = int(n * 0.75)
        for p in backbone_params[:split]:
            p.requires_grad = False
        for p in backbone_params[split:]:
            p.requires_grad = True
        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params[split:], "lr": lr * 0.1},
                {"params": head_params, "lr": lr},
            ]
        )
        logger.info(f"Strategy partial_finetune: last 25% backbone ({n - split} tensors) + head")

    elif strategy == "full_finetune":
        third = n // 3
        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params[:third], "lr": lr * 0.01},
                {"params": backbone_params[third : 2 * third], "lr": lr * 0.1},
                {"params": backbone_params[2 * third :] + head_params, "lr": lr},
            ]
        )
        logger.info("Strategy full_finetune: all layers with differential LR")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return optimizer


def build_scheduler(name: str, optimizer, epochs: int, lr: float, warmup_epochs: int = 3):
    """
    Returns (scheduler, needs_val_loss).
      needs_val_loss=False → call scheduler.step() after each epoch (cosine).
      needs_val_loss=True  → call scheduler.step(val_loss) after validation (plateau).
    """
    if name == "none":
        return None, False

    if name == "cosine":
        warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_epochs)
        decay = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01)
        return SequentialLR(
            optimizer, schedulers=[warmup, decay], milestones=[warmup_epochs]
        ), False

    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=5, min_lr=lr * 1e-3
        ), True

    raise ValueError(f"Unknown scheduler: {name}")


def train_one_epoch(
    model, loader, optimizer, criterion, device, scheduler=None, needs_val_loss=False
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        running_total += labels.size(0)
    # cosine scheduler steps per epoch (not per batch)
    if scheduler is not None and not needs_val_loss:
        scheduler.step()
    return running_loss / len(loader.dataset), running_correct / running_total


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
    metadata_path: Path = MODELS_DIR / "model_metadata.json",
):
    set_seed(PARAMS.get("seed", 42))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    strategy = PARAMS.get("strategy", "linear_probe")
    model_arch = PARAMS.get("model_arch", "resnet18")
    lr = PARAMS.get("lr", 1e-3)
    scheduler_name = PARAMS.get("scheduler", "none")
    warmup_epochs = PARAMS.get("warmup_epochs", 3)

    mlflow.set_experiment("garbage-classification/arch-search")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Model    : {model_arch}")
    logger.info(f"Strategy : {strategy}")
    logger.info(f"Scheduler: {scheduler_name}")
    logger.info(f"Epochs   : {PARAMS['epochs']}")
    logger.info(f"Train    : {train_dir}")
    logger.info(f"Val      : {val_dir}")
    logger.info(f"Device   : {device}")

    # Build model with placeholder num_classes to obtain data_cfg (transforms, input size)
    model, data_cfg = build_model(model_arch, num_classes=1)
    input_size = data_cfg["input_size"][-2:]  # (H, W) — robust for H != W
    train_transform = create_transform(**data_cfg, is_training=True)
    val_transform = create_transform(**data_cfg, is_training=False)
    logger.info(f"Input size: {input_size}")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=PARAMS["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=PARAMS["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    NUM_CLASSES = len(train_dataset.classes)
    model.reset_classifier(NUM_CLASSES)  # replace placeholder head in-place
    model = model.to(device)

    with mlflow.start_run(run_name=f"{model_arch}_{strategy}") as run:
        params_to_log = {f"train.{k}": v for k, v in PARAMS.items() if k != "warmup_epochs"}
        params_to_log["train.warmup_epochs"] = (
            warmup_epochs if scheduler_name == "cosine" else "N/A"
        )
        mlflow.log_params(params_to_log)
        mlflow.set_tag("mlflow.runName", f"{model_arch}_{strategy}")
        mlflow.set_tag("strategy", strategy)
        mlflow.set_tag("model_arch", model_arch)
        mlflow.set_tags(get_git_info())

        optimizer = apply_strategy(model, strategy, lr)
        scheduler, needs_val_loss = build_scheduler(
            scheduler_name, optimizer, PARAMS["epochs"], lr, warmup_epochs
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_val_acc = 0.0
        no_improve = 0
        epochs_run = 0

        for epoch in range(PARAMS["epochs"]):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scheduler, needs_val_loss
            )
            val_loss, val_acc = evaluate_loader(model, val_loader, device)
            epochs_run = epoch + 1

            # ReduceLROnPlateau steps after validation
            if scheduler is not None and needs_val_loss:
                scheduler.step(val_loss)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )
            logger.info(
                f"Epoch {epoch + 1:>3}/{PARAMS['epochs']}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                no_improve = 0
                torch.save(model.state_dict(), model_path)
                logger.info(
                    f"  -> Best checkpoint (val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.4f})"
                )
            else:
                no_improve += 1
                if no_improve >= PARAMS["early_stopping_patience"]:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1} "
                        f"(patience={PARAMS['early_stopping_patience']})"
                    )
                    break

        # Save metadata (arch + classes + input size) for predict.py
        metadata_path.write_text(
            json.dumps(
                {
                    "arch": model_arch,
                    "num_classes": NUM_CLASSES,
                    "input_size": list(input_size),
                },
                indent=2,
            )
        )

        # Save metrics
        mlflow.log_metrics({"best_val_acc": best_val_acc, "best_val_loss": best_val_loss})
        mlflow.log_metric("epochs_run", epochs_run)

        # MLflow model signature
        model.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *input_size).to(device)
            dummy_output = model(dummy_input)
        signature = infer_signature(dummy_input.cpu().numpy(), dummy_output.cpu().numpy())

        mlflow.pytorch.log_model(
            model, name="model", registered_model_name="garbage-classifier", signature=signature
        )

        # Promote to @champion if this run beats the current one
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if versions:
            new_version = versions[0].version
            mlflow.set_tag("registered_model_version", new_version)

            champion_acc = 0.0
            try:
                champion_mv = client.get_model_version_by_alias("garbage-classifier", "champion")
                champion_run = client.get_run(champion_mv.run_id)
                champion_acc = champion_run.data.metrics.get("best_val_acc", 0.0)
            except Exception:
                pass

            if best_val_acc > champion_acc:
                client.set_registered_model_alias("garbage-classifier", "champion", new_version)
                mlflow.set_tag("promoted_to_champion", "true")
                logger.info(
                    f"New @champion: v{new_version} "
                    f"(val_acc={best_val_acc:.4f} > current={champion_acc:.4f})"
                )
            else:
                mlflow.set_tag("promoted_to_champion", "false")
                logger.info(
                    f"Champion unchanged — v{new_version} val_acc={best_val_acc:.4f} "
                    f"did not beat current champion ({champion_acc:.4f})"
                )

        train_metrics = {
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "epochs_run": epochs_run,
        }
        metrics_path.write_text(json.dumps(train_metrics, indent=2))
        mlflow.log_artifact(str(metrics_path))

        (METRICS_DIR / "mlflow_run_id.txt").write_text(run.info.run_id)

    logger.success(f"Training complete — model → {model_path}")


if __name__ == "__main__":
    app()

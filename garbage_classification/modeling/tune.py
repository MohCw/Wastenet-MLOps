"""Optuna grid search for the champion config: learning rate x scheduler."""

import json
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
import mlflow
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import GridSampler
from timm.data import create_transform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import typer

from garbage_classification.config import METRICS_DIR, PROCESSED_DATA_DIR
from garbage_classification.modeling.train import (
    PARAMS,
    apply_strategy,
    build_model,
    build_scheduler,
    evaluate_loader,
    get_git_info,
    set_seed,
    train_one_epoch,
)

load_dotenv()

app = typer.Typer()

MODEL_ARCH = "convnext_tiny.in12k_ft_in1k"
STRATEGY = "partial_finetune"

EPOCHS = PARAMS["epochs"]
BATCH_SIZE = PARAMS["batch_size"]
PATIENCE = PARAMS["early_stopping_patience"]
WARMUP_EPOCHS = PARAMS.get("warmup_epochs", 3)
SEED = PARAMS.get("seed", 42)

EXPERIMENT_NAME = "garbage-classification/tune-lr-scheduler"
SCHEDULER_CHOICES = ["none", "cosine", "plateau"]

LR_GRID = [1e-4, 3e-4, 1e-3]


def _build_loaders(device):
    _, data_cfg = build_model(MODEL_ARCH, num_classes=1)
    input_size = data_cfg["input_size"][-2:]
    train_tf = create_transform(**data_cfg, is_training=True)
    val_tf = create_transform(**data_cfg, is_training=False)
    train_ds = datasets.ImageFolder(PROCESSED_DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(PROCESSED_DATA_DIR / "val", transform=val_tf)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    logger.info(f"Loaders built — {len(train_ds.classes)} classes, input {input_size}")
    return train_loader, val_loader, len(train_ds.classes)


def make_objective(train_loader, val_loader, num_classes, device):
    """Build the Optuna objective. Returns best validation accuracy (to maximize)."""

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", min(LR_GRID), max(LR_GRID), log=True)
        scheduler_name = trial.suggest_categorical("scheduler", SCHEDULER_CHOICES)

        set_seed(SEED)

        model, _ = build_model(MODEL_ARCH, num_classes=1)
        model.reset_classifier(num_classes)
        model = model.to(device)

        optimizer = apply_strategy(model, STRATEGY, lr)
        scheduler, needs_val_loss = build_scheduler(
            scheduler_name, optimizer, EPOCHS, lr, WARMUP_EPOCHS
        )
        criterion = nn.CrossEntropyLoss()

        # One MLflow run per trial; lr+scheduler in the name avoids name collisions.
        run_name = f"{MODEL_ARCH}_{STRATEGY}_lr{lr:.1e}_{scheduler_name}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(
            {
                "train.model_arch": MODEL_ARCH,
                "train.strategy": STRATEGY,
                "train.lr": lr,
                "train.scheduler": scheduler_name,
                "train.epochs": EPOCHS,
                "train.batch_size": BATCH_SIZE,
                "train.early_stopping_patience": PATIENCE,
                "train.warmup_epochs": WARMUP_EPOCHS if scheduler_name == "cosine" else "N/A",
                "train.seed": SEED,
            }
        )
        mlflow.set_tag("strategy", STRATEGY)
        mlflow.set_tag("model_arch", MODEL_ARCH)
        mlflow.set_tag("tuning", "optuna")
        mlflow.set_tag("optuna_trial", trial.number)
        mlflow.set_tags(get_git_info())

        best_val_loss = float("inf")
        best_val_acc = 0.0
        no_improve = 0
        epochs_run = 0
        pruned = False

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scheduler, needs_val_loss
            )
            val_loss, val_acc = evaluate_loader(model, val_loader, device)
            epochs_run = epoch + 1

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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                no_improve = 0
            else:
                no_improve += 1

            trial.report(val_acc, epoch)
            if trial.should_prune():
                pruned = True
                logger.info(f"Trial {trial.number} pruned at epoch {epoch + 1}")
                break

            if no_improve >= PATIENCE:
                logger.info(f"Trial {trial.number} early-stopped at epoch {epoch + 1}")
                break

        mlflow.log_metrics({"best_val_acc": best_val_acc, "best_val_loss": best_val_loss})
        mlflow.log_metric("epochs_run", epochs_run)
        mlflow.set_tag("pruned", "true" if pruned else "false")
        mlflow.end_run()

        if pruned:
            raise optuna.TrialPruned()
        return best_val_acc

    return objective


@app.command()
def main(
    n_trials: Optional[int] = None,
    experiment: str = EXPERIMENT_NAME,
):
    """Run the lr x scheduler grid study and report the best config."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_experiment(experiment)

    search_space = {"lr": LR_GRID, "scheduler": SCHEDULER_CHOICES}
    grid_size = len(LR_GRID) * len(SCHEDULER_CHOICES)
    if n_trials is None:
        n_trials = grid_size  # full grid by default

    logger.info(
        f"Tuning {MODEL_ARCH} + {STRATEGY} | grid lr={LR_GRID} "
        f"x scheduler={SCHEDULER_CHOICES} ({grid_size} combos) | {n_trials} trials | {device}"
    )

    train_loader, val_loader, num_classes = _build_loaders(device)

    sampler = GridSampler(search_space, seed=SEED)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner, study_name="lr-scheduler"
    )
    study.optimize(
        make_objective(train_loader, val_loader, num_classes, device),
        n_trials=n_trials,
    )

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    logger.success(
        f"Best val_acc={study.best_value:.4f} with {study.best_params} "
        f"({len(study.trials)} trials, {n_pruned} pruned)"
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "tune_best.json").write_text(
        json.dumps(
            {
                "best_val_acc": study.best_value,
                **study.best_params,
                "n_trials": len(study.trials),
                "n_pruned": n_pruned,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()

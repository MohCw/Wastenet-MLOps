from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torchvision.utils import make_grid


def plot_class_distribution(class_counts: dict, save_path: Path | None = None) -> None:
    """Bar chart + pie chart of image counts per class."""
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = plt.cm.Set2.colors[: len(classes)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Class Distribution", fontsize=14, fontweight="bold")

    # Bar chart
    bars = axes[0].bar(classes, counts, color=colors, edgecolor="white")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Number of images")
    axes[0].set_title("Images per class")
    for bar, count in zip(bars, counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Pie chart
    axes[1].pie(counts, labels=classes, colors=colors, autopct="%1.1f%%", startangle=140)
    axes[1].set_title("Class proportions")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved class distribution plot to {save_path}")
    plt.show()


def plot_sample_images(
    images_per_class: dict,
    n_per_class: int = 5,
    save_path: Path | None = None,
) -> None:
    """Grid of sample images per class using torchvision make_grid.

    Args:
        images_per_class: dict mapping class_name -> list of tensors (C, H, W) in [0, 1]
        n_per_class: number of samples to show per class
        save_path: optional path to save the figure
    """
    rows = []
    for class_name, tensors in images_per_class.items():
        samples = tensors[:n_per_class]
        if len(samples) < n_per_class:
            logger.warning(f"Class '{class_name}' has only {len(samples)} samples")
        rows.append(torch.stack(samples))  # (n, C, H, W)

    grid_tensor = torch.cat(rows, dim=0)  # (n_classes * n_per_class, C, H, W)
    grid = make_grid(grid_tensor, nrow=n_per_class, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).numpy()

    n_classes = len(images_per_class)
    fig, ax = plt.subplots(figsize=(n_per_class * 2, n_classes * 2))
    ax.imshow(grid_np)
    ax.axis("off")
    ax.set_title("Sample images per class", fontsize=13, fontweight="bold")

    # Class name labels on the left
    img_h = grid_np.shape[0] / n_classes
    for i, class_name in enumerate(images_per_class.keys()):
        ax.text(
            -5,
            img_h * i + img_h / 2,
            class_name,
            va="center",
            ha="right",
            fontsize=10,
            fontweight="bold",
            transform=ax.transData,
        )

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved sample images plot to {save_path}")
    plt.show()

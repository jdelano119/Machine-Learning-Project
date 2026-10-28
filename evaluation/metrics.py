import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    label_names: List[str],
    output_path: Path,
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def save_results_json(results: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def plot_experiment_comparison(results: Dict[str, Dict[str, float]], output_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    experiments = list(results.keys())

    x = np.arange(len(experiments))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for index, metric_name in enumerate(metrics):
        values = [results[exp][metric_name] for exp in experiments]
        ax.bar(x + (index - 1.5) * width, values, width=width, label=metric_name.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison Across Experiments")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

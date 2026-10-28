import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from evaluation.metrics import (
    compute_metrics,
    plot_experiment_comparison,
    save_confusion_matrix,
    save_results_json,
)
from model_training.dataset import PokemonImageDataset
from model_training.models import build_model


EXPERIMENTS = {
    "original_to_original": {"train": "original", "val": "original", "test": "original"},
    "original_to_silhouette": {"train": "original", "val": "original", "test": "silhouette"},
    "silhouette_to_silhouette": {"train": "silhouette", "val": "silhouette", "test": "silhouette"},
}


def make_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for inputs, labels, _ in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (predictions == labels).sum().item()
        running_total += labels.size(0)

    return running_loss / running_total, running_correct / running_total


def evaluate(model, loader, criterion, device) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for inputs, labels, _ in tqdm(loader, desc="Eval", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            running_loss += loss.item() * inputs.size(0)
            running_correct += (predictions == labels).sum().item()
            running_total += labels.size(0)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return running_loss / running_total, running_correct / running_total, y_true, y_pred


def make_loader(manifest_path, data_root, split, mode, transform, batch_size, shuffle) -> DataLoader:
    dataset = PokemonImageDataset(
        manifest_path=manifest_path,
        data_root=data_root,
        split=split,
        image_mode=mode,
        transform=transform,
        require_pairs=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def run_experiment(
    experiment_name: str,
    experiment_modes: Dict[str, str],
    manifest_path: str,
    data_root: str,
    output_dir: Path,
    model_type: str,
    num_classes: int,
    label_names: List[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    image_size: int,
    device: torch.device,
):
    transform = make_transforms(image_size)

    train_loader = make_loader(manifest_path, data_root, "train", experiment_modes["train"], transform, batch_size, True)
    val_loader = make_loader(manifest_path, data_root, "val", experiment_modes["val"], transform, batch_size, False)
    test_loader = make_loader(manifest_path, data_root, "test", experiment_modes["test"], transform, batch_size, False)

    model = build_model(model_type=model_type, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{experiment_name}] epoch {epoch}/{epochs} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(y_true, y_pred)
    metrics["test_accuracy_from_loop"] = test_acc

    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), experiment_dir / f"{model_type}_weights.pt")
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
        output_path=experiment_dir / "confusion_matrix.png",
        title=f"{model_type} - {experiment_name}",
    )

    save_results_json(
        {
            "experiment": experiment_name,
            "model_type": model_type,
            "modes": experiment_modes,
            "history": history,
            "metrics": metrics,
        },
        experiment_dir / "results.json",
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pokémon classifiers across image-domain experiments.")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model", default="cnn", choices=["cnn", "mobilenetv2"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["original_to_original", "original_to_silhouette", "silhouette_to_silhouette"],
        choices=list(EXPERIMENTS.keys()),
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    valid_manifest = manifest[manifest["status"] == "ok"].copy()
    label_table = valid_manifest[["label_idx", "label"]].drop_duplicates().sort_values("label_idx")
    label_names = label_table["label"].tolist()
    num_classes = len(label_names)

    if num_classes < 2:
        raise ValueError("Need at least two classes to train a classifier.")

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    summary: Dict[str, Dict[str, float]] = {}

    for experiment_name in args.experiments:
        metrics = run_experiment(
            experiment_name=experiment_name,
            experiment_modes=EXPERIMENTS[experiment_name],
            manifest_path=args.manifest,
            data_root=args.data_root,
            output_dir=output_dir,
            model_type=args.model,
            num_classes=num_classes,
            label_names=label_names,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            image_size=args.image_size,
            device=device,
        )
        summary[experiment_name] = metrics

    save_results_json(summary, output_dir / "summary_metrics.json")
    plot_experiment_comparison(summary, output_dir / "experiment_comparison.png")

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    print("Saved outputs to", output_dir)


if __name__ == "__main__":
    main()

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize original and silhouette image pairs.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    subset = manifest[(manifest["split"] == args.split) & (manifest["status"] == "ok")]

    if subset.empty:
        raise ValueError("No paired images with status='ok' were found for this split.")

    random.seed(args.seed)
    rows = subset.to_dict("records")
    chosen = random.sample(rows, k=min(args.num_samples, len(rows)))

    fig, axes = plt.subplots(len(chosen), 2, figsize=(8, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = [axes]

    for axis_pair, item in zip(axes, chosen):
        original_image = Image.open(data_dir / item["original_path"])
        silhouette_image = Image.open(data_dir / item["silhouette_path"])

        axis_pair[0].imshow(original_image)
        axis_pair[0].set_title(f"Original: {item['label']}")
        axis_pair[0].axis("off")

        axis_pair[1].imshow(silhouette_image)
        axis_pair[1].set_title("Silhouette")
        axis_pair[1].axis("off")

    plt.tight_layout()

    if args.save_path:
        output_path = Path(args.save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

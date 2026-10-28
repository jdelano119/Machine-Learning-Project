from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PokemonImageDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        split: str,
        image_mode: str,
        transform=None,
        require_pairs: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.transform = transform

        manifest = pd.read_csv(manifest_path)
        subset = manifest[manifest["split"] == split].copy()

        if image_mode == "silhouette":
            subset = subset[subset["status"] == "ok"]
            subset = subset[subset["silhouette_path"].astype(str).str.len() > 0]
        elif image_mode == "original" and require_pairs:
            subset = subset[subset["status"] == "ok"]

        self.samples = subset.reset_index(drop=True)
        self.image_column = "original_path" if image_mode == "original" else "silhouette_path"

        if self.samples.empty:
            raise ValueError(f"No samples found for split={split}, image_mode={image_mode}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples.iloc[index]
        image_path = self.data_root / row[self.image_column]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label_idx"])
        return image, label, row["image_id"]

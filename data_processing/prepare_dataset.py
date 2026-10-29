import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import ClassLabel, Image as HFImageFeature
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split


DEFAULT_SPECIES = [
    "Bulbasaur",
    "Ivysaur",
    "Venusaur",
    "Charmander",
    "Charmeleon",
    "Charizard",
    "Squirtle",
    "Wartortle",
    "Blastoise",
    "Pikachu",
    "Raichu",
    "Jigglypuff",
    "Meowth",
    "Psyduck",
    "Machop",
    "Geodude",
    "Eevee",
    "Snorlax",
]


def normalize_name(value: str) -> str:
    return "".join(char.lower() for char in str(value) if char.isalnum())


def infer_columns(dataset_split) -> Tuple[str, str]:
    image_col = None
    for column_name, feature in dataset_split.features.items():
        if isinstance(feature, HFImageFeature):
            image_col = column_name
            break

    if image_col is None:
        raise ValueError("Could not find an image column in this dataset.")

    candidate_label_columns = [
        "pokemon",
        "name",
        "species",
        "label",
        "class",
        "category",
    ]
    available_columns = set(dataset_split.column_names)

    for candidate in candidate_label_columns:
        if candidate in available_columns:
            return image_col, candidate

    non_image_columns = [column for column in dataset_split.column_names if column != image_col]
    if not non_image_columns:
        raise ValueError("Could not find a label column in this dataset.")

    return image_col, non_image_columns[0]


def extract_species(raw_label: str) -> str:
    text = str(raw_label)
    text = text.replace("_", " ").replace("-", " ")
    parts = text.split()
    if not parts:
        return text

    if parts[0].isdigit() and len(parts) > 1:
        parts = parts[1:]

    return " ".join(part.capitalize() for part in parts)


def pil_from_dataset_image(image_object) -> Image.Image:
    if isinstance(image_object, Image.Image):
        return image_object
    if isinstance(image_object, dict) and "bytes" in image_object and image_object["bytes"] is not None:
        from io import BytesIO

        return Image.open(BytesIO(image_object["bytes"])).convert("RGBA")
    if isinstance(image_object, dict) and "path" in image_object and image_object["path"]:
        return Image.open(image_object["path"])
    raise ValueError("Unsupported image format returned by dataset.")


def create_silhouette_with_alpha(image: Image.Image, threshold: int = 128) -> Tuple[Optional[Image.Image], str]:
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.split()[-1])

    has_transparency = np.any(alpha < 255)
    if not has_transparency:
        return None, "no_transparent_background"

    silhouette = np.zeros((rgba.height, rgba.width, 4), dtype=np.uint8)
    silhouette[:, :, 3] = (alpha >= threshold).astype(np.uint8) * 255

    return Image.fromarray(silhouette, mode="RGBA"), "ok"


def make_stratified_split(ids: List[str], labels: List[int], seed: int) -> Dict[str, List[int]]:
    indices = list(range(len(ids)))
    label_counts = Counter(labels)
    can_stratify = min(label_counts.values()) >= 3

    stratify_input = labels if can_stratify else None

    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices,
        labels,
        test_size=0.3,
        random_state=seed,
        stratify=stratify_input,
    )

    if can_stratify:
        temp_label_counts = Counter(temp_labels)
        stratify_temp = temp_labels if min(temp_label_counts.values()) >= 2 else None
    else:
        stratify_temp = None

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=seed,
        stratify=stratify_temp,
    )

    return {"train": train_indices, "val": val_indices, "test": test_indices}


def parse_species_list(species_text: Optional[str]) -> List[str]:
    if not species_text:
        return DEFAULT_SPECIES
    return [item.strip() for item in species_text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Pokémon images and generate silhouette pairs.")
    parser.add_argument("--dataset", default="RogerKoala/gen1-pokemon-images")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--species", default=None, help="Comma-separated species list.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_species = parse_species_list(args.species)
    normalized_species = {normalize_name(name): name for name in selected_species}

    dataset_dict = load_dataset(args.dataset)
    split_name = "train" if "train" in dataset_dict else list(dataset_dict.keys())[0]
    dataset_split = dataset_dict[split_name]

    image_col, label_col = infer_columns(dataset_split)
    print(f"Using split='{split_name}', image column='{image_col}', label column='{label_col}'.")

    filtered_rows = []
    for row_index, row in enumerate(dataset_split):
        raw_label = row[label_col]
        label_feature = dataset_split.features[label_col]
        if isinstance(label_feature, ClassLabel):
            raw_label = label_feature.int2str(raw_label)
        species_name = extract_species(raw_label)
        normalized_label = normalize_name(species_name)
        if normalized_label in normalized_species:
            filtered_rows.append(
                {
                    "row_index": row_index,
                    "species": normalized_species[normalized_label],
                    "normalized_species": normalized_label,
                }
            )

    if not filtered_rows:
        raise ValueError("No rows matched the selected species list.")

    label_names = sorted({row["species"] for row in filtered_rows})
    label_to_index = {label: index for index, label in enumerate(label_names)}

    image_ids = [f"{split_name}_{row['row_index']}" for row in filtered_rows]
    labels = [label_to_index[row["species"]] for row in filtered_rows]
    split_indices = make_stratified_split(image_ids, labels, args.seed)

    records = []
    review_records = []

    for split, indices in split_indices.items():
        for index in indices:
            row_meta = filtered_rows[index]
            row = dataset_split[row_meta["row_index"]]
            label = row_meta["species"]
            label_index = label_to_index[label]
            image_id = f"{split_name}_{row_meta['row_index']}"

            source_image = pil_from_dataset_image(row[image_col]).convert("RGBA")

            original_dir = output_dir / "images" / "original" / split / label
            silhouette_dir = output_dir / "images" / "silhouette" / split / label
            original_dir.mkdir(parents=True, exist_ok=True)
            silhouette_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{image_id}.png"
            original_path = original_dir / filename
            silhouette_path = silhouette_dir / filename

            source_image.save(original_path)

            silhouette_image, status = create_silhouette_with_alpha(source_image)
            if silhouette_image is not None:
                silhouette_image.save(silhouette_path)
                silhouette_rel_path = os.path.relpath(silhouette_path, output_dir)
            else:
                silhouette_rel_path = ""
                review_records.append(
                    {
                        "image_id": image_id,
                        "label": label,
                        "split": split,
                        "reason": status,
                        "original_path": os.path.relpath(original_path, output_dir),
                    }
                )

            records.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "label": label,
                    "label_idx": label_index,
                    "status": status,
                    "original_path": os.path.relpath(original_path, output_dir),
                    "silhouette_path": silhouette_rel_path,
                }
            )

    manifest_path = output_dir / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    review_path = output_dir / "needs_review.csv"
    pd.DataFrame(review_records).to_csv(review_path, index=False)

    class_map_path = output_dir / "class_mapping.json"
    with open(class_map_path, "w", encoding="utf-8") as file:
        json.dump(label_to_index, file, indent=2)

    summary = {
        "dataset": args.dataset,
        "split_used": split_name,
        "image_column": image_col,
        "label_column": label_col,
        "selected_species": selected_species,
        "counts": {
            "total": len(records),
            "train": len(split_indices["train"]),
            "val": len(split_indices["val"]),
            "test": len(split_indices["test"]),
            "needs_review": len(review_records),
            "silhouettes_generated": sum(1 for row in records if row["status"] == "ok"),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Saved:")
    print(f"  - {manifest_path}")
    print(f"  - {review_path}")
    print(f"  - {class_map_path}")
    print(f"  - {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

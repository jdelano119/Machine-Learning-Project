# Who's That Pokémon? — Image Recognition Using Machine Learning

This repository contains a beginner-friendly undergraduate research project that studies how color and internal visual details affect Pokémon image classification.

The central idea is to compare model performance between original color sprites and silhouette-only versions while keeping outline, pose, size, and position unchanged.

## Project structure

- `/data_processing` — dataset download, class filtering, split generation, silhouette generation, pair visualization script
- `/model_training` — dataset loader, model definitions, training loop for required experiments
- `/evaluation` — metrics, confusion matrix plots, experiment comparison charts
- `/notebooks` — notebook for visualizing original/silhouette pairs
- `/data` — generated local data files (created after running preprocessing)

## Tools and libraries

- Python
- PyTorch + torchvision
- Pillow
- scikit-learn
- matplotlib
- Hugging Face `datasets`

## 1) Installation

### Local environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Google Colab

```python
!pip install -r requirements.txt
```

> If you use Colab, clone your repository first and run commands from the repo root.

## 2) Download and prepare data (focus-first pipeline)

This first script handles:
- downloading the Hugging Face Generation 1 dataset
- selecting ~15–20 target Pokémon species
- splitting image IDs into train/val/test **before** silhouette generation (leakage prevention)
- generating silhouette images using transparency masks
- flagging non-transparent images for manual review (instead of generating incorrect black rectangles)

Run:

```bash
python -m data_processing.prepare_dataset --output-dir data
```

Optional species override (comma-separated):

```bash
python -m data_processing.prepare_dataset \
  --output-dir data \
  --species "Bulbasaur,Ivysaur,Venusaur,Charmander,Charmeleon,Charizard,Squirtle,Wartortle,Blastoise,Pikachu,Raichu,Jigglypuff,Meowth,Psyduck,Machop,Geodude,Eevee,Snorlax"
```

### Output files

- `data/manifest.csv` — full image manifest with split, label, and matching paths
- `data/needs_review.csv` — images without transparency that require manual handling
- `data/class_mapping.json` — class-to-index mapping
- `data/summary.json` — preprocessing summary counts
- `data/images/original/...` and `data/images/silhouette/...` — matched image folders by split and label

## 3) Visualize image pairs

### Script

```bash
python -m data_processing.visualize_pairs --data-dir data --split train --num-samples 8
```

### Notebook

Open:
- `notebooks/pokemon_pairs_demo.ipynb`

The notebook displays original Pokémon images beside their silhouette counterparts.

## 4) Train models

Two model options are included:

1. Baseline small CNN (`cnn`)
2. Transfer learning with pretrained MobileNetV2 (`mobilenetv2`)

### Supported experiments

- `original_to_original`: train on original color images, test on original color images
- `original_to_silhouette`: train on original color images, test on silhouettes
- `silhouette_to_silhouette`: train on silhouettes, test on silhouettes

### Train baseline CNN

```bash
python -m model_training.train \
  --manifest data/manifest.csv \
  --data-root data \
  --model cnn \
  --epochs 8 \
  --batch-size 32 \
  --output-dir outputs
```

### Train MobileNetV2

```bash
python -m model_training.train \
  --manifest data/manifest.csv \
  --data-root data \
  --model mobilenetv2 \
  --epochs 8 \
  --batch-size 32 \
  --output-dir outputs
```

## 5) Evaluation outputs

For each model, the training script saves:

- per-experiment `results.json`
- confusion matrix plot (`confusion_matrix.png`)
- `summary_metrics.json` with accuracy / precision / recall / F1
- `experiment_comparison.png` comparing all experiments

All outputs are saved under:

- `outputs/cnn/...`
- `outputs/mobilenetv2/...`

## Reproducibility notes

- Splits are generated once from image IDs before silhouette creation to reduce leakage risk.
- Silhouettes are generated only when transparency masks are available.
- Non-transparent images are flagged for review in `needs_review.csv`.
- Do not commit large generated datasets or trained model weights to GitHub.

## Suggested next steps for the research team

1. Review `needs_review.csv` and clean/fix problematic sprites.
2. Run both models with the same seeds and hyperparameters.
3. Compare experiment metrics and confusion matrices.
4. Write a short analysis of which Pokémon classes are most affected by silhouette-only inputs.

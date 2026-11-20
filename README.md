# Who's That Pokémon? Investigating How Image Augmentations Work Together

**CS 445 — Machine Learning Poster Research Project**  
**Authors:** Jack Delano and John Paul

## Project Overview

This project investigates how different image augmentation techniques affect a neural network's ability to recognize Generation 1 Pokémon.

Image augmentation is a common machine learning technique that modifies training images to help models generalize to new examples. Common transformations include blurring images, removing portions of an image, and flipping images horizontally.

While these techniques may improve classification accuracy individually, it is not always clear whether combining them produces even better results.

Our project investigates whether the improvements from individual augmentations stack together, overlap, or interfere with one another.

We will train multiple versions of the same image classification model using different combinations of augmentation techniques and compare their performance on the same test dataset.

## Research Question

**Does combining Gaussian blur, random erasing, and horizontal flipping improve Pokémon classification accuracy beyond the gains achieved by each augmentation technique individually?**

More specifically, we want to investigate:

1. How much does each individual augmentation improve classification accuracy?
2. Do combinations of two augmentations outperform the individual techniques?
3. Does combining all three augmentations produce the highest accuracy?
4. Are the improvements additive, or do some techniques interfere with each other?
5. Do these augmentations also improve recognition of Pokémon silhouettes?

### Hypothesis

We hypothesize that certain combinations of image augmentations will improve classification accuracy more than individual techniques.

However, combining all three augmentations may not necessarily provide the greatest improvement. Some techniques may provide overlapping benefits or remove too much visual information during training.

## Dataset

We use the Generation 1 Pokémon image dataset available on Hugging Face.

**Dataset:** [RogerKoala/gen1-pokemon-images](https://huggingface.co/datasets/RogerKoala/gen1-pokemon-images)

The dataset contains labeled images of original Generation 1 Pokémon. The
dataset preparation script uses 18 species by default and creates
train/validation/test splits for both original images and silhouettes.

## Running the Project

Run the commands below from the repository root.

### 1. Create an environment and install dependencies

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Prepare the dataset

Download the dataset from Hugging Face, select the default Pokémon species,
split the images into train/validation/test sets, and generate silhouettes:

```bash
python -m data_processing.prepare_dataset
```

An internet connection is required for this download. The Hugging Face dataset
cache is reused on later runs.

This creates `data/manifest.csv`, `data/class_mapping.json`,
`data/summary.json`, and the image files under `data/images/`.

To select a different set of species or output directory:

```bash
python -m data_processing.prepare_dataset \
  --species "Pikachu,Raichu,Eevee" \
  --output-dir data
```

### 3. Visualize original/silhouette pairs

```bash
python -m data_processing.visualize_pairs --split train --num-samples 8
```

To save the visualization instead of displaying it:

```bash
python -m data_processing.visualize_pairs \
  --split test \
  --num-samples 8 \
  --save-path outputs/test_pairs.png
```

### 4. Train and evaluate the models

The default training command runs the three configured experiments:

```bash
python -m model_training.train
```

Training outputs are saved under `outputs/cnn/`, including model weights,
metrics, confusion matrices, and a comparison plot. Useful options include:

```bash
python -m model_training.train \
  --model mobilenetv2 \
  --epochs 8 \
  --batch-size 32 \
  --experiments original_to_original original_to_silhouette
```

The script automatically uses a CUDA GPU when one is available; otherwise it
uses the CPU.

### 5. Run the demonstration notebook

After preparing the dataset, start Jupyter with:

```bash
jupyter notebook
```

Open `notebooks/pokemon_pairs_demo.ipynb` in the browser and run its cells.

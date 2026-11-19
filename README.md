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

The dataset contains labeled images of original Generation 1 Pokémon.

We plan to select approximately 15–20 Pokémon species with enough

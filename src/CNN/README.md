
# Sounding Canvas: Image-to-Sound Feature Mapping

This module is the "offline" component of the Sounding Canvas project, responsible for training a system that maps visual characteristics of an image to a set of predefined sound descriptors. This mapping is encapsulated in a transformation matrix `T`, which can then be used in the "online" part of the project to dynamically generate sound based on visual input.

## Overview

The core idea is to learn a linear relationship between high-dimensional image features (extracted by a Convolutional Neural Network) and a concise 10-dimensional vector of perceptual sound features.

The workflow involves:
1.  **Image Feature Extraction**: Processing images to get their numerical visual representations.
2.  **Audio Feature Extraction & Normalization**: Analyzing audio files to extract and normalize their core acoustic descriptors.
3.  **T Matrix Calculation**: Training a simple linear model to learn the mapping from image features to sound features.
4.  **Prediction**: Using the learned `T` matrix to generate sound descriptors for new, unseen images.

## Setup

### Prerequisites

* Python 3.8+ (recommended)
* `pip` for package management

### Installation

Navigate to the `src/CNN` directory (or wherever your Python scripts reside) and install the necessary libraries:

```bash
pip install torch torchvision pandas numpy librosa scikit-learn Pillow
```

**Note for Librosa:** For full audio format support (like MP3s), you might also need `ffmpeg` installed on your system.
* **Ubuntu/Debian:** `sudo apt-get install ffmpeg`
* **macOS (Homebrew):** `brew install ffmpeg`
* **Windows:** Download binaries from the [FFmpeg website](https://ffmpeg.org/download.html) and add them to your system's PATH.

**Important:** For the training phase, ensure your image filenames in the `images/` directory correspond to the audio filenames in the `sounds/` directory by their base name (e.g., `my_visual.jpg` should correspond to `my_visual.wav`). This is crucial for aligning the datasets.

## Usage Workflow

Follow these steps sequentially to train your model and predict sound features.

### 1. Extract Image Features (Manual Step / Assumed Pre-existing)

This step involves extracting deep features from your training images using a pre-trained Convolutional Neural Network (e.g., ResNet50). Afters using feature_extractor.py, it's assumed that you have processed your images and have a `feature_vectors.csv` file ready.

* **Input:** Image files in the `images/` directory.
* **Expected Output:** `feature_vectors.csv` (a CSV file where each row is a 2048-dimensional feature vector for an image, indexed by its base filename).
* **Note:** The image feature extractor (e.g., the ResNet50 model and its preprocessing steps) used to generate this file must be consistent with what `predict_sound_features.py` will use later.

### 2. Extract and Normalize Sound Features

This script processes your audio files, extracts 10 key perceptual features, and normalizes them for consistent training.

* **Script:** `extract_and_normalize_sound_features.py`
* **Input:** Audio files in the `sounds/` directory.
* **Command:**
    ```bash
    python extract_and_normalize_sound_features.py
    ```
* **Outputs:**
    * `sound_features.csv`: Raw, unnormalized sound features.
    * `sound_features_normalized.csv`: Normalized sound features (scaled to [0, 1]).
    * `sound_normalization_params.json`: A JSON file containing the min/max values used for normalization, crucial for denormalizing predictions later.

### 3. Calculate the T Matrix

This script trains the linear transformation model, learning the matrix `T` that maps image features to sound features.

* **Script:** `calculate_T.py`
* **Inputs:**
    * `feature_vectors.csv` (from Step 1)
    * `sound_features_normalized.csv` (from Step 2)
    * `sound_normalization_params.json` (from Step 2)
* **Command:**
    ```bash
    python calculate_T.py
    ```
* **Output:**
    * `T_matrix.pt`: The trained PyTorch tensor representing the linear transformation matrix.

### 4. Predict Sound Features from a New Image

This script demonstrates how to use the trained `T_matrix.pt` to predict the 10 sound feature values for any new, unseen image.

* **Script:** `predict_sound_features.py`
* **Inputs:**
    * `T_matrix.pt` (from Step 3)
    * `sound_normalization_params.json` (from Step 2)
    * A new image file (e.g., `test.png`) placed in the same directory as the script.
* **Command:**
    ```bash
    python predict_sound_features.py
    ```
* **Output:** Prints the 10 predicted, denormalized, and clamped sound feature values directly to your console. These values can then be used by your online sound synthesis algorithms.


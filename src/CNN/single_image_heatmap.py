# single_image_heatmap.py
import torch
import matplotlib.pyplot as plt
import math
import numpy as np
from pathlib import Path
from PIL import Image

# Paths
workspace_dir = Path(__file__).parent / "workspace_example_outputs"
images_dir = Path(__file__).parent / "images"

# Load T matrices
T_path = workspace_dir / "T_matrix.pt"
T_matrices = torch.load(T_path)  # assuming shape [num_images, H, W] or [4, H, W]

# List training images (sorted to match T order)
image_files = sorted(images_dir.glob("*.png"))

# Check number of images and T matrices
num_images = min(len(image_files), len(T_matrices))

# Plot each image and its corresponding T heatmap
fig, axes = plt.subplots(num_images, 2, figsize=(8, 4*num_images))

for i in range(num_images):
    # Load image
    img = Image.open(image_files[i])
    axes[i, 0].imshow(img)
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f"Image: {image_files[i].name}")

    # Plot T heatmap
    heatmap_vector = T_matrices[i].detach().cpu().numpy()
    # reshape to 2D (32 x 64)
    heatmap = heatmap_vector.reshape(32, 64)
    im = axes[i, 1].imshow(heatmap, cmap='viridis')
    axes[i, 1].set_title(f"T Matrix Heatmap {i+1}")
    plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


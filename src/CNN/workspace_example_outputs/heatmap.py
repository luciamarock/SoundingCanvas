import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Load T matrix
T_matrix = torch.load("T_matrix.pt")  # shape: [OUTPUT_DIM, INPUT_DIM]

plt.figure(figsize=(12,6))
sns.heatmap(T_matrix.numpy(), cmap='viridis', cbar=True)
plt.xlabel('Image Feature Index')
plt.ylabel('Sound Feature Index')
plt.title('Linear Transformation Matrix T')
plt.show()


"""
Each row corresponds to a single sound feature (y).

Each column corresponds to a CNN image feature (x).

The heatmap intensity shows how strongly each image feature contributes to each sound feature.
"""

output_index = 0  # e.g., spectral centroid
weights = T_matrix[output_index].numpy()

plt.figure(figsize=(10,4))
plt.bar(range(len(weights)), weights)
plt.xlabel('Image Feature Index')
plt.ylabel(f'Contribution to sound feature {output_index}')
plt.title('Contribution of CNN features to one sound feature')
plt.show()


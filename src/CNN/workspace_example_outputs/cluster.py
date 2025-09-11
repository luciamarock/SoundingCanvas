import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- Load image features ---
image_features_df = pd.read_csv("feature_vectors.csv", index_col=0)

# --- Load T matrix ---
T_matrix = torch.load("T_matrix.pt")

# --- Heatmap ---
plt.figure(figsize=(12,6))
sns.heatmap(T_matrix.numpy(), cmap='viridis', cbar=True)
plt.xlabel('Image Feature Index')
plt.ylabel('Sound Feature Index')
plt.title('Linear Transformation Matrix T')
plt.show()

# --- PCA + color by first sound feature ---
X = image_features_df.values
X_tensor = torch.tensor(X, dtype=torch.float32)  # force float32

y_pred = (X_tensor @ T_matrix.T).numpy()

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c=y_pred[:,0], cmap='viridis')
plt.colorbar(label='Predicted sound feature 0')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Image features projected + colored by sound feature')
plt.show()

"""
Each dot represents one image feature vector that has been transformed into a sound feature vector
PCA 1 and PCA 2 are the two projection axes
 If dots are far apart: these images produce very different sound outputs
"""

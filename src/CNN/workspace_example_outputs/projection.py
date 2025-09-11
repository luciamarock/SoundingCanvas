import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D # REQUIRED for 3D plotting

# Load feature vectors
df = pd.read_csv("feature_vectors.csv", index_col=0)
features = df.values
image_names = df.index.tolist()

# Choose dimensionality reduction: PCA (fast) for 3D
pca = PCA(n_components=3) # Set n_components to 3 for 3D visualization
reduced = pca.fit_transform(features)

# Plot in 3D
fig = plt.figure(figsize=(12, 10)) # Increased figure size for better readability
ax = fig.add_subplot(111, projection='3d') # Add a 3D subplot to the figure

# Loop to plot each point and add text label
for i, name in enumerate(image_names):
    x, y, z = reduced[i] # Get the 3 components (x, y, z coordinates)
    ax.scatter(x, y, z, s=50) # Plot the point. 's' adjusts marker size.

    # Add the text label near the dot
    # Adjusting offsets (e.g., +0.5, +0.5, +0.5) can help position the text
    # relative to the point, but visual overlap is still likely.
    ax.text(x, y, z, name.split('.')[0], fontsize=8, color='black')


plt.title("PCA of Sounding Canvas CNN Features (3D) with Image Names")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3") # New z-axis label for 3D

plt.tight_layout() # Adjusts plot parameters for a tight layout
plt.show() # Display the plot

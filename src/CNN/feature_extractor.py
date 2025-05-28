import os
import torch
import numpy as np
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1]) # remove the final classification layer
model.eval()

# Image preprocessing with augmentation
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Parameters
image_dir = "./images"
num_augmentations = 5  # number of augmentations per image

output_vectors = []
image_names = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        img = Image.open(image_path).convert("RGB")
        
        features_per_image = []

        for _ in range(num_augmentations):
            img_tensor = augment_transform(img).unsqueeze(0)
            with torch.no_grad():
                feature = model(img_tensor).squeeze().numpy()
                features_per_image.append(feature)

        avg_feature = np.mean(features_per_image, axis=0)
        output_vectors.append(avg_feature)
        image_names.append(filename)

# Convert to DataFrame and save
df = pd.DataFrame(output_vectors, index=image_names)
df.to_csv("feature_vectors.csv")
torch.save(torch.tensor(np.array(output_vectors)), "feature_vectors.pt")

print("Saved averaged feature vectors for", len(image_names), "images.")


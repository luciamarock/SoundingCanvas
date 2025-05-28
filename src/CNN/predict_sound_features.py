import torch
import torch.nn as nn
import torchvision.models as models
# Import the specific Weights enum for ResNet50
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont # Added ImageDraw, ImageFont for better dummy image
import numpy as np
import json
import os

# --- Configuration ---
T_MATRIX_PATH = "T_matrix.pt"
NORM_PARAMS_PATH = "sound_normalization_params.json"

# --- 1. Define the FeatureExtractor (Updated for 'weights' argument) ---
class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super(FeatureExtractor, self).__init__()
        if model_name == 'resnet50':
            # Use 'weights' argument instead of 'pretrained'
            # ResNet50_Weights.DEFAULT gets the most up-to-date weights (ImageNet1K_V2 if available, else V1)
            # If pretrained is False, pass None for weights.
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            
            # Remove the final classification layer (the average pooling and fc layer)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:
            raise ValueError(f"Model '{model_name}' not supported for feature extraction.")

        # Ensure the model is in evaluation mode
        self.model.eval()

    def forward(self, x):
        # Forward pass through the truncated model
        features = self.model(x)
        # Flatten the features (from e.g., (batch_size, 2048, 1, 1) to (batch_size, 2048))
        features = features.view(features.size(0), -1)
        return features

# --- 2. Image Preprocessing Transformation ---
# IMPORTANT: This must exactly match the preprocessing used when extracting features
# for our 'feature_vectors.csv' and during the CNN training.
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. Main Prediction Function ---
def predict_sound_features_from_image(image_path):
    """
    Predicts 10 sound feature values for a given image.

    Args:
        image_path (str): The path to the input image file (e.g., "test.png").

    Returns:
        numpy.ndarray: A 1D numpy array containing the 10 predicted,
                       denormalized, and clamped sound feature values.
                       Returns None if any required file is missing or an error occurs.
    """
    # --- Check for required files ---
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at: {image_path}")
        return None
    if not os.path.exists(T_MATRIX_PATH):
        print(f"Error: T matrix not found at: {T_MATRIX_PATH}. Please ensure 'calculate_T.py' has been run successfully.")
        return None
    if not os.path.exists(NORM_PARAMS_PATH):
        print(f"Error: Normalization parameters not found at: {NORM_PARAMS_PATH}. Please ensure 'extract_and_normalize_sound_features.py' has been run successfully.")
        return None

    try:
        # Load the trained T matrix
        loaded_T_matrix = torch.load(T_MATRIX_PATH)

        # Load normalization parameters
        with open(NORM_PARAMS_PATH, 'r') as f:
            norm_params = json.load(f)
        original_sound_min_values_tensor = torch.tensor(norm_params['min_values'], dtype=torch.float32)
        original_sound_max_values_tensor = torch.tensor(norm_params['max_values'], dtype=torch.float32)

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = image_transform(image).unsqueeze(0) # Add batch dimension (1, C, H, W)

        # Initialize the image feature extractor
        feature_extractor = FeatureExtractor(model_name='resnet50', pretrained=True) # pretrained=True still controls if weights are loaded

        # Move model and input to the correct device (CPU/GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor.to(device)
        image_tensor = image_tensor.to(device)
        loaded_T_matrix = loaded_T_matrix.to(device)
        original_sound_min_values_tensor = original_sound_min_values_tensor.to(device)
        original_sound_max_values_tensor = original_sound_max_values_tensor.to(device)


        # Extract image features
        with torch.no_grad(): # No need to calculate gradients for inference
            image_features = feature_extractor(image_tensor)

        # Perform the linear transformation
        predicted_sound_features_normalized = image_features @ loaded_T_matrix.T

        # Denormalize
        y_pred_original_scale = predicted_sound_features_normalized * \
                                (original_sound_max_values_tensor - original_sound_min_values_tensor) + \
                                original_sound_min_values_tensor

        # Clamp to ensure values stay within physical/perceptual bounds
        y_pred_clamped = torch.max(original_sound_min_values_tensor,
                                   torch.min(original_sound_max_values_tensor, y_pred_original_scale))

        return y_pred_clamped.squeeze(0).cpu().numpy() # Remove batch dim and convert to numpy array on CPU

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

# --- 4. Example Usage when running the script directly ---
if __name__ == "__main__":
    # path to our test image
    input_image_file = "test.png" # Make sure this file exists in the same directory!

    # --- Optional: Create a dummy image if 'test.png' doesn't exist for initial testing ---
    if not os.path.exists(input_image_file):
        print(f"'{input_image_file}' not found. Creating a dummy image for demonstration purposes.")
        try:
            # Try to use a default font or handle error if not found
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24) # Common font on Linux
            except IOError:
                font = ImageFont.load_default() # Fallback to default PIL font
            
            dummy_image = Image.new('RGB', (224, 224), color = 'blue')
            d = ImageDraw.Draw(dummy_image)
            d.text((20,90), "Test Image", fill=(255,255,255), font=font)
            dummy_image.save(input_image_file)
            print(f"Dummy image '{input_image_file}' created.")
        except Exception as e:
            print(f"Error creating dummy image: {e}. Please ensure Pillow is installed correctly and try installing a font if needed.")
            print(f"Please manually create a '{input_image_file}' image in this directory for testing.")
            exit()


    print(f"Attempting to predict sound features for: {input_image_file}")
    predicted_features = predict_sound_features_from_image(input_image_file)

    if predicted_features is not None:
        print("\n--- Predicted Sound Feature Values ---")
        # Here's a simple display.
        feature_names = [ # These should match the order from our feature_extraction script
            'spectral_centroid_mean', 'spectral_rolloff_85_mean', 'spectral_flux_mean',
            'spectral_bandwidth_mean', 'spectral_flatness_mean', 'zero_crossing_rate_mean',
            'rms_db_mean', 'tempo_bpm', 'attack_time_s', 'decay_time_s'
        ]
        for i, val in enumerate(predicted_features):
            print(f"{feature_names[i]}: {val:.4f}")
        print("\nThese are the 10 values you can use for sound synthesis or manipulation.")

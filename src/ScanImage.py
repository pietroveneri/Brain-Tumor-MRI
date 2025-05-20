import os
from PIL import Image as PILImage # type: ignore
import numpy as np # type: ignore
import matplotlib # type: ignore
matplotlib.use('TkAgg') # Add this line for WSL compatibility
import matplotlib.pyplot as plt # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tf_keras_vis.utils.scores import CategoricalScore # type: ignore
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus # type: ignore
import cv2 # type: ignore

# Load the model directly
model = load_model('modelResNet50_Tumor.keras')
# Custom model modifier for ResNet50 model
def _resnet50_replace_to_linear(model_instance_to_modify: tf.keras.Model) -> tf.keras.Model:
    """
    Modifies a ResNet50 model's last layer activation to linear.
    This function is intended to be used as a model_modifier in tf-keras-vis.
    """
    if hasattr(model_instance_to_modify.layers[-1], 'activation'):
        model_instance_to_modify.layers[-1].activation = tf.keras.activations.linear
    else:
        print("Warning: Last layer of the model does not have an 'activation' attribute.")
    return model_instance_to_modify

def preprocess_image(img_path):
    # Load and resize image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    
    # ResNet50 preprocessing
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def show_gradcam(img_path, model_for_prediction, class_labels):
    # --- Preprocess and Predict ---
    img_array = preprocess_image(img_path)
    prediction = model_for_prediction.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]

    # --- True Class Extraction (from folder name) ---
    true_class_raw = img_path.split("/")[-2].lower()
    if 'notumor' in true_class_raw:
        true_class = 'notumor'
    elif 'glioma' in true_class_raw:
        true_class = 'glioma'
    elif 'meningioma' in true_class_raw:
        true_class = 'meningioma'
    elif 'pituitary' in true_class_raw:
        true_class = 'pituitary'
    else:
        true_class = 'unknown'

    # --- Load Original Image ---
    original_img = PILImage.open(img_path)
    original_img = original_img.resize((224, 224))
    if original_img.mode != 'RGB': # Ensure it's RGB
        original_img = original_img.convert('RGB')
    original_img_array = np.array(original_img)
    
    # --- GradCAM Calculation ---
    gradcam = GradcamPlusPlus(model_for_prediction,
                             model_modifier=_resnet50_replace_to_linear,
                             clone=True)
    
    # Create score function for the predicted class
    score = CategoricalScore(predicted_class_index)
    
    # Generate GradCAM - using the last conv layer of ResNet50
    # Try different layer names if this one doesn't work
    try:
        cam = gradcam(score,
                     img_array,
                     penultimate_layer='conv5_block3_out')  # Last conv layer in ResNet50
    except ValueError:
        # Fallback to alternative layer names
        try:
            cam = gradcam(score,
                         img_array,
                         penultimate_layer='conv5_block3_3_conv')  # Alternative layer name
        except ValueError:
            print("Warning: Could not find the expected layer. Using default layer.")
            cam = gradcam(score, img_array)
    
    heatmap = np.maximum(cam[0], 0)
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    
    # Resize and colorize heatmap
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create the overlay
    alpha = 0.5
    overlay = cv2.addWeighted(original_img_array, 1-alpha, heatmap_colored, alpha, 0)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original_img_array)
    ax1.set_title(f'Original Image\nTrue Class: {true_class}')
    ax1.axis('off')
    
    ax2.imshow(overlay)
    ax2.set_title(f'GradCAM Overlay\nPredicted: {predicted_class}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print prediction probabilities
    print("\nPrediction Probabilities:")
    for i, (label, prob) in enumerate(zip(class_labels, prediction[0])):
        print(f"{label}: {prob:.4f}" + (" (predicted)" if i == predicted_class_index else ""))

# Test the visualization
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Test with different classes
image_paths = [
    "brain-tumor-mri-dataset/Testing/glioma/Te-glTr_0002.jpg",
    "brain-tumor-mri-dataset/Testing/notumor/Te-noTr_0003.jpg",
    "brain-tumor-mri-dataset/Testing/meningioma/Te-meTr_0007.jpg",
    "brain-tumor-mri-dataset/Testing/pituitary/Te-piTr_0001.jpg",
    "brain-tumor-mri-dataset/Testing/glioma/Te-gl_0266.jpg",
    "brain-tumor-mri-dataset/Testing/notumor/Te-no_0345.jpg",
    "brain-tumor-mri-dataset/Testing/meningioma/Te-me_0273.jpg",
    "brain-tumor-mri-dataset/Testing/pituitary/Te-pi_0236.jpg"
]

for image_path in image_paths:
    print(f"\nProcessing: {image_path}")
    show_gradcam(image_path, model, class_labels)
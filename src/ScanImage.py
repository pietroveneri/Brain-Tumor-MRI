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
model = load_model('modelResNet50.keras')
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
    # Load image
    img = PILImage.open(img_path)
    
    # Convert grayscale to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize((224, 224))
    
    # Convert to array
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
    confidence = prediction[0][predicted_class_index]

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

    original_img = PILImage.open(img_path)
    
    # Convert grayscale to RGB if needed (before resizing)
    if original_img.mode != 'RGB':
        original_img = original_img.convert('RGB')
        
    # Resize the image
    original_img = original_img.resize((224, 224))
    original_img_array = np.array(original_img)
    
    gradcam = GradcamPlusPlus(model_for_prediction,
                             model_modifier=_resnet50_replace_to_linear,
                             clone=True)
    
    # Create score function for the predicted class
    score = CategoricalScore(predicted_class_index)
    
    # Generate GradCAM - using the last conv layer of ResNet50
    try:
        cam = gradcam(score,
                     img_array,
                     penultimate_layer='conv5_block3_out')
    except ValueError:
        try:
            cam = gradcam(score,
                         img_array,
                         penultimate_layer='conv5_block3_3_conv')
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
    
    # Create the overlay with improved transparency
    alpha = 0.6  
    overlay = cv2.addWeighted(original_img_array, 1-alpha, heatmap_colored, alpha, 0)
    
    # Plot results with improved layout
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2)
    
    # Original Image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(original_img_array)
    ax1.set_title(f'Original Image\nTrue Class: {true_class}', fontsize=12, pad=10)
    ax1.axis('off')
    
    # GradCAM Overlay
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(overlay)
    ax2.set_title(f'GradCAM Overlay\nPredicted: {predicted_class}\nConfidence: {confidence:.2%}', 
                 fontsize=12, pad=10)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed prediction probabilities
    print("\nPrediction Probabilities:")
    print("-" * 40)
    for i, (label, prob) in enumerate(zip(class_labels, prediction[0])):
        confidence_str = f"{prob:.2%}"
        if i == predicted_class_index:
            print(f"{label:12} {confidence_str:>8} (predicted)")
        else:
            print(f"{label:12} {confidence_str:>8}")
    print("-" * 40)
    print(f"Model Confidence: {confidence:.2%}")
    print(f"Prediction Correct: {predicted_class == true_class}")

# Test the visualization
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Test with different classes
image_paths = [
    # Add your image paths for Testing!
]

for image_path in image_paths:
    print(f"\nProcessing: {image_path}")
    show_gradcam(image_path, model, class_labels)

import os
import time
from PIL import Image as PILImage # type: ignore
import numpy as np # type: ignore
import matplotlib # type: ignore
# Try different backends for WSL compatibility - prioritize interactive backends
try:
    matplotlib.use('TkAgg')  # Try interactive backend first
    import matplotlib.pyplot as plt # type: ignore
    print("Using TkAgg backend for matplotlib")
except:
    try:
        matplotlib.use('Qt5Agg')  # Try another interactive backend
        import matplotlib.pyplot as plt # type: ignore
        print("Using Qt5Agg backend for matplotlib")
    except:
        try:
            matplotlib.use('Agg')  # Fallback to non-interactive
            import matplotlib.pyplot as plt # type: ignore
            print("Using Agg backend for matplotlib (non-interactive)")
        except:
            import matplotlib.pyplot as plt # type: ignore
            print("Using default matplotlib backend")
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tf_keras_vis.utils.scores import CategoricalScore # type: ignore
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus # type: ignore
from tf_keras_vis.gradcam import Gradcam # type: ignore
import cv2 # type: ignore

# Load the model directly
model = load_model('modelResNet50_44.keras')
# Custom model modifier for ResNet50 model
def _resnet50_replace_to_linear(model_instance_to_modify: tf.keras.Model) -> tf.keras.Model:
    """
    Modifies a ResNet50 model's last layer activation to linear.
    This function is intended to be used as a model_modifier in tf-keras-vis.
    """
    print(f"Model modifier: Original last layer activation: {model_instance_to_modify.layers[-1].activation}")
    if hasattr(model_instance_to_modify.layers[-1], 'activation'):
        model_instance_to_modify.layers[-1].activation = tf.keras.activations.linear
        print(f"Model modifier: Changed to linear activation")
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
    
    # Measure inference time
    start_time = time.time()
    prediction = model_for_prediction.predict(img_array)
    end_time = time.time()
    inference_time = end_time - start_time
    
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
    
    # Create score function for the predicted class
    score = CategoricalScore(predicted_class_index)
    
    # Try different GradCAM implementations and layers
    layer_names_to_try = [
        'conv5_block3_out',
        'conv5_block3_3_conv', 
        'conv4_block6_out',
        'conv4_block6_2_conv',
        'conv3_block4_out',
        'conv2_block3_out'
    ]
    
    cam = None
    
    # Try GradCAM++ first
    gradcam_plus = GradcamPlusPlus(model_for_prediction,
                                  model_modifier=_resnet50_replace_to_linear,
                                  clone=True)
    
    for layer_name in layer_names_to_try:
        try:
            cam = gradcam_plus(score, img_array, penultimate_layer=layer_name)
            if np.max(cam[0]) > 0:  # Check if we got meaningful gradients
                break
        except ValueError:
            continue
    
    # If GradCAM++ didn't work, try regular GradCAM
    if cam is None or np.max(cam[0]) == 0:
        gradcam_regular = Gradcam(model_for_prediction,
                                 model_modifier=_resnet50_replace_to_linear,
                                 clone=True)
        
        for layer_name in layer_names_to_try:
            try:
                cam = gradcam_regular(score, img_array, penultimate_layer=layer_name)
                if np.max(cam[0]) > 0:  # Check if we got meaningful gradients
                    break
            except ValueError:
                continue
    
    if cam is None or np.max(cam[0]) == 0:
        try:
            cam = gradcam_regular(score, img_array)
        except Exception:
            return
    
    heatmap = np.maximum(cam[0], 0)
    
    # Normalize heatmap
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    else:
        heatmap = np.ones_like(heatmap) * 0.1  # Create a minimal heatmap
    
    # Resize and colorize heatmap
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create the overlay with improved transparency
    alpha = 0.6  
    overlay = cv2.addWeighted(original_img_array, 1-alpha, heatmap_colored, alpha, 0)
    
    # Plot results - only original image and overlay, no titles or spacing
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2)
    
    # Original Image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(original_img_array)
    ax1.axis('off')
    
    # GradCAM Overlay
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(overlay)
    ax2.axis('off')
    
    # Remove all spacing and margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # Display the plot
    try:
        plt.show(block=False)  # Non-blocking display
        plt.pause(0.1)  # Brief pause to ensure display
        print("GradCAM plot displayed successfully")
        
        # Wait for user input to continue
        input("Press Enter to close the plot and continue...")
        
    except Exception as e:
        print(f"Could not display plot interactively: {e}")
        print("Continuing...")
    
    plt.close()  # Close the figure to free memory
    
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
    print(f"Inference Time: {inference_time:.4f} seconds ({inference_time*1000:.2f} ms)")

def measure_inference_time(img_path, model_for_prediction, num_runs=5):
    """
    Measure inference time for a single image with multiple runs for more accurate timing.
    
    Args:
        img_path: Path to the image
        model_for_prediction: The model to use for prediction
        num_runs: Number of runs to average the timing (default: 5)
    
    Returns:
        dict: Contains average inference time and other statistics
    """
    img_array = preprocess_image(img_path)
    
    # Warm up the model
    _ = model_for_prediction.predict(img_array, verbose=0)
    
    # Measure inference time over multiple runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        prediction = model_for_prediction.predict(img_array, verbose=0)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    return {
        'average_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'all_times': times
    }

# Main execution
if __name__ == "__main__":
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Default test image - you can change this path or add more images
    image_paths = [
    ]
    
    print("Brain Tumor MRI Classification and GradCAM Visualization")
    print("=" * 60)
    
    for image_path in image_paths:
        if os.path.exists(image_path):
            print(f"\nProcessing: {image_path}")
            # Display image and statistics without saving
            show_gradcam(image_path, model, class_labels)
            
            # Additional detailed timing analysis
            print(f"\nDetailed timing analysis for: {image_path}")
            timing_stats = measure_inference_time(image_path, model, num_runs=10)
            print(f"Average inference time: {timing_stats['average_time']:.4f} seconds ({timing_stats['average_time']*1000:.2f} ms)")
            print(f"Standard deviation: {timing_stats['std_time']:.4f} seconds ({timing_stats['std_time']*1000:.2f} ms)")
            print(f"Min time: {timing_stats['min_time']:.4f} seconds ({timing_stats['min_time']*1000:.2f} ms)")
            print(f"Max time: {timing_stats['max_time']:.4f} seconds ({timing_stats['max_time']*1000:.2f} ms)")
        else:
            print(f"Error: Image file not found: {image_path}")
            print("Please check the file path and try again.")
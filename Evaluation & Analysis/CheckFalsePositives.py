import os
from PIL import Image as PILImage  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore


def preprocess_image(img_path):
    img = PILImage.open(img_path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def main():
    model = load_model('modelResNet50.keras')
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Check both Testing and Training sets
    dataset_dirs = [
        os.path.join('Dataset', 'Testing', 'notumor'),
        os.path.join('Dataset', 'Training', 'notumor')
    ]

    false_positives = []
    total_processed = 0

    for dataset_dir in dataset_dirs:
        if not os.path.isdir(dataset_dir):
            print(f"Warning: Directory not found: {dataset_dir}")
            continue

        image_filenames = [
            f for f in os.listdir(dataset_dir)
            if os.path.isfile(os.path.join(dataset_dir, f)) and f.lower().endswith(('.jpg', '.jpeg'))
        ]

        print(f"Processing {len(image_filenames)} images from {dataset_dir}...")

        for filename in sorted(image_filenames):
            img_path = os.path.join(dataset_dir, filename)
            img_array = preprocess_image(img_path)
            prediction = model.predict(img_array, verbose=0)
            predicted_class_index = int(np.argmax(prediction))
            predicted_class = class_labels[predicted_class_index]

            if predicted_class != 'notumor':
                # Include directory info to distinguish between sets
                relative_path = os.path.join(os.path.basename(os.path.dirname(dataset_dir)), 
                                           os.path.basename(dataset_dir), filename)
                false_positives.append(relative_path)

        total_processed += len(image_filenames)

    output_path = 'false_positives.txt'
    with open(output_path, 'w') as f:
        for name in false_positives:
            f.write(f"{name}\n")

    print(f"\nProcessed {total_processed} images total.")
    print(f"False positives (notumor predicted as tumor): {len(false_positives)}")
    print(f"Saved list to {output_path}")


if __name__ == '__main__':
    main()



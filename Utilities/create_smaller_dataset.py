import os
import shutil
import random
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedKFold
import json

# Set random seeds for reproducibility
SEED = 44
random.seed(SEED)
np.random.seed(SEED)

def create_smaller_dataset(source_dir, target_dir, max_per_class=500):
    """
    Create a smaller, balanced dataset by sampling equal numbers from each class.
    
    Args:
        source_dir: Path to the original training dataset
        target_dir: Path where the smaller dataset will be created
        max_per_class: Maximum number of images per class
    """
    
    # Create target directory structure
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all classes
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    classes.sort()
    
    print(f"Found classes: {classes}")
    
    # Count images in each class and determine sampling size
    class_counts = {}
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(image_files)
        print(f"{class_name}: {len(image_files)} images")
    
    # Determine the minimum number of images across all classes
    min_images = min(class_counts.values())
    target_per_class = min(min_images, max_per_class)
    
    print(f"\nTarget images per class: {target_per_class}")
    
    # Sample and copy images
    total_copied = 0
    dataset_info = {}
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        target_class_path = os.path.join(target_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Randomly sample images
        if len(image_files) > target_per_class:
            selected_images = random.sample(image_files, target_per_class)
        else:
            selected_images = image_files
        
        # Copy selected images
        copied_count = 0
        for img_file in selected_images:
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(target_class_path, img_file)
            
            try:
                # Verify image is valid before copying
                with Image.open(src_path) as img:
                    img.verify()
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
                continue
        
        total_copied += copied_count
        dataset_info[class_name] = copied_count
        print(f"Copied {copied_count} images from {class_name}")
    
    print(f"\nTotal images copied: {total_copied}")
    
    # Save dataset information
    info_path = os.path.join(target_dir, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump({
            'total_images': total_copied,
            'images_per_class': dataset_info,
            'classes': classes,
            'source_dataset': source_dir,
            'creation_date': str(np.datetime64('now'))
        }, f, indent=2)
    
    return dataset_info

def create_kfold_splits(dataset_dir, n_splits=5, output_file="kfold_splits.json"):
    """
    Create k-fold cross-validation splits for the dataset.
    
    Args:
        dataset_dir: Path to the dataset
        n_splits: Number of folds
        output_file: Output file to save the splits
    """
    
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    classes.sort()
    
    # Create class mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    
    for class_name in classes:
        class_path = os.path.join(dataset_dir, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            image_paths.append(os.path.join(class_name, img_file))
            labels.append(class_to_idx[class_name])
    
    # Create stratified k-fold splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        fold_data = {
            'fold': fold_idx + 1,
            'train': {
                'paths': [image_paths[i] for i in train_idx],
                'labels': [labels[i] for i in train_idx]
            },
            'validation': {
                'paths': [image_paths[i] for i in val_idx],
                'labels': [labels[i] for i in val_idx]
            }
        }
        
        # Count samples per class in each split
        train_class_counts = {}
        val_class_counts = {}
        
        for label in fold_data['train']['labels']:
            class_name = classes[label]
            train_class_counts[class_name] = train_class_counts.get(class_name, 0) + 1
        
        for label in fold_data['validation']['labels']:
            class_name = classes[label]
            val_class_counts[class_name] = val_class_counts.get(class_name, 0) + 1
        
        fold_data['train']['class_counts'] = train_class_counts
        fold_data['validation']['class_counts'] = val_class_counts
        
        folds.append(fold_data)
        
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Validation: {len(val_idx)} samples")
        print(f"  Train class distribution: {train_class_counts}")
        print(f"  Val class distribution: {val_class_counts}")
        print()
    
    # Save folds information
    folds_info = {
        'n_splits': n_splits,
        'classes': classes,
        'class_to_idx': class_to_idx,
        'total_samples': len(image_paths),
        'folds': folds,
        'creation_date': str(np.datetime64('now'))
    }
    
    with open(output_file, 'w') as f:
        json.dump(folds_info, f, indent=2)
    
    print(f"K-fold splits saved to {output_file}")
    return folds_info

if __name__ == "__main__":
    # Configuration
    source_dataset = "Dataset/Training"
    smaller_dataset = "SmallerDataset"
    max_images_per_class = 500  # Adjust this value as needed
    
    print("Creating smaller, balanced dataset...")
    print("=" * 50)
    
    # Create smaller dataset
    dataset_info = create_smaller_dataset(source_dataset, smaller_dataset, max_images_per_class)
    
    print("\n" + "=" * 50)
    print("Creating 5-fold cross-validation splits...")
    print("=" * 50)
    
    # Create k-fold splits
    folds_info = create_kfold_splits(smaller_dataset, n_splits=5)
    
    print("\nDataset preparation completed!")
    print(f"Smaller dataset created at: {smaller_dataset}")
    print(f"K-fold splits saved to: kfold_splits.json") 

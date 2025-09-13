import os
import pandas as pd
from collections import Counter

def count_images_in_dataset(base_path):
    class_counts = Counter()
    
    for root, dirs, files in os.walk(base_path):
        # Get the class name from the directory path
        class_name = os.path.basename(root)
        
        # Skip the base directories themselves
        if class_name in ['Dataset', 'Training', 'Testing', 'NewDataset']:
            continue
            
        # Count only image files
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        if image_files:
            class_counts[class_name] += len(image_files)
    
    return class_counts

# Count images in main dataset
print("Counting images in Dataset...")
dataset_counts = count_images_in_dataset("Dataset")

# Also check NewDataset if it exists
newdataset_counts = None
if os.path.exists("NewDataset"):
    print("Counting images in NewDataset...")
    newdataset_counts = count_images_in_dataset("NewDataset")

# Display results
print("\nSlices per class in Dataset:")
for class_name, count in dataset_counts.items():
    print(f"{class_name}: {count} slices")

if newdataset_counts:
    print("\nSlices per class in NewDataset:")
    for class_name, count in newdataset_counts.items():
        print(f"{class_name}: {count} slices")

# Create a summary DataFrame
results = pd.DataFrame([dataset_counts])
if newdataset_counts:
    results = pd.concat([results, pd.DataFrame([newdataset_counts])], keys=['Dataset', 'NewDataset'])
else:
    results.index = ['Dataset']

total_images = sum(dataset_counts.values())
print(f"\nTotal number of slices: {total_images}")

# Calculate percentages
if total_images > 0:
    print("\nClass distribution:")
    for class_name, count in dataset_counts.items():
        percentage = (count / total_images) * 100
        print(f"{class_name}: {percentage:.2f}%") 
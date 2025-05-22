import hashlib
import os

# Define constants based on project structure
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']


def compute_hash(file):
    hasher = hashlib.md5()
    with open(file, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def list_files(hash_dict):
    for data_type in ['Training', 'Testing']:
        for label in LABELS:
            folder_path = os.path.join(PROJECT_DIR, 'Dataset', data_type, label)
            if not os.path.exists(folder_path):
                print(f"Skipping non-existent path: {folder_path}")
                continue
                
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        file_path = os.path.join(root, file)
                        file_hash = compute_hash(file_path)
                        if file_hash in hash_dict:
                            hash_dict[file_hash].append(file_path)
                        else:
                            hash_dict[file_hash] = [file_path]


def remove_duplicates(hash_dict):
    duplicate_count = 0
    for hash_value, file_paths in hash_dict.items():
        if len(file_paths) > 1:
            print(f"Found duplicate with hash: {hash_value}")
            print(f"Keeping: {file_paths[0]}")
            for file_path in file_paths[1:]:
                print(f"Removing: {file_path}")
                os.remove(file_path)
                duplicate_count += 1
    print(f"Number of duplicates removed: {duplicate_count}")


def count_images():
    """Count the number of images in each category folder"""
    counts = {}
    total_images = 0
    
    for data_type in ['Training', 'Testing']:
        counts[data_type] = {}
        type_total = 0
        
        for label in LABELS:
            folder_path = os.path.join(PROJECT_DIR, 'Dataset', data_type, label)
            if not os.path.exists(folder_path):
                print(f"Skipping non-existent path: {folder_path}")
                counts[data_type][label] = 0
                continue
                
            image_count = 0
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_count += 1
            
            counts[data_type][label] = image_count
            type_total += image_count
            print(f"{data_type}/{label}: {image_count} images")
        
        print(f"Total {data_type} images: {type_total}")
        total_images += type_total
    
    print(f"Total images in dataset: {total_images}")
    return counts


if __name__ == "__main__":
    action = input("Choose action (1: Remove duplicates, 2: Count images): ")
    
    if action == "1":
        print("Searching for duplicate images in Dataset/Training and Dataset/Testing...")
        hash_dict = {}
        list_files(hash_dict)
        remove_duplicates(hash_dict)
        print("Duplicate removal completed.")
    elif action == "2":
        print("Counting images in each category...")
        count_images()
    else:
        print("Invalid action. Please choose 1 or 2.")

# Interpretable Brain Tumor Classification via ResNet50 and Grad-CAM++

## 1. Description

This project implements a deep learning pipeline to classify brain tumors from axial T1-weighted contrast-enhanced MRI slices using a fine-tuned ResNet50 model. Grad-CAM++ is used for post hoc interpretability, visualizing model attention on relevant tumor regions.

## 2. Dataset Information

The model was trained on a combined dataset from two publicly available sources:
- Kaggle Brain Tumor Dataset (DOI: 10.34740/kaggle/dsv/2645886)
- Figshare Brain Tumor Dataset (DOI: 10.6084/m9.figshare.1512427.v8)

The dataset contains 9,790 labeled slices divided into four classes: glioma, meningioma, pituitary tumor, and no tumor. All images were resized to 224x224 pixels.

## 3. Code Information

Two primary scripts are included:
- `ModelTraining.py`: trains a ResNet50 classifier with data augmentation, mixed precision, and class balancing.
- `ScanImage.py`: generates Grad-CAM++ visualizations from trained models.

## 4. Usage Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/pietroveneri/Brain-Tumor-MRI.git
   cd Brain-Tumor-MRI
2. Organise the dataset 

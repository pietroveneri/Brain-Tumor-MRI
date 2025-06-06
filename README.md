
# Interpretable Brain Tumor Classification via ResNet50 and Grad-CAM++

## Description

This project implements a deep learning pipeline for classifying brain tumors from axial T1-weighted contrast-enhanced MRI slices using a fine-tuned ResNet50 model. Grad-CAM++ is utilized to visually interpret model predictions, highlighting relevant tumor regions.

## Dataset Information

The dataset combines two publicly available sources:
- Kaggle Brain Tumor Dataset ([DOI: 10.34740/kaggle/dsv/2645886](https://doi.org/10.34740/kaggle/dsv/2645886))
- Figshare Brain Tumor Dataset ([DOI: 10.6084/m9.figshare.1512427.v8](https://doi.org/10.6084/m9.figshare.1512427.v8))

The combined dataset includes 9,790 images labeled as glioma, meningioma, pituitary tumor, or no tumor. Images are resized to 224x224 pixels.

## Code Information

Scripts included:
- `ModelTraining.py`: Training script using ResNet50 with data augmentation, mixed precision, and regularization.
- `ScanImage.py`: Generates Grad-CAM++ visualizations for predictions.

## Usage Instructions

1. Clone this repository:
```bash
git clone https://github.com/pietroveneri/Brain-Tumor-MRI.git
cd Brain-Tumor-MRI
```

2. Arrange dataset in the `Dataset/` directory.

3. Train the model:
```bash
python ModelTraining.py
```

4. Generate Grad-CAM++ visualizations:
```bash
python ScanImage.py
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Main dependencies include:
- Python >= 3.8
- TensorFlow >= 2.11
- numpy, matplotlib, Pillow, opencv-python
- tf-keras-vis

## Methodology

- **Preprocessing**: Image resizing, grayscale conversion, and balancing class representation.
- **Model**: Fine-tuned ResNet50 initialized from ImageNet weights.
- **Training**: Multi-phase progressive training strategy, including dropout, Gaussian noise, and label smoothing.
- **Evaluation**: Stratified 80/20 train-test split.

## Citations

Please cite this project if used:
```
Pietro Veneri. Interpretable classification of brain tumors from MRI using fine-tuned ResNet50 and Grad-CAM++ visualization. 
```

## License

This project is licensed under the MIT License.

## Contribution Guidelines

This is a single-author project. Issues, feedback, and forks are welcome.

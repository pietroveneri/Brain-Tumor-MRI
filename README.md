# Brain Tumor MRI Classification

A comprehensive deep learning project for brain tumor classification from MRI images using transfer learning with VGG16 and ResNet50 architectures. This project includes advanced training strategies, cross-validation evaluation, and GradCAM visualization.

## Project Overview

This project classifies brain tumors from MRI images into 4 categories:
- **Glioma** - A type of tumor that starts in the glial cells
- **Meningioma** - A tumor that arises from the meninges
- **Pituitary** - A tumor in the pituitary gland
- **No Tumor** - Normal brain tissue

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
# Create balanced dataset (500 images per class)
python create_smaller_dataset.py

# Clean damaged images
python clean_dataset.py
```

### 3. Train Models
```bash
# Train VGG16 model
python ModelTrainingVGG16.py

# Train ResNet50 model
python ModelTraining.py
```

### 4. Run Cross-Validation
```bash
# Run 5-fold cross-validation
python run_cv.py
```

### 5. Analyze Images
```bash
# Visualize predictions with GradCAM
python ScanImage.py
```

## Project Structure

```
BrainTumorMRI/

├── Training Scripts
│   ├── ModelTrainingVGG16.py              # VGG16 training with transfer learning
│   ├── ModelTrainingResNet50.py           # ResNet50 training with fine-tuningn
├── Evaluation & Analysis
│   ├── cross_validation.py                # 5-fold cross-validation system
│   ├── run_cv.py                          # Cross-validation runner
│   ├── cv_config.py                       # Cross-validation configuration
│   ├── CheckFalsePositives.py             # False positive analysis
│   └── ScanImage.py                       # GradCAM visualization
├── Utilities
│   ├── create_smaller_dataset.py          # Dataset balancing for CV
│   ├── clean_dataset.py                   # Dataset cleaning from duplicates
│   ├── count_slices.py                    # Dataset analysis


```

## Technical Details

### Models
- **VGG16**: Transfer learning with custom head
- **ResNet50**: Multi-stage fine-tuning approach
  - Stage 1: Head training (frozen backbone)
  - Stage 2: Partial fine-tuning (last few layers)
  - Stage 3: Full fine-tuning (all layers)

### Training Features
- **Data Augmentation**: Zoom, rotation, brightness, shifts
- **Mixed Precision**: FP16 training for efficiency
- **Class Balancing**: Automatic class weight computation
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rates

### Evaluation
- **5-Fold Cross-Validation**: Stratified sampling
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC AUC
- **Visualization**: Confusion matrices, performance plots
- **Error Analysis**: False positive identification

## Usage Examples

### Training a New Model
```python
# VGG16 Training
python ModelTrainingVGG16.py

# ResNet50 Training  
python ModelTrainingResNet50.py
```

### Cross-Validation
```python
# Run full cross-validation
python run_cv.py

# Or run directly
python cross_validation.py
```

### Image Analysis
```python
# Analyze specific images with GradCAM
python ScanImage.py
```

## Key Features

- **High Performance**: 96%+ accuracy on brain tumor classification
- **Robust Evaluation**: Comprehensive cross-validation system
- **Visualization**: GradCAM++ heatmaps for model interpretability
- **Error Analysis**: Detailed false positive analysis
- **Reproducibility**: Fixed random seeds and version control
- **Scalability**: Efficient training with mixed precision

## Requirements

- Python 3.7+
- TensorFlow 2.15.0
- OpenCV 4.9.0
- Scikit-learn 1.3.2
- Matplotlib 3.8.2
- Seaborn 0.13.0
- tf-keras-vis 0.8.6

## Dataset

This project uses the [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) with the following structure:

```
Dataset/
├── Training/
│   ├── glioma/        # Glioma tumor images
│   ├── meningioma/    # Meningioma tumor images
│   ├── notumor/       # Normal brain images
│   └── pituitary/     # Pituitary tumor images
└── Testing/
    └── [same structure as Training]
```

## Model Interpretability

The project includes GradCAM++ visualization to understand what the model focuses on:

- **Heatmap Generation**: Shows attention regions
- **Class-Specific Analysis**: Different heatmaps for each prediction
- **Overlay Visualization**: Combines original image with attention map

## Results Analysis

### Cross-Validation Summary
- **Consistent Performance**: Low standard deviation across folds
- **High Confidence**: 99.8% ROC AUC indicates excellent discrimination
- **Balanced Performance**: Similar metrics across all classes

### Model Comparison
- **ResNet50**: Best overall performance with 96.2% accuracy
- **VGG16**: Good performance with different architectural benefits
- **Ensemble Potential**: Multiple models for improved robustness

## Troubleshooting

### Common Issues
1. **CUDA/GPU Errors**: Models automatically fall back to CPU
2. **Memory Issues**: Use batch prediction mode in config
3. **Dataset Not Found**: Run `create_smaller_dataset.py` first
4. **Model Loading**: Ensure .keras files are in the correct directory

### Performance Tips
- Use GPU for faster training
- Enable mixed precision for memory efficiency
- Adjust batch size based on available memory
- Use data augmentation for better generalization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please open an issue in the repository.

---


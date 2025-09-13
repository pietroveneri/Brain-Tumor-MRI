# Cross-Validation Configuration File
# Modify these parameters as needed

# Dataset and model paths
DATASET_PATH = "SmallerDataset"
MODELS_DIR = "."
IMAGE_SIZE = (224, 224)

# Cross-validation parameters
N_SPLITS = 5
RANDOM_SEED = 43

# Model files to evaluate (relative to MODELS_DIR)
MODEL_FILES = {
    # 'resnet50': 'modelResNet50.keras',
    'vgg16': 'modelVGG16_44.keras'
}

# Output file names
OUTPUT_FILES = {
    'summary_json': 'cv_summary_statistics.json',
    'results_csv': 'cv_results.csv',
    'summary_plot': 'cv_results_summary.png',
    'confusion_matrix_prefix': 'confusion_matrix'
}

# Visualization settings
PLOT_SETTINGS = {
    'figure_size': (15, 12),
    'dpi': 300,
    'color_palette': 'husl',
    'grid_alpha': 0.3
}

# Performance settings
PERFORMANCE = {
    'batch_prediction': False,  # Set to True for faster processing (may use more memory)
    'verbose_predictions': False,  # Set to False to reduce output during prediction
    'save_predictions': True  # Save detailed prediction results
}

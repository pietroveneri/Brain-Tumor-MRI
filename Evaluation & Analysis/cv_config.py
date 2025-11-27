# Cross-Validation Configuration File
# Modify these parameters as needed

# Dataset and model paths
DATASET_PATH = "Dataset"
MODELS_DIR = "."
IMAGE_SIZE = (224, 224)
CHECKPOINT_DIR = "cv_checkpoints"

# Cross-validation parameters
N_SPLITS = 5
RANDOM_SEED = 43
TRAIN_VALIDATION_SPLIT = 0.2
LABEL_SMOOTHING = 0.1
TEST_BATCH_SIZE = 8

# Training stage configurations (should mirror ModelTraining.py)
HEAD_TRAINING = {
    'epochs': 15,
    'batch_size': 32,
    'learning_rate': 1e-4
}

PARTIAL_FINE_TUNING = {
    'epochs': 3,
    'batch_size': 32,
    'learning_rate': 3e-5,
    'weight_decay': 1e-4,
    'unfreeze_last_layers': 15
}

FULL_FINE_TUNING = {
    'epochs': 8,
    'batch_size': 32,
    'initial_learning_rate': 1e-5,
    'end_learning_rate': 1e-6,
    'weight_decay': 1e-4
}

# Callback settings
REDUCE_LR_ON_PLATEAU = {
    'monitor': 'val_loss',
    'factor': 0.5,
    'patience': 3,
    'min_lr': 1e-6
}

HEAD_EARLY_STOPPING = {
    'monitor': 'val_loss',
    'patience': 8,
    'restore_best_weights': True,
    'verbose': 1
}

PARTIAL_EARLY_STOPPING = {
    'monitor': 'val_loss',
    'patience': 5,
    'restore_best_weights': True,
    'verbose': 1
}

FULL_EARLY_STOPPING = {
    'monitor': 'val_loss',
    'patience': 8,
    'restore_best_weights': True,
    'verbose': 1
}

# Data augmentation parameters applied uniformly across folds
AUGMENTATION_PARAMS = {
    'rotation_range': 25,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.2,
    'brightness_range': (0.6, 1.4),
    'shear_range': 0.2,
    'channel_shift_range': 10,
    'horizontal_flip': True,
    'fill_mode': 'reflect'
}

# Output file names
OUTPUT_FILES = {
    'summary_json': 'cv_summary_statistics.json',
    'results_csv': 'cv_results.csv',
    'summary_plot': 'cv_results_summary.png',
    'confusion_matrix_prefix': 'confusion_matrix'
}

# Performance settings
PERFORMANCE = {
    'save_predictions': True,  # Save detailed prediction results per fold
    'skip_full_finetuning': False  # Set to True to skip full fine-tuning if OOM persists
}

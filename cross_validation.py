import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# Import configuration
try:
    from cv_config import *
except ImportError:
    # Fallback configuration if cv_config.py doesn't exist
    DATASET_PATH = "SmallerDataset"
    MODELS_DIR = "."
    IMAGE_SIZE = (224, 224)
    N_SPLITS = 5
    RANDOM_SEED = 44
    MODEL_FILES = {
        # 'resnet50': 'modelResNet50.keras',
        'vgg16': 'modelVGG16_44.keras'
    }
    OUTPUT_FILES = {
        'summary_json': 'cv_summary_statistics.json',
        'results_csv': 'cv_results.csv',
        'summary_plot': 'cv_results_summary.png',
        'confusion_matrix_prefix': 'confusion_matrix'
    }
    PLOT_SETTINGS = {
        'figure_size': (15, 12),
        'dpi': 300,
        'color_palette': 'husl',
        'grid_alpha': 0.3
    }
    PERFORMANCE = {
        'batch_prediction': False,
        'verbose_predictions': False,
        'save_predictions': True
    }

# Set random seeds for reproducibility
SEED = RANDOM_SEED
np.random.seed(SEED)
tf.random.set_seed(SEED)

class BrainTumorCrossValidator:
    def __init__(self, dataset_path=None, models_dir=None, image_size=None):
        """
        Initialize the cross-validator for brain tumor classification.
        
        Args:
            dataset_path: Path to the smaller dataset
            models_dir: Directory containing pre-trained models
            image_size: Target image size for the models
        """
        self.dataset_path = dataset_path or DATASET_PATH
        self.models_dir = models_dir or MODELS_DIR
        self.image_size = image_size or IMAGE_SIZE
        self.classes = None
        self.class_to_idx = None
        self.models = {}
        self.results = {}
        
        # Load dataset information
        self._load_dataset_info()
        
        # Load pre-trained models
        self._load_models()
        
        # Initialize data generator
        self.datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input
        )
    
    def _load_dataset_info(self):
        """Load dataset information and class mappings."""
        info_path = os.path.join(self.dataset_path, "dataset_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.classes = info['classes']
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            # Fallback: scan directory for classes
            self.classes = [d for d in os.listdir(self.dataset_path) 
                           if os.path.isdir(os.path.join(self.dataset_path, d))]
            self.classes.sort()
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Loaded {len(self.classes)} classes: {self.classes}")
    
    def _load_models(self):
        """Load pre-trained models from the models directory."""
        model_files = MODEL_FILES
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.models_dir, filename)
            if os.path.exists(model_path):
                try:
                    print(f"Loading {model_name} from {filename}...")
                    self.models[model_name] = load_model(model_path)
                    print(f"✓ {model_name} loaded successfully")
                except Exception as e:
                    print(f"✗ Failed to load {model_name}: {e}")
            else:
                print(f"⚠ Model file not found: {filename}")
        
        if not self.models:
            raise ValueError("No models could be loaded. Please check the models directory.")
    
    def _prepare_data(self):
        """Prepare image paths and labels for cross-validation."""
        image_paths = []
        labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                image_paths.append(os.path.join(class_path, img_file))
                labels.append(self.class_to_idx[class_name])
        
        return np.array(image_paths), np.array(labels)
    
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image."""
        try:
            img = image.load_img(image_path, target_size=self.image_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.resnet50.preprocess_input(x)
            return x
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def _evaluate_model_on_fold(self, model, X_val, y_val, fold_idx, model_name):
        """Evaluate a single model on a validation fold."""
        print(f"  Evaluating {model_name} on fold {fold_idx + 1}...")
        
        # Predict on validation set
        y_pred_proba = []
        y_pred = []
        
        for img_path in X_val:
            img_array = self._load_and_preprocess_image(img_path)
            if img_array is not None:
                pred = model.predict(img_array, verbose=0)
                y_pred_proba.append(pred[0])
                y_pred.append(np.argmax(pred[0]))
        
        if not y_pred:
            print(f"    Warning: No valid predictions for fold {fold_idx + 1}")
            return None
        
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        # Calculate per-class metrics
        precision_per_class = precision_score(y_val, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_val, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_val, y_pred, average=None, zero_division=0)
        
        # Calculate ROC AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(tf.keras.utils.to_categorical(y_val, num_classes=len(self.classes)), 
                                   y_pred_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = None
        
        # Create confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        
        # Store results
        fold_results = {
            'fold': fold_idx + 1,
            'model_name': model_name,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            },
            'per_class_metrics': {
                'precision': dict(zip(self.classes, precision_per_class)),
                'recall': dict(zip(self.classes, recall_per_class)),
                'f1': dict(zip(self.classes, f1_per_class))
            },
            'confusion_matrix': cm,
            'predictions': {
                'true_labels': y_val.tolist(),
                'predicted_labels': y_pred.tolist(),
                'predicted_probabilities': y_pred_proba.tolist()
            }
        }
        
        return fold_results
    
    def perform_cross_validation(self, n_splits=None):
        """Perform k-fold cross-validation on all models."""
        n_splits = n_splits or N_SPLITS
        print(f"\n{'='*60}")
        print(f"Starting {n_splits}-fold Cross-Validation")
        print(f"{'='*60}")
        
        # Prepare data
        X, y = self._prepare_data()
        print(f"Total samples: {len(X)}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        
        # Store all results
        all_results = []
        
        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
            print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Evaluate each model on this fold
            fold_results = []
            for model_name, model in self.models.items():
                result = self._evaluate_model_on_fold(model, X_val, y_val, fold_idx, model_name)
                if result:
                    fold_results.append(result)
                    all_results.append(result)
            
            # Print fold summary
            if fold_results:
                print(f"  Fold {fold_idx + 1} Results:")
                for result in fold_results:
                    metrics = result['metrics']
                    print(f"    {result['model_name']}: "
                          f"Acc={metrics['accuracy']:.4f}, "
                          f"Prec={metrics['precision']:.4f}, "
                          f"Rec={metrics['recall']:.4f}, "
                          f"F1={metrics['f1']:.4f}")
        
        # Store all results
        self.results = all_results
        
        # Generate comprehensive summary
        self._generate_summary()
        
        return all_results
    
    def _generate_summary(self):
        """Generate comprehensive summary of cross-validation results."""
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        # Group results by model
        model_results = {}
        for result in self.results:
            model_name = result['model_name']
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)
        
        # Calculate statistics for each model
        summary_stats = {}
        for model_name, results in model_results.items():
            print(f"\n📊 {model_name.upper()} RESULTS:")
            print("-" * 40)
            
            # Extract metrics
            accuracies = [r['metrics']['accuracy'] for r in results]
            precisions = [r['metrics']['precision'] for r in results]
            recalls = [r['metrics']['recall'] for r in results]
            f1_scores = [r['metrics']['f1'] for r in results]
            
            # Calculate statistics
            stats = {
                'accuracy': {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'min': np.min(accuracies),
                    'max': np.max(accuracies)
                },
                'precision': {
                    'mean': np.mean(precisions),
                    'std': np.std(precisions),
                    'min': np.min(precisions),
                    'max': np.max(precisions)
                },
                'recall': {
                    'mean': np.mean(recalls),
                    'std': np.std(recalls),
                    'min': np.min(recalls),
                    'max': np.max(recalls)
                },
                'f1': {
                    'mean': np.mean(f1_scores),
                    'std': np.std(f1_scores),
                    'min': np.min(f1_scores),
                    'max': np.max(f1_scores)
                }
            }
            
            summary_stats[model_name] = stats
            
            # Print statistics
            print(f"Accuracy:  {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f} "
                  f"[{stats['accuracy']['min']:.4f}, {stats['accuracy']['max']:.4f}]")
            print(f"Precision: {stats['precision']['mean']:.4f} ± {stats['precision']['std']:.4f} "
                  f"[{stats['precision']['min']:.4f}, {stats['precision']['max']:.4f}]")
            print(f"Recall:    {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f} "
                  f"[{stats['recall']['min']:.4f}, {stats['recall']['max']:.4f}]")
            print(f"F1-Score:  {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f} "
                  f"[{stats['f1']['min']:.4f}, {stats['f1']['max']:.4f}]")
        
        # Save detailed results
        self._save_results(summary_stats)
        
        # Create visualizations
        self._create_visualizations(summary_stats)
    
    def _save_results(self, summary_stats):
        """Save detailed results to files."""
        # Save summary statistics
        summary_file = OUTPUT_FILES['summary_json']
        with open(summary_file, 'w') as f:
            json.dump({
                'cross_validation_summary': summary_stats,
                'detailed_results': self.results,
                'dataset_info': {
                    'classes': self.classes,
                    'total_samples': len(self.results[0]['predictions']['true_labels']) if self.results else 0,
                    'n_folds': len(set(r['fold'] for r in self.results)),
                    'models_evaluated': list(self.models.keys())
                },
                'timestamp': str(np.datetime64('now'))
            }, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {summary_file}")
        
        # Save results as CSV for easy analysis
        csv_data = []
        for result in self.results:
            row = {
                'fold': result['fold'],
                'model_name': result['model_name'],
                'accuracy': result['metrics']['accuracy'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'f1_score': result['metrics']['f1'],
                'roc_auc': result['metrics']['roc_auc']
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = OUTPUT_FILES['results_csv']
        df.to_csv(csv_file, index=False)
        print(f"📊 CSV results saved to: {csv_file}")
    
    def _create_visualizations(self, summary_stats):
        """Create visualizations of the cross-validation results."""
        print("\n🎨 Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=PLOT_SETTINGS['figure_size'])
        fig.suptitle('Cross-Validation Results Summary', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            # Extract data for this metric
            model_names = list(summary_stats.keys())
            means = [summary_stats[model][metric]['mean'] for model in model_names]
            stds = [summary_stats[model][metric]['std'] for model in model_names]
            
            # Create bar plot with error bars
            bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.8)
            
            # Customize the plot
            ax.set_title(f'{metric_name} Across Folds', fontweight='bold')
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
            
            # Add grid
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_FILES['summary_plot'], dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight')
        print(f"📈 Visualization saved as: {OUTPUT_FILES['summary_plot']}")
        
        # Create confusion matrix heatmap for the best performing model
        if self.results:
            best_result = max(self.results, key=lambda x: x['metrics']['accuracy'])
            self._plot_confusion_matrix(best_result)
    
    def _plot_confusion_matrix(self, result):
        """Plot confusion matrix for a specific result."""
        cm = result['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title(f'Confusion Matrix - {result["model_name"]} (Fold {result["fold"]})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        confusion_file = f'{OUTPUT_FILES["confusion_matrix_prefix"]}_{result["model_name"]}_fold{result["fold"]}.png'
        plt.savefig(confusion_file, dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight')
        print(f"🔍 Confusion matrix saved for {result['model_name']} fold {result['fold']}")

def main():
    """Main function to run cross-validation."""
    print("🧠 Brain Tumor MRI Cross-Validation")
    print("=" * 50)
    
    # Initialize cross-validator
    try:
        cv = BrainTumorCrossValidator()
        
        # Perform cross-validation
        results = cv.perform_cross_validation()
        
        print(f"\n✅ Cross-validation completed successfully!")
        print(f"📁 Results saved to:")
        print(f"   - {OUTPUT_FILES['summary_json']} (detailed results)")
        print(f"   - {OUTPUT_FILES['results_csv']} (CSV format)")
        print(f"   - {OUTPUT_FILES['summary_plot']} (visualization)")
        
    except Exception as e:
        print(f"❌ Error during cross-validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

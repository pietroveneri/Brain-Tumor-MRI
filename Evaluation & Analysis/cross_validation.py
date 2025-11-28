import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision
import warnings

warnings.filterwarnings("ignore")
mixed_precision.set_global_policy("mixed_float16")
# Import configuration
try:
    from cv_config import *
except ImportError:
    # Fallback configuration if cv_config.py doesn't exist
    DATASET_PATH = "Dataset"
    MODELS_DIR = "."
    IMAGE_SIZE = (224, 224)
    CHECKPOINT_DIR = "cv_checkpoints"
    N_SPLITS = 2
    RANDOM_SEED = 43
    TRAIN_VALIDATION_SPLIT = 0.2
    LABEL_SMOOTHING = 0.1
    TEST_BATCH_SIZE = 32
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
        'unfreeze_last_layers': 30
    }
    FULL_FINE_TUNING = {
        'epochs': 8,
        'batch_size': 32,
        'initial_learning_rate': 1e-5,
        'end_learning_rate': 1e-6,
        'weight_decay': 1e-4
    }
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
    OUTPUT_FILES = {
        'summary_json': 'cv_summary_statistics.json',
        'results_csv': 'cv_results.csv',
        'summary_plot': 'cv_results_summary.png',
        'confusion_matrix_prefix': 'confusion_matrix'
    }
    PERFORMANCE = {
        'save_predictions': True
    }

# Set random seeds for reproducibility
SEED = RANDOM_SEED
np.random.seed(SEED)
tf.random.set_seed(SEED)

class BrainTumorCrossValidator:

    def __init__(self, dataset_path=None, models_dir=None, image_size=None):
        self.dataset_path = dataset_path or DATASET_PATH
        self.models_dir = models_dir or MODELS_DIR
        self.checkpoint_dir = os.path.join(self.models_dir, CHECKPOINT_DIR)
        self.image_size = image_size or IMAGE_SIZE
        self.label_smoothing = LABEL_SMOOTHING
        self.test_batch_size = TEST_BATCH_SIZE
        self.classes = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.num_classes = 0
        self.data_df = None
        self.results = []
        self.model_name = "resnet50_imagenet"

        # Load dataset metadata and samples
        self._load_dataset_info()

        if not 0 < TRAIN_VALIDATION_SPLIT < 1:
            raise ValueError("TRAIN_VALIDATION_SPLIT must be between 0 and 1 for early stopping.")
        self.validation_split = TRAIN_VALIDATION_SPLIT
        self._prepare_data()
        self._build_data_generators()
        self._ensure_checkpoint_dir()
    
    def _load_dataset_info(self):
        """Load dataset information and class mappings."""
        info_path = os.path.join(self.dataset_path, "dataset_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.classes = info['classes']
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            # Fallback: scan Training directory for actual classes (not Testing/Training)
            training_path = os.path.join(self.dataset_path, "Training")
            if os.path.exists(training_path):
                self.classes = [d for d in os.listdir(training_path) 
                               if os.path.isdir(os.path.join(training_path, d))]
                self.classes.sort()
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            else:
                # Second fallback if Training directory doesn't exist
                self.classes = [d for d in os.listdir(self.dataset_path) 
                               if os.path.isdir(os.path.join(self.dataset_path, d))]
                self.classes.sort()
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Loaded {len(self.classes)} classes: {self.classes}")
    
    def _ensure_checkpoint_dir(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _build_data_generators(self):

        augmentation = AUGMENTATION_PARAMS.copy()
        self.train_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
            validation_split=self.validation_split,
            **augmentation
        )
        self.test_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input
        )

    def _prepare_data(self):
        # Create a deduplicated dataframe of all images and labels.
        records = []
        seen = set()

        candidate_roots = [self.dataset_path]
        for split in ['Training', 'Testing']:
            split_path = os.path.join(self.dataset_path, split)
            if os.path.isdir(split_path):
                candidate_roots.append(split_path)

        for root_dir in candidate_roots:
            for class_name in self.classes:
                class_path = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                for filename in os.listdir(class_path):
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    full_path = os.path.abspath(os.path.join(class_path, filename))
                    if full_path in seen:
                        continue
                    seen.add(full_path)
                    records.append({'filepath': full_path, 'label': class_name})

        if not records:
            raise ValueError("No images found for cross-validation.")

        df = pd.DataFrame(records).drop_duplicates(subset='filepath').reset_index(drop=True)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        df['label_idx'] = df['label'].map(self.class_to_idx)
        self.data_df = df
        self.num_classes = len(self.classes)
        print(f"Prepared {len(self.data_df)} unique images for cross-validation.")
    
    def _build_model(self):
        """Instantiate a ResNet50 classifier mirroring ModelTrainingTested.py."""
        inputs = layers.Input(shape=(*self.image_size, 3))
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        base_model.trainable = False

        x = layers.GlobalAveragePooling2D(name="gap")(base_model.output)
        x = layers.GaussianNoise(0.1, name="noise1")(x)
        x = layers.Dense(512, activation="relu", name="fc1", kernel_regularizer=l2(1e-4))(x)
        x = layers.Dropout(0.5, name="dropout1")(x)
        x = layers.GaussianNoise(0.05, name="noise2")(x)
        x = layers.Dense(128, activation="relu", name="fc2", kernel_regularizer=l2(1e-4))(x)
        x = layers.Dropout(0.5, name="dropout2")(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name="predictions")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name=self.model_name)
        return model, base_model

    def _freeze_batch_norm_layers(self, base_model):
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    def _configure_head_training(self, base_model):
        base_model.trainable = False
        self._freeze_batch_norm_layers(base_model)

    def _configure_partial_finetuning(self, base_model):
        base_model.trainable = False
        unfreeze_last = PARTIAL_FINE_TUNING.get('unfreeze_last_layers', 30)
        if unfreeze_last > 0:
            for layer in base_model.layers[-unfreeze_last:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
        self._freeze_batch_norm_layers(base_model)

    def _configure_full_finetuning(self, base_model):
        # Unfreeze all layers for full fine-tuning
        base_model.trainable = True
        self._freeze_batch_norm_layers(base_model)

    def _compile_model(self, model, optimizer, jit_compile=True):
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=self.label_smoothing)
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            jit_compile=jit_compile
        )

    def _checkpoint_callback(self, filepath, monitor='val_accuracy', initial_value=None):
        kwargs = {
            'filepath': filepath,
            'save_best_only': True,
            'monitor': monitor,
            'save_weights_only': True,
            'verbose': 1
        }
        if initial_value is not None:
            kwargs['initial_value_threshold'] = initial_value
        return ModelCheckpoint(**kwargs)

    def _build_head_callbacks(self, checkpoint_path):
        reduce_cfg = REDUCE_LR_ON_PLATEAU
        callbacks = [
            ReduceLROnPlateau(
                monitor=reduce_cfg.get('monitor', 'val_loss'),
                factor=reduce_cfg.get('factor', 0.5),
                patience=reduce_cfg.get('patience', 3),
                min_lr=reduce_cfg.get('min_lr', 1e-6),
                verbose=1
            ),
            EarlyStopping(**HEAD_EARLY_STOPPING),
            self._checkpoint_callback(checkpoint_path, monitor='val_accuracy')
        ]
        return callbacks

    def _build_partial_callbacks(self, checkpoint_path, initial_accuracy=None):
        callbacks = [
            EarlyStopping(**PARTIAL_EARLY_STOPPING),
            self._checkpoint_callback(
                checkpoint_path,
                monitor='val_accuracy',
                initial_value=initial_accuracy
            )
        ]
        return callbacks

    def _build_full_callbacks(self, checkpoint_path, initial_accuracy=None):
        callbacks = [
            EarlyStopping(**FULL_EARLY_STOPPING),
            self._checkpoint_callback(
                checkpoint_path,
                monitor='val_accuracy',
                initial_value=initial_accuracy
            )
        ]
        return callbacks

    def _create_train_val_generators(self, train_df, batch_size):
        if self.validation_split <= 0:
            raise ValueError("Validation split must be greater than zero to use early stopping.")

        common_args = {
            'x_col': 'filepath',
            'y_col': 'label',
            'class_mode': 'categorical',
            'target_size': self.image_size,
            'color_mode': 'rgb',
            'batch_size': batch_size,
            'classes': self.classes
        }

        train_generator = self.train_datagen.flow_from_dataframe(
            train_df,
            subset='training',
            shuffle=True,
            seed=SEED,
            **common_args
        )

        val_generator = self.train_datagen.flow_from_dataframe(
            train_df,
            subset='validation',
            shuffle=False,
            seed=SEED,
            **common_args
        )

        return train_generator, val_generator

    def _create_test_generator(self, test_df):
        return self.test_datagen.flow_from_dataframe(
            test_df,
            x_col='filepath',
            y_col='label',
            class_mode='categorical',
            target_size=self.image_size,
            color_mode='rgb',
            batch_size=self.test_batch_size,
            shuffle=False,
            seed=SEED,
            classes=self.classes
        )

    def _compute_class_weights(self, labels):
        classes = np.arange(self.num_classes)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
        return {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    def _train_and_evaluate_fold(self, fold_idx, train_df, test_df):
        """Train a fresh model on the current fold and evaluate on its holdout split."""
        (model, base_model) = self._build_model()
        fold_prefix = os.path.join(self.checkpoint_dir, f"{self.model_name}_fold{fold_idx + 1}")

        test_gen = self._create_test_generator(test_df)
        class_weights = self._compute_class_weights(train_df['label_idx'])

        history_records = {}

        # Stage 1: head training
        head_cfg = HEAD_TRAINING
        head_ckpt = f"{fold_prefix}_head.weights.h5"
        train_gen, val_gen = self._create_train_val_generators(train_df, head_cfg['batch_size'])
        self._configure_head_training(base_model)
        self._compile_model(model, Adam(learning_rate=head_cfg['learning_rate']), jit_compile=True)
        head_history = model.fit(
            train_gen,
            epochs=head_cfg['epochs'],
            validation_data=val_gen,
            callbacks=self._build_head_callbacks(head_ckpt),
            class_weight=class_weights,
            verbose=1
        )
        history_records['head'] = head_history.history
        best_head_acc = max(head_history.history.get('val_accuracy', [0]))
        if os.path.exists(head_ckpt):
            model.load_weights(head_ckpt)
        
        # Re-evaluate to get best accuracy from loaded weights
        eval_results_val = model.evaluate(val_gen, verbose=0)
        best_head_acc = eval_results_val[1]  # Accuracy is the second metric

        # Stage 2: partial fine-tuning (last layers)
        partial_cfg = PARTIAL_FINE_TUNING
        partial_ckpt = f"{fold_prefix}_partial.weights.h5"
        train_gen, val_gen = self._create_train_val_generators(train_df, partial_cfg['batch_size'])
        self._configure_partial_finetuning(base_model)
        self._compile_model(
            model,
            AdamW(
                learning_rate=partial_cfg['learning_rate'],
                weight_decay=partial_cfg.get('weight_decay', 0.0)
            ),
            jit_compile=False
        )
        partial_history = model.fit(
            train_gen,
            epochs=partial_cfg['epochs'],
            validation_data=val_gen,
            callbacks=self._build_partial_callbacks(partial_ckpt, initial_accuracy=best_head_acc),
            class_weight=class_weights,
            verbose=1
        )
        history_records['partial'] = partial_history.history
        if os.path.exists(partial_ckpt):
            model.load_weights(partial_ckpt)
        best_partial_acc = model.evaluate(val_gen, verbose=0)[1]

        # Stage 3: full fine-tuning
        full_cfg = FULL_FINE_TUNING
        full_ckpt = f"{fold_prefix}_full.weights.h5"
        train_gen, val_gen = self._create_train_val_generators(train_df, full_cfg['batch_size'])
        self._configure_full_finetuning(base_model)
        steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
        decay_steps = max(1, steps_per_epoch * full_cfg['epochs'])
        lr_schedule = PolynomialDecay(
            initial_learning_rate=full_cfg['initial_learning_rate'],
            decay_steps=decay_steps,
            end_learning_rate=full_cfg['end_learning_rate'],
            power=1.0
        )
        self._compile_model(
            model,
            AdamW(
                learning_rate=lr_schedule,
                weight_decay=full_cfg.get('weight_decay', 0.0)
            ),
            jit_compile=False
        )
        full_history = model.fit(
            train_gen,
            epochs=full_cfg['epochs'],
            validation_data=val_gen,
            callbacks=self._build_full_callbacks(full_ckpt, initial_accuracy=best_partial_acc),
            class_weight=class_weights,
            verbose=1
        )
        history_records['full'] = full_history.history
        if os.path.exists(full_ckpt):
            model.load_weights(full_ckpt)

        # Evaluation on the test split for this fold
        test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
        y_pred_proba = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = test_gen.classes

        label_indices = np.arange(self.num_classes)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        precision_per_class = precision_score(
            y_true, y_pred, labels=label_indices, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true, y_pred, labels=label_indices, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true, y_pred, labels=label_indices, average=None, zero_division=0
        )

        try:
            roc_auc = roc_auc_score(
                tf.keras.utils.to_categorical(y_true, num_classes=self.num_classes),
                y_pred_proba,
                multi_class='ovr',
                average='weighted'
            )
        except Exception:
            roc_auc = None
        cm = confusion_matrix(y_true, y_pred, labels=label_indices)

        predictions_detail = None
        if PERFORMANCE.get('save_predictions', False):
            predictions_detail = {
                'filepaths': test_gen.filepaths,
                'true_labels': [self.idx_to_class[idx] for idx in y_true],
                'predicted_labels': [self.idx_to_class[idx] for idx in y_pred],
                'predicted_probabilities': y_pred_proba.tolist()
            }

        fold_result = {
            'fold': fold_idx + 1,
            'model_name': self.model_name,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            },
            'per_class_metrics': {
                'precision': dict(zip(self.classes, precision_per_class)),
                'recall': dict(zip(self.classes, recall_per_class)),
                'f1': dict(zip(self.classes, f1_per_class))
            },
            'confusion_matrix': cm.tolist(),
            'history': history_records,
            'checkpoints': {
                'head': head_ckpt,
                'partial': partial_ckpt,
                'full': full_ckpt
            },
            'predictions': predictions_detail
        }

        return fold_result

    def perform_cross_validation(self, n_splits=None):
        """Perform stratified k-fold CV, training from ImageNet weights each time."""
        n_splits = n_splits or N_SPLITS
        print(f"\n{'=' * 60}")
        print(f"Starting {n_splits}-fold Cross-Validation with {self.model_name}")
        print(f"{'=' * 60}")

        y = self.data_df['label_idx'].values
        class_counts = np.bincount(y, minlength=self.num_classes)
        print(f"Total samples: {len(self.data_df)}")
        print(f"Class distribution: {class_counts}")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        self.results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.data_df['filepath'], y)):
            print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
            print(f"Train indices: {len(train_idx)}, Test indices: {len(test_idx)}")

            train_df = self.data_df.iloc[train_idx].reset_index(drop=True)
            test_df = self.data_df.iloc[test_idx].reset_index(drop=True)

            fold_result = self._train_and_evaluate_fold(fold_idx, train_df, test_df)
            self.results.append(fold_result)


            metrics = fold_result['metrics']
            print(f"Fold {fold_idx + 1} => "
                  f"Acc={metrics['accuracy']:.4f}, "
                  f"Prec={metrics['precision']:.4f}, "
                  f"Rec={metrics['recall']:.4f}, "
                  f"F1={metrics['f1']:.4f}")

        self._generate_summary()
        return self.results
    
    def _generate_summary(self):
        """Generate comprehensive summary of cross-validation results."""
        if not self.results:
            print("No fold results to summarize.")
            return

        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        accuracies = [r['metrics']['accuracy'] for r in self.results]
        precisions = [r['metrics']['precision'] for r in self.results]
        recalls = [r['metrics']['recall'] for r in self.results]
        f1_scores = [r['metrics']['f1'] for r in self.results]

        summary_stats = {
            self.model_name: {
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
        }

        print(f"\n{self.model_name.upper()} RESULTS:")
        print("-" * 40)
        stats = summary_stats[self.model_name]
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
                    'total_samples': len(self.data_df),
                    'n_folds': len(self.results),
                    'model_name': self.model_name
                },
                'timestamp': str(np.datetime64('now'))
            }, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {summary_file}")
        
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
        print(f"CSV results saved to: {csv_file}")
    

def main():
    #Main function to run cross-validation
    print("Brain Tumor MRI Cross-Validation")
    print("=" * 50)
    
    # Initialize cross-validator
    try:
        cv = BrainTumorCrossValidator()
        
        # Perform cross-validation
        results = cv.perform_cross_validation()
        
        print(f"\nCross-validation completed successfully!")
        print(f"Results saved to:")
        print(f"   - {OUTPUT_FILES['summary_json']} (detailed results)")
        print(f"   - {OUTPUT_FILES['results_csv']} (CSV format)")
        
    except Exception as e:
        print(f"Error during cross-validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

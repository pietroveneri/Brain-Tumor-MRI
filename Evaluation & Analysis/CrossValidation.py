import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image as PILImage # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score # type: ignore
from sklearn.model_selection import StratifiedKFold # type: ignore
from sklearn.utils.class_weight import compute_class_weight # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.applications import ResNet50, VGG16 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, GaussianNoise # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam, SGD, AdamW # type: ignore
from tensorflow.keras.metrics import Precision, Recall # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras import mixed_precision # type: ignore
import warnings
warnings.filterwarnings("ignore")
import json
import pandas as pd # type: ignore

mixed_precision.set_global_policy("mixed_float16")



K_FOLDS = 5
RANDOM_SEED = 42  
image_size = (224, 224)
validation_split = 0.2  


def remove_damaged_images(dataset_path):
    removed_count = 0
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            file_path = os.path.join(root, file)
            try:
                with PILImage.open(file_path) as img:
                    img.verify()
                    if img.size[0] < 10 or img.size[1] < 10:
                        raise ValueError("Image too small")
            except (IOError, SyntaxError, ValueError) as e:
                print(f"Removing damaged image: {file_path} - Error: {e}")
                try:
                    os.remove(file_path)
                    removed_count += 1
                except OSError as e:
                    print(f"Error removing file {file_path}: {e}")

    print(f"Total damaged images removed: {removed_count}")

def prepare_data_from_training_folder(training_dir):

    data = []
    classes = []
    
    for item in os.listdir(training_dir):
        class_path = os.path.join(training_dir, item)
        if os.path.isdir(class_path):
            classes.append(item)
    
    classes.sort()
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for class_name in classes:
        class_path = os.path.join(training_dir, class_name)
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(class_path, filename)
                data.append((filepath, class_name, class_to_idx[class_name]))
    
    print(f"Found {len(classes)} classes: {classes}")
    print(f"Total images: {len(data)}")
    
    return data, classes, class_to_idx


def build_model(num_classes=4):
#base_model = ResNet50(
#    weights="imagenet",
#    include_top=False,
#     input_shape=(224, 224, 3)
# )
    
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = GaussianNoise(0.1, name="noise1")(x)
    x = Dense(512, activation="relu", name="fc1", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5, name="dropout1")(x)
    x = GaussianNoise(0.05, name="noise2")(x)
    x = Dense(128, activation="relu", name="fc2", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5, name="dropout2")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name="VGG16_Tumor")
    return model, base_model


def train_head(model, base_model, train_gen, val_gen, fold_idx, epochs=15):
    """Stage 1: Head training (base model congelato)"""
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1} - Stage 1: Head Training")
    print(f"{'='*60}")
    
    base_model.trainable = False
    
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        jit_compile=True
    )
    
    classes = train_gen.classes
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    class_weights = dict(enumerate(class_weights_array))
    
    checkpoint_path = f"best_resnet_fold{fold_idx + 1}_head.keras"
    
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    model.load_weights(checkpoint_path)
    eval_results = model.evaluate(val_gen, verbose=0)
    best_val_accuracy = eval_results[1]
    print(f"Best validation accuracy after head training: {best_val_accuracy:.4f}")
    
    return history, best_val_accuracy

def train_partial_finetuning(model, base_model, train_gen, val_gen, fold_idx, initial_acc, epochs=3):
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1} - Stage 2: Partial Fine-tuning")
    print(f"{'='*60}")
    
    base_model.trainable = False
    
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    model.compile(
        optimizer=AdamW(learning_rate=3e-5, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        jit_compile=False
    )
    
    classes = train_gen.classes
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    class_weights = dict(enumerate(class_weights_array))
    
    checkpoint_path = f"best_resnet_fold{fold_idx + 1}_partial.keras"
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            initial_value_threshold=initial_acc
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    model.load_weights(checkpoint_path)
    eval_results = model.evaluate(val_gen, verbose=0)
    best_val_accuracy = eval_results[1]
    print(f"Best validation accuracy after partial fine-tuning: {best_val_accuracy:.4f}")
    
    return history, best_val_accuracy

def train_full_finetuning(model, base_model, train_gen, val_gen, fold_idx, initial_acc, epochs=8):
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1} - Stage 3: Full Fine-tuning")
    print(f"{'='*60}")
    
    base_model.trainable = True
    
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    steps_per_epoch = train_gen.samples // train_gen.batch_size
    total_steps = steps_per_epoch * epochs
    initial_learning_rate = 1e-5
    
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=total_steps,
        end_learning_rate=1e-6,
        power=1.0,
        name='PolynomialDecay'
    )
    
    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        jit_compile=False
    )
    
    classes = train_gen.classes
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    class_weights = dict(enumerate(class_weights_array))
    
    checkpoint_path = f"best_resnet_fold{fold_idx + 1}_full.keras"
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            initial_value_threshold=initial_acc
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    model.load_weights(checkpoint_path)
    eval_results = model.evaluate(val_gen, verbose=0)
    best_val_accuracy = eval_results[1]
    print(f"Best validation accuracy after full fine-tuning: {best_val_accuracy:.4f}")
    
    return history, best_val_accuracy


def create_temporary_directories(train_data, val_data, fold_idx, base_dir="temp_cv_folds"):

    train_dir = os.path.join(base_dir, f"fold{fold_idx + 1}_train")
    val_dir = os.path.join(base_dir, f"fold{fold_idx + 1}_val")
    
    for dir_path in [train_dir, val_dir]:
        os.makedirs(dir_path, exist_ok=True)
        for class_name in set([d[1] for d in train_data] + [d[1] for d in val_data]):
            os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)
    
    import shutil
    for filepath, class_name, _ in train_data:
        filename = os.path.basename(filepath)
        dest = os.path.join(train_dir, class_name, filename)
        shutil.copy2(filepath, dest)
    
    for filepath, class_name, _ in val_data:
        filename = os.path.basename(filepath)
        dest = os.path.join(val_dir, class_name, filename)
        shutil.copy2(filepath, dest)
    
    return train_dir, val_dir

def cleanup_temporary_directories(base_dir="temp_cv_folds"):
    import shutil
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        print(f"Cleaned up temporary directories: {base_dir}")

def train_and_evaluate_fold(fold_idx, train_data, val_data, classes, class_to_idx, num_classes):

    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx + 1}/{K_FOLDS}")
    print(f"{'='*80}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    train_dir, val_dir = create_temporary_directories(train_data, val_data, fold_idx)
    
    try:
        model, base_model = build_model(num_classes)
        
        batch_size = 32
        
        train_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
            validation_split=validation_split,
            zoom_range=0.2,
            brightness_range=(0.6, 1.4),
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            rotation_range=25,
            horizontal_flip=True,
            channel_shift_range=10,
            fill_mode="reflect",
        )
        
        val_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
            validation_split=validation_split
        )
        
        val_fold_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input
        )
        
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=RANDOM_SEED,
            classes=classes
        )
        
        val_gen = val_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=RANDOM_SEED,
            classes=classes
        )
        
        val_fold_gen = val_fold_datagen.flow_from_directory(
            val_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
            seed=RANDOM_SEED,
            classes=classes
        )
        
        head_history, best_head_acc = train_head(model, base_model, train_gen, val_gen, fold_idx)
        
        batch_size = 16
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=RANDOM_SEED,
            classes=classes
        )
        
        val_gen = val_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=RANDOM_SEED,
            classes=classes
        )
        
        partial_history, best_partial_acc = train_partial_finetuning(
            model, base_model, train_gen, val_gen, fold_idx, best_head_acc
        )
        
        full_history, best_full_acc = train_full_finetuning(
            model, base_model, train_gen, val_gen, fold_idx, best_partial_acc
        )
        
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1} - Valutazione sul Validation Fold")
        print(f"{'='*60}")
        
        metrics = model.evaluate(val_fold_gen, verbose=1)
        val_loss = metrics[0]
        val_accuracy = metrics[1]
        val_precision = metrics[2]
        val_recall = metrics[3]
        val_f1 = (2 * val_precision * val_recall) / (val_precision + val_recall + 1e-8)
        y_pred_proba = model.predict(val_fold_gen, verbose=0)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        y_true = val_fold_gen.classes
        
        accuracy = accuracy_score(y_true, y_pred_classes)
        precision = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)
        
        precision_per_class = precision_score(
            y_true, y_pred_classes, labels=np.arange(num_classes), average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true, y_pred_classes, labels=np.arange(num_classes), average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true, y_pred_classes, labels=np.arange(num_classes), average=None, zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred_classes, labels=np.arange(num_classes))
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=classes))100
        
        fold_results = {
            'fold': fold_idx + 1,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'val_loss': float(val_loss),
                'val_accuracy': float(val_accuracy),
                'val_precision': float(val_precision),
                'val_recall': float(val_recall),
                'val_f1_score': float(val_f1)
            },
            'per_class_metrics': {
                'precision': {cls: float(p) for cls, p in zip(classes, precision_per_class)},
                'recall': {cls: float(r) for cls, r in zip(classes, recall_per_class)},
                'f1_score': {cls: float(f) for cls, f in zip(classes, f1_per_class)}
            },
            'confusion_matrix': cm.tolist(),
            'best_val_accuracies': {
                'head': float(best_head_acc),
                'partial': float(best_partial_acc),
                'full': float(best_full_acc)
            }
        }
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        return fold_results
        
    finally:
        cleanup_temporary_directories()

def perform_cross_validation():

    print("="*80)
    print("CROSS-VALIDATION STRATIFICATA K-FOLD")
    print("="*80)
    
    dataset_path = os.path.abspath(os.path.join(os.getcwd(), "Dataset"))
    training_dir = os.path.join(dataset_path, "Training")
    
    if not os.path.exists(training_dir):
        raise ValueError(f"Training directory not found: {training_dir}")
    
    print("\nRimozione immagini danneggiate...")
    remove_damaged_images(training_dir)
    
    print("\nPreparazione dati...")
    data, classes, class_to_idx = prepare_data_from_training_folder(training_dir)
    
    if len(data) == 0:
        raise ValueError("No images found in Training directory")
    
    num_classes = len(classes)
    
    filepaths = np.array([d[0] for d in data])
    labels = np.array([d[2] for d in data])  
    
    print(f"\nEsecuzione {K_FOLDS}-fold cross-validation stratificata...")
    print(f"Shuffling: True, Random Seed: {RANDOM_SEED}")
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    all_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(filepaths, labels)):
        train_data = [(filepaths[i], classes[labels[i]], labels[i]) for i in train_idx]
        val_data = [(filepaths[i], classes[labels[i]], labels[i]) for i in val_idx]
        
        fold_results = train_and_evaluate_fold(
            fold_idx, train_data, val_data, classes, class_to_idx, num_classes
        )
        all_results.append(fold_results)
    
    print(f"\n{'='*80}")
    print("RISULTATI FINALI - MEDIA SU TUTTI I FOLD")
    print(f"{'='*80}")
    
    accuracies = [r['metrics']['accuracy'] for r in all_results]
    precisions = [r['metrics']['precision'] for r in all_results]
    recalls = [r['metrics']['recall'] for r in all_results]
    f1_scores = [r['metrics']['f1_score'] for r in all_results]
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"\nMetriche Aggregate:")
    print(f"  Accuracy:  {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"  Recall:    {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"  F1-Score:  {mean_f1:.4f} ± {std_f1:.4f}")
    
    summary = {
        'k_folds': K_FOLDS,
        'random_seed': RANDOM_SEED,
        'total_samples': len(data),
        'classes': classes,
        'mean_metrics': {
            'accuracy': float(mean_accuracy),
            'std_accuracy': float(std_accuracy),
            'precision': float(mean_precision),
            'std_precision': float(std_precision),
            'recall': float(mean_recall),
            'std_recall': float(std_recall),
            'f1_score': float(mean_f1),
            'std_f1_score': float(std_f1)
        },
        'fold_results': all_results
    }
    
    with open('cross_validation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nRisultati salvati in: cross_validation_results.json")
    
    df_results = pd.DataFrame([
        {
            'fold': r['fold'],
            'accuracy': r['metrics']['accuracy'],
            'precision': r['metrics']['precision'],
            'recall': r['metrics']['recall'],
            'f1_score': r['metrics']['f1_score'],
            'val_loss': r['metrics']['val_loss'],
            'val_accuracy': r['metrics']['val_accuracy']
        }
        for r in all_results
    ])
    df_results.to_csv('cross_validation_results.csv', index=False)
    print(f"Risultati CSV salvati in: cross_validation_results.csv")
    
    return summary

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    try:
        results = perform_cross_validation()
        print("\n" + "="*80)
        print("CROSS-VALIDATION COMPLETATA CON SUCCESSO!")
        print("="*80)
    except Exception as e:
        print(f"\nErrore durante la cross-validation: {e}")
        import traceback
        traceback.print_exc()
        raise


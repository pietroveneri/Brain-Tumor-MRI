import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image as PILImage # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam, SGD # type: ignore
from tensorflow.keras.metrics import Precision, Recall # type: ignore
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras import mixed_precision # type: ignore
mixed_precision.set_global_policy("mixed_float16")

# Set random seeds for reproducibility
SEED = 44
import random
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

max_images_per_class = 4
image_size = (224,224)


images = []
labels = []
dataset_path = os.path.abspath(os.path.join(os.getcwd(), "Dataset"))
training_dir = os.path.join(dataset_path, "Training")
testing_dir = os.path.join(dataset_path, "Testing")

# Remove demaged images

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
                    # Verify image can be loaded
                    img.verify()
                    # Check image dimensions
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

remove_damaged_images(training_dir)
remove_damaged_images(testing_dir)

# Normalize the images

batch_size = 32  

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2,
    zoom_range=0.2,
    brightness_range=(0.6,1.4),
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    rotation_range=25,
    horizontal_flip=True,
    channel_shift_range=10,
    fill_mode="reflect",
)

# Validation data generator - only preprocessing, no augmentation
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2,
)

# Test data generator - only preprocessing
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
 
)

val_generator = val_datagen.flow_from_directory(
    training_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False,

)

test_generator = test_datagen.flow_from_directory(
    testing_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

# Print shapes for debugging
print("\nTest generator class indices:", test_generator.class_indices)
print("Number of test samples:", test_generator.samples)
print("Number of classes:", len(test_generator.class_indices))

# Build the model

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False  

from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.layers import GaussianNoise, Input# type: ignore

x = base_model.output
x = GlobalAveragePooling2D(name="gap")(x)
x = GaussianNoise(0.1, name="noise1")(x)
x = Dense(512, activation="relu", name="fc1", kernel_regularizer=l2(1e-4))(x)  
x = Dropout(0.5, name="dropout1")(x)
x = GaussianNoise(0.05, name="noise2")(x)
x = Dense(128, activation="relu", name="fc2", kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.5, name="dropout2")(x)
outputs = Dense(4, activation="softmax", name="predictions")(x) 

model = Model(inputs=base_model.input, outputs=outputs, name="ResNet50_Tumor")

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    jit_compile=True
)   

model.summary()

epochs_head = 15

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
        "best_resnet.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
]

from sklearn.utils.class_weight import compute_class_weight # type: ignore

classes = train_generator.classes
# Calculate balanced class weights
class_weights_array = compute_class_weight(
    class_weight = 'balanced',
    classes=np.unique(classes),
    y=classes,

)
class_weights = dict(enumerate(class_weights_array))

history_head = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_head,
    callbacks=callbacks,
    class_weight=class_weights
)

# Load the best weights saved during head training
print("Loading best model weights from head training phase...")
model.load_weights("best_resnet.keras")

# Evaluate this model on the validation set to get its val_accuracy
print("Evaluating loaded model on validation set...")
eval_results_val = model.evaluate(val_generator, verbose=0)
best_val_accuracy_from_head = eval_results_val[1]  # Accuracy is the second metric
print(f"Best val_accuracy from head training (loaded and re-evaluated): {best_val_accuracy_from_head:.4f}")

# Reduce batch size for fine-tuning
batch_size = 16

# First stage of fine-tuning: unfreeze only the last few ResNet blocks
base_model.trainable = False

for layer in base_model.layers[-30:]: 
    layer.trainable = True

# Freeze BatchNormalization layers
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle = True,
)

val_generator = val_datagen.flow_from_directory(
    training_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle = False,
)

from tensorflow.keras.optimizers import AdamW # type: ignore

epochs_partial_finetune = 3

steps_per_epoch = train_generator.samples // batch_size

# Compile for partial fine-tuning with slightly higher learning rate
model.compile(
    optimizer=AdamW(learning_rate=3e-5, weight_decay=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    jit_compile=False
)

# Add callbacks for partial fine-tuning
callbacks_partial_ft = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        "best_resnet_partial_ft.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        initial_value_threshold=best_val_accuracy_from_head
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./logs/partial_ft', histogram_freq=1)
]

print("\nPartial fine-tuning (last few layers only)...")
history_partial_ft = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_partial_finetune,
    callbacks=callbacks_partial_ft, 
    class_weight=class_weights
)

# Load best weights from partial fine-tuning
model.load_weights("best_resnet_partial_ft.keras")
best_val_accuracy_from_partial_ft = model.evaluate(val_generator, verbose=0)[1]
print(f"Best val_accuracy from partial fine-tuning: {best_val_accuracy_from_partial_ft:.4f}")

# Second stage: full fine-tuning
print("\nFull fine-tuning (all layers)...")
# Unfreeze all layers for final fine-tuning
base_model.trainable = True

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

epochs_finetune = 8

total_steps = steps_per_epoch * epochs_finetune
initial_learning_rate = 1e-5

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=total_steps,
    end_learning_rate=1e-6,
    power=1.0,
    name = 'PolynomialDecay'
)

model.compile(
    optimizer=AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    jit_compile=False
)

# Add memory-efficient callbacks for full fine-tuning
callbacks_ft = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        "best_resnet.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        initial_value_threshold=best_val_accuracy_from_partial_ft  # Use best accuracy from partial fine-tuning
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./logs/full_ft', histogram_freq=1)
]

history_ft = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_finetune,
    callbacks=callbacks_ft, 
    class_weight=class_weights
)

# Evaluate the model
print("\nEvaluating model...")
metrics = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {metrics[0]:.4f}")
print(f"Test Accuracy: {metrics[1]:.4f}")

# Get predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred_classes)

print(classification_report(y_true, y_pred_classes, target_names=class_names))

model.save('modelResNet50_44.keras')
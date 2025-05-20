import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import opendatasets as od # type: ignore
od.download("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data")
from PIL import Image as PILImage # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from sklearn.metrics import classification_report, confusion_matrix # type: ignore

import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam, SGD # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore
from tensorflow.keras.metrics import Precision, Recall # type: ignore
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras import mixed_precision # type: ignore
mixed_precision.set_global_policy("mixed_float16")

max_images_per_class = 4
image_size = (224,224)

images = []
labels = []
dataset_path = os.path.abspath(os.path.join(os.getcwd(), "brain-tumor-mri-dataset"))
training_dir = os.path.join(dataset_path, "Training")
testing_dir = os.path.join(dataset_path, "Testing")

# Verify directories exist
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset directory not found at: {dataset_path}")
if not os.path.exists(training_dir):
    raise FileNotFoundError(f"Training directory not found at: {training_dir}")
if not os.path.exists(testing_dir):
    raise FileNotFoundError(f"Testing directory not found at: {testing_dir}")


# Remove demaged images

def remove_damaged_images(dataset_path):
    removed_count = 0
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            file_path = os.path.join(root, file)
            try:
                img = PILImage.open(file_path)
                img.verify()
            except (IOError, SyntaxError):
                print(f"Removing damaged image: {file_path}")
                os.remove(file_path)
                removed_count += 1

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

# Separate generator for validation data - only rescaling
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle = True,

)

val_generator = val_datagen.flow_from_directory( # Use val_datagen here
    training_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle = False, 

)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    testing_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle = False,
)

# Build the model

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False  # freeze all for first tuning

from tensorflow.keras.regularizers import l2 # type: ignore

x = base_model.output
x = GlobalAveragePooling2D(name="gap")(x)
x = Dense(512, activation="relu", name="fc1", kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.4, name="dropout")(x)
x = Dense(128, activation="relu", name="fc2", kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.4, name="dropout2")(x)
outputs = Dense(4, activation="softmax", name="predictions")(x)

model = Model(inputs=base_model.input, outputs=outputs, name="ResNet50_Tumor")

# Compilazione
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    jit_compile=True
)   

model.summary()

epochs_head = 30

callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_resnet.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
]

from sklearn.utils.class_weight import compute_class_weight # type: ignore

classes = train_generator.classes
# Calculate balanced class weights
class_weights_array = compute_class_weight(
    class_weight = 'balanced',
    classes=np.unique(classes),
    y=classes
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

from tensorflow.keras.optimizers import SGD # type: ignore

# Reduce batch size for fine-tuning
batch_size = 16

# Unfreeze all layers for fine-tuning
base_model.trainable = True

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

val_generator = val_datagen.flow_from_directory( # Use val_datagen here
    training_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle = False,

)

model.compile(
    optimizer=SGD(learning_rate=5e-5, momentum=0.9, nesterov=True),  # Reduced learning rate
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    jit_compile=True
)

# Add memory-efficient callbacks
callbacks_ft = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        "best_resnet.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        initial_value_threshold=best_val_accuracy_from_head  # Use the best val_accuracy from head training
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)  # For monitoring
]

epochs_finetune = 15

# Use mixed precision training

history_ft = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_finetune,
    callbacks=callbacks_ft, # Use the new callbacks list
    class_weight=class_weights
)

# Evaluate the model
metrics = model.evaluate(test_generator)
print("\nTest Metrics:")
print(f"Test Loss: {metrics[0]:.4f}")
print(f"Test Accuracy: {metrics[1]:.4f}")

# Get predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_true, y_pred_classes, target_names=class_names))
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(history_head.history['accuracy'])
plt.plot(history_head.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history_head.history['loss'])
plt.plot(history_head.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(history_ft.history['accuracy'])
plt.plot(history_ft.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history_ft.history['loss'])
plt.plot(history_ft.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()

# Uncomment the following line to save the model 
# model.save('modelResNet50_Tumor.keras')

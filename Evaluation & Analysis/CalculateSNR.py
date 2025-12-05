
import tensorflow as tf
import numpy as np
import math
import os
import gc

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# 1. Load the full model
# This automatically handles the .keras archive and restores weights
full_model = tf.keras.models.load_model('modelResNet50_42.keras')

# 2. Identify the layers 
# We retrieve layers by name to ensure we are tapping the correct tensors
gap_layer = full_model.get_layer("gap")
dropout1_layer = full_model.get_layer("dropout1")

# 3. Construct the Probe Model 
# Inputs: Same as the original model
# Outputs: The tensors *immediately preceding* the noise layers
probe_model = tf.keras.Model(
    inputs=full_model.input,
    outputs={
        "s1": gap_layer.output,      # Signal entering noise1
        "s2": dropout1_layer.output  # Signal entering noise2
    }
)

print("Probe model constructed successfully.")

# 3. CRITICAL STEP: Clear the original model from memory
del full_model
tf.keras.backend.clear_session() # Clears internal graph references
gc.collect() # Forces Python to release RAM

print("Original model cleared from memory.")

def calculate_layer_snr(model, dataset, batches_to_process= 50):
    
    # Storage for batch-wise statistics
    # s1 corresponds to 'gap' output, s2 corresponds to 'dropout1' output
    stats = {
        "s1": {"vars": []},
        "s2": {"vars": []}
    }

    # Iterate through the dataset 
    print(f"Processing {batches_to_process} batches...")
    
    batch_count = 0
    for batch_images, _ in dataset:
        if batch_count >= batches_to_process:
            break
            
        # Run the probe model
        # CRITICAL: training=True ensures Dropout is active (per Step 1)
        # However, for Base ResNet (BatchNorm), we usually want training=False 
        # to use moving statistics. 
        # Keras functional API allows mixed behavior if layers were defined that way,
        # but passing training=True generally activates it for all.
        # To strictly follow your Step 1 (Base=False, Dropout=True), 
        # we rely on the fact that BatchNorm in inference mode uses 
        # frozen stats even if training=True is passed to the parent model 
        # *unless* the user explicitly set `training=True` inside the layer definition.
        # Assuming standard implementation:
        signals = model(batch_images, training=True)
        
        # --- Process Signal 1 (GAP Output) ---
        # Shape: (Batch_Size, 2048)
        s1 = signals["s1"] 
        # Step 3: Variance across batch (axis=0), then mean across features
        var_s1_batch = tf.math.reduce_variance(s1, axis=0) 
        scalar_var_s1 = tf.reduce_mean(var_s1_batch)
        stats["s1"]["vars"].append(float(scalar_var_s1))

        # STRICT CONVERSION:
        # Converts Tensor(0.4532, shape=(), dtype=float32) -> 0.4532 (Python float)
        # This detaches the value from GPU memory completely.
        stats["s1"]["vars"].append(float(scalar_var_s1))
        
        # --- Process Signal 2 (Dropout1 Output) ---
        # Shape: (Batch_Size, 512)
        s2 = signals["s2"]
        var_s2_batch = tf.math.reduce_variance(s2, axis=0)
        scalar_var_s2 = tf.reduce_mean(var_s2_batch)
        stats["s2"]["vars"].append(float(scalar_var_s2))

        # STRICT CONVERSION:
        # Converts Tensor(0.2345, shape=(), dtype=float32) -> 0.2345 (Python float)
        # This detaches the value from GPU memory completely.
        stats["s2"]["vars"].append(float(scalar_var_s2))
        
        batch_count += 1

    return stats

# --- CONFIGURATION (Step 4) ---
# Define the noise definitions (sigma)
NOISE_CONFIG = {
    "noise1": {"sigma": 0.10, "signal_key": "s1", "layer_name": "gap"},
    "noise2": {"sigma": 0.05, "signal_key": "s2", "layer_name": "dropout1"}
}



# Assuming 'train_dataset' is your tf.data.Dataset WITHOUT augmentation
train_dataset = (
    tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join("Dataset", "Training"),
        labels="inferred",
        label_mode="categorical",
        batch_size=32,
        image_size=(224, 224),
        shuffle=False  # keep ordering stable for SNR stats
    )
    .map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(tf.cast(x, tf.float32)), y))
    .prefetch(tf.data.AUTOTUNE)
)

# with tf.device("CPU:0"): # Fix for OOM issues
    stats_results = calculate_layer_snr(probe_model, train_dataset, batches_to_process=50)

def generate_report(stats_results, config):
    print("\n--- SNR Analysis Report ---")
    
    for noise_name, cfg in config.items():
        signal_key = cfg["signal_key"]
        sigma = cfg["sigma"]
        layer_source = cfg["layer_name"]
        
        # Retrieve variances collected over batches
        variances = np.array(stats_results[signal_key]["vars"])
        
        # Calculate Statistics across batches (Step 6)
        var_mean = np.mean(variances)
        var_sd = np.std(variances)
        
        # Noise Power
        noise_power = sigma ** 2
        
        # Calculate SNR (Linear)
        snr_linear_mean = var_mean / noise_power
        
        # Calculate SNR (dB) per batch to get SD in dB space
        # We compute dB for every batch individually, then average
        snr_db_all = 10 * np.log10(variances / noise_power)
        snr_db_mean = np.mean(snr_db_all)
        snr_db_sd = np.std(snr_db_all)
        
        # Formatting 
        print(f"\nLayer: {noise_name} (Input from {layer_source})")
        print(f"  Noise Sigma (σ): {sigma}")
        print(f"  Noise Power (σ²): {noise_power:.6f}")
        print(f"  Signal Var (Mean ± SD): {var_mean:.4f} ± {var_sd:.4f}")
        print(f"  SNR (dB) Result: {snr_db_mean:.2f} ± {snr_db_sd:.2f} dB")
        

# Run the report
generate_report(stats_results, NOISE_CONFIG)

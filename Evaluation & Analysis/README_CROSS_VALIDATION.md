# Brain Tumor MRI Cross-Validation System

This system performs 5-way cross-validation on your pre-trained brain tumor classification models using the smaller dataset you've created. It saves computational time by reusing your already trained models instead of retraining them.

## Quick Start

1. **Run the cross-validation:**
   ```bash
   python run_cv.py
   ```

2. **Or run directly:**
   ```bash
   python cross_validation.py
   ```

## Files Overview

- **`cross_validation.py`** - Main cross-validation implementation
- **`run_cv.py`** - Simple runner script with error checking
- **`cv_config.py`** - Configuration file for easy customization
- **`README_CROSS_VALIDATION.md`** - This documentation file

## Requirements

- Python 3.7+
- TensorFlow 2.15.0
- Your pre-trained models (`.keras` files)
- SmallerDataset directory (created by `create_smaller_dataset.py`)
- All dependencies from `requirements.txt`

## What the System Does

### 1. **Model Loading**
- Automatically detects and loads your pre-trained models:
  - `modelResNet50.keras`
  - `best_resnet.keras`
  - `best_resnet_partial_ft.keras`

### 2. **5-Fold Cross-Validation**
- Splits your smaller dataset (2000 images) into 5 folds
- Each fold uses 80% for training (1600 images) and 20% for validation (400 images)
- Maintains class balance across folds using stratified sampling

### 3. **Comprehensive Evaluation**
For each model and each fold, calculates:
- **Accuracy** - Overall correct predictions
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1-Score** - Harmonic mean of precision and recall
- **ROC AUC** - Area under the ROC curve (when possible)
- **Per-class metrics** - Individual performance for each tumor type
- **Confusion matrices** - Detailed error analysis

### 4. **Statistical Summary**
- **Mean ± Standard Deviation** across all 5 folds
- **Min/Max ranges** for each metric
- **Model comparison** to identify the best performer

## 📈 Output Files

### 1. **`cv_summary_statistics.json`**
Comprehensive results including:
- Cross-validation summary with mean ± std for each metric
- Detailed results for each fold and model
- Dataset information and metadata

### 2. **`cv_results.csv`**
Tabular format for easy analysis in Excel/Python:
- Fold number, model name, and all metrics
- One row per fold-model combination

### 3. **`cv_results_summary.png`**
Visual summary showing:
- Bar charts for accuracy, precision, recall, and F1-score
- Error bars showing standard deviation across folds
- Easy comparison between models

### 4. **Confusion Matrix Images**
- Individual confusion matrices for the best performing model
- Shows where each model makes classification errors


## Understanding the Results

### **Metrics Interpretation**

- **Accuracy > 0.90**: Excellent performance
- **Accuracy 0.80-0.90**: Good performance  
- **Accuracy 0.70-0.80**: Acceptable performance
- **Accuracy < 0.70**: May need model improvement

### **Standard Deviation**
- **Low std (< 0.05)**: Consistent performance across folds
- **High std (> 0.10)**: Unstable performance, may indicate overfitting

### **Model Comparison**
- Compare mean performance across all metrics
- Consider both performance and stability (lower std is better)
- Best model: highest mean with lowest standard deviation

## ⚙Customization

### **Modify `cv_config.py` to change:**

- Dataset path and image size
- Number of cross-validation folds
- Model files to evaluate
- Output file names
- Visualization settings

### **Example modifications:**

```python
# Change to 10-fold CV
N_SPLITS = 10

# Add custom model
MODEL_FILES['custom_model'] = 'my_model.keras'

# Change output directory
OUTPUT_FILES['summary_json'] = 'results/my_cv_summary.json'
```

## Troubleshooting

### **Common Issues:**

1. **"No models could be loaded"**
   - Check that `.keras` files exist in the current directory
   - Verify file permissions

2. **Memory errors during prediction**
   - Set `PERFORMANCE['batch_prediction'] = True` in config
   - Reduce image size if needed

3. **CUDA/GPU errors**
   - The script automatically handles TensorFlow warnings
   - Models will run on CPU if GPU is unavailable

### **Performance Tips:**

- **Faster processing**: Set `PERFORMANCE['batch_prediction'] = True`
- **Less output**: Set `PERFORMANCE['verbose_predictions'] = False`
- **Save memory**: Set `PERFORMANCE['save_predictions'] = False`

## 📊 Example Output

```
 Brain Tumor MRI Cross-Validation
==================================================

Loaded 4 classes: ['glioma', 'meningioma', 'notumor', 'pituitary']
Loading resnet50 from modelResNet50.keras...
✓ resnet50 loaded successfully
Loading best_resnet from best_resnet.keras...
✓ best_resnet loaded successfully

============================================================
Starting 5-fold Cross-Validation
============================================================
Total samples: 2000
Class distribution: [500 500 500 500]

--- Fold 1/5 ---
Train samples: 1600, Validation samples: 400
  Evaluating resnet50 on fold 1...
  Evaluating best_resnet on fold 1...
  Fold 1 Results:
    resnet50: Acc=0.8925, Prec=0.8942, Rec=0.8925, F1=0.8923
    best_resnet: Acc=0.9150, Prec=0.9168, Rec=0.9150, F1=0.9149

============================================================
CROSS-VALIDATION SUMMARY
============================================================

 RESNET50 RESULTS:
----------------------------------------
Accuracy:  0.8845 ± 0.0234 [0.8525, 0.9125]
Precision: 0.8867 ± 0.0241 [0.8542, 0.9142]
Recall:    0.8845 ± 0.0234 [0.8525, 0.9125]
F1-Score:  0.8842 ± 0.0237 [0.8523, 0.9123]

 BEST_RESNET RESULTS:
----------------------------------------
Accuracy:  0.9085 ± 0.0189 [0.8850, 0.9300]
Precision: 0.9102 ± 0.0195 [0.8868, 0.9318]
Recall:    0.9085 ± 0.0189 [0.8850, 0.9300]
F1-Score:  0.9084 ± 0.0192 [0.8849, 0.9299]

 Detailed results saved to: cv_summary_statistics.json
 CSV results saved to: cv_results.csv
 Visualization saved as: cv_results_summary.png
 Confusion matrix saved for best_resnet fold 3
```


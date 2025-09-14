"""
Simple runner script for Brain Tumor MRI Cross-Validation
Usage: python run_cv.py
"""

import sys
import os
from cross_validation import BrainTumorCrossValidator, main

def check_requirements():
    """Check if all required files and directories exist."""
    print(" Checking requirements...")
    
    # Check if dataset exists
    if not os.path.exists("Dataset"):
        print(" Error: Dataset directory not found!")
        print("   Please run create_smaller_dataset.py first to create the dataset.")
        return False
    
    # Check if at least one model exists
    model_files = [
        #"modelResNet50_43.keras",
        "modelVGG16_43.keras"
    ]
    
    existing_models = [f for f in model_files if os.path.exists(f)]
    if not existing_models:
        print(" Error: No pre-trained models found!")
        print("   Please ensure at least one of these files exists:")
        for f in model_files:
            print(f"   - {f}")
        return False
    
    print(f" Found {len(existing_models)} model(s): {', '.join(existing_models)}")
    print(" Dataset directory found")
    return True

def run_cross_validation():
    """Run the cross-validation with proper setup."""
    print(" Starting Brain Tumor MRI Cross-Validation")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Run cross-validation
    try:
        main()
        print("\n Cross-validation completed successfully!")
    except KeyboardInterrupt:
        print("\n  Cross-validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Cross-validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_cross_validation()

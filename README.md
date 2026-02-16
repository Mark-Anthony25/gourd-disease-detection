# Gourd Disease Detection - Improved Version

A hierarchical deep learning system for detecting crop types and diseases in gourd plants using transfer learning and out-of-distribution detection.

## 🎯 Overview

This project implements a two-stage classification system:
1. **Crop Classification**: Identifies the type of crop (e.g., Bitter Gourd, Ridge Gourd, Okra)
2. **Disease Classification**: Identifies specific diseases for each identified crop
3. **OOD Detection**: Detects unknown diseases or misclassified crops using energy-based methods

## 📁 Repository Structure

```
gourd-disease-detection/
├── disease-detection.ipynb           # Original notebook (for reference)
├── disease-detection-improved.ipynb  # ✨ Improved version with all fixes
├── IMPROVEMENTS.md                   # Detailed documentation of changes
└── README.md                         # This file
```

## 🚀 What's New in the Improved Version

### Major Improvements

✅ **Proper Augmentation Handling**
- Dataset already includes comprehensive offline augmentation
- Removed redundant online augmentation to avoid double augmentation
- Uses pre-augmented dataset as intended by dataset creators

✅ **Explicit Normalization**
- Added Rescaling layer (1./255.0) for consistent preprocessing

✅ **Training Callbacks**
- EarlyStopping: Prevents overfitting
- ReduceLROnPlateau: Adaptive learning rate
- ModelCheckpoint: Saves best model weights

✅ **Class Weighting**
- Automatic computation to handle imbalanced datasets
- Ensures all classes contribute fairly to training

✅ **Comprehensive Metrics**
- Accuracy, Precision, Recall, F1-Score
- Better understanding of model performance

✅ **Clean Code**
- Removed all duplicate functions
- Added comprehensive documentation
- 17 markdown sections for clear navigation
- Configurable paths (no hardcoding)

✅ **Better Architecture**
- Added BatchNormalization layers
- Optimized dropout rates
- Increased batch size (16 → 32)

## 🔍 Issues Addressed

### Critical Issues Fixed

1. **Unrealistically High Accuracy (97-100%)**
   - Added warnings when accuracy > 95%
   - Recommendations for validation with external data
   - Improved metrics beyond just accuracy

2. **Proper Data Augmentation Strategy**
   - Recognized dataset already contains 22,825 pre-augmented images
   - Removed redundant online augmentation to prevent double augmentation
   - Uses pre-augmented dataset correctly (5x augmentation ratio already applied)

3. **Missing Training Safeguards**
   - Added callbacks to prevent overfitting
   - Model checkpointing to save best weights
   - Learning rate scheduling for better convergence

4. **Imbalanced Dataset**
   - Automatic class weight computation
   - Fair training across all classes

5. **Code Quality**
   - Removed duplicate code (cells 10 & 11 were identical)
   - Consolidated repeated functions
   - Added proper documentation

## 📊 Model Architecture

### Crop Classification Model
```
Input (224×224×3)
    ↓
EfficientNetB0 (ImageNet pre-trained)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(64/128/256, relu) [tuned]
    ↓
Dropout(0.3-0.6) [tuned]
    ↓
Dense(num_crops, softmax)
```

### Disease Classification Model (per crop)
```
Input (224×224×3)
    ↓
EfficientNetB0/MobileNetV3Large (frozen)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(128, relu)
    ↓
Dropout(0.5)
    ↓
Dense(num_diseases, softmax)
```

## 🛠️ Installation & Usage

### Requirements
```bash
pip install tensorflow>=2.10.0
pip install keras-tuner
pip install scikit-learn
pip install scipy
pip install numpy
```

### Running the Improved Notebook

1. Open `disease-detection-improved.ipynb` in Jupyter or Kaggle
2. Update the paths in the configuration section (Cell 1) if needed
3. Run all cells sequentially
4. Review warnings and recommendations in the output

### Using Your Own Data

Update the configuration section:
```python
BASE_PATH = "/path/to/your/dataset"
RAW_DATA = os.path.join(BASE_PATH, "Raw Data")
AUG_DATA = os.path.join(BASE_PATH, "Augmented Data")
WORKING_DIR = "/path/to/working/directory"
```

Expected structure:
```
Dataset/
├── Raw Data/
│   ├── Crop1/
│   │   ├── Disease1/
│   │   │   ├── image1.jpg
│   │   │   └── image2.jpg
│   │   └── Disease2/
│   └── Crop2/
└── Augmented Data/
    ├── Crop1/
    │   └── Disease1/
    │       ├── aug_image1.jpg
    │       └── aug_image2.jpg
    └── Crop2/
```

## ⚠️ Important Notes

### Why Model Results Might Feel Unrealistic

The original model showed 97-100% accuracy, which is suspicious for real-world agricultural data. This could indicate:

1. **Data Leakage**: Train/test images too similar
2. **Limited Diversity**: All images from similar conditions
3. **Small Dataset**: Not enough samples to capture complexity
4. **Controlled Environment**: Lab images, not field images

### Recommendations for Production

1. **Test with External Data**
   - Images from different cameras
   - Different lighting conditions
   - Different locations and backgrounds

2. **Implement Confidence Thresholds**
   ```python
   if confidence < 0.8:
       return "Low confidence - needs human review"
   ```

3. **Monitor Performance**
   - Track accuracy on real-world data
   - Compare with expert assessments
   - Retrain regularly

4. **Start Conservative**
   - Begin with human verification
   - Gradually increase automation
   - Always flag low-confidence predictions

## 📈 Results

### Crop Model Performance
- Hyperparameter optimization with Keras Tuner
- Early stopping to prevent overfitting
- Test accuracy reported with warnings if > 95%

### Disease Model Performance
- Trained separately for each crop
- Both EfficientNetB0 and MobileNetV3Large tested
- Metrics: Accuracy, Precision, Recall, F1-Score
- Class weights applied for imbalanced data

### OOD Detection
- Energy-based method for unknown disease detection
- Calibrated thresholds using validation set
- Cross-crop testing to verify OOD rejection

## 📚 Documentation

- **IMPROVEMENTS.md**: Detailed explanation of all changes and improvements
- **Notebook Comments**: Comprehensive inline documentation
- **Docstrings**: All functions have detailed docstrings

## 🤝 Contributing

Suggestions for improvement:
1. Collect more diverse data (different conditions, locations, cameras)
2. Implement cross-validation for robust evaluation
3. Add confusion matrix analysis
4. Implement model ensembling
5. Add real-time inference optimization
6. Create deployment pipeline

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

- Original implementation insights
- TensorFlow and Keras teams
- EfficientNet and MobileNet architectures
- Kaggle dataset providers

## 📧 Contact

For questions or issues:
1. Review the IMPROVEMENTS.md file
2. Check the notebook documentation
3. Test with your own data
4. Open an issue if problems persist

---

**Note**: This is an improved version addressing code quality and best practices. However, the fundamental challenge of dataset diversity for production deployment remains. Always validate with real-world data before production use.
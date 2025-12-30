# Fetal Ultrasound Cranium Segmentation & Biometry

Deep learning-based automated detection and measurement of Biparietal Diameter (BPD) and Occipitofrontal Diameter (OFD) from fetal ultrasound images.

##  Overview

This project implements U-Net based semantic segmentation models to identify fetal cranium in ultrasound images and automatically compute biometric measurements (BPD & OFD). The system uses ellipse fitting for robust measurement extraction from segmented masks.

### Key Features
- Multiple U-Net architectures (Standard, Reduced Channels, Attention U-Net)
- Ultrasound-specific data augmentation pipeline
- Automated biometry computation using ellipse completion
- Comprehensive evaluation metrics (Jaccard, F1, Precision, Recall)
- Visual result generation with measurement overlays

## Task Description

**Objective**: Develop an algorithm to identify BPD and OFD landmark points (2 per biometry) in fetal ultrasound images.

**Medical Context**:
- Ultrasound is the primary method for detecting fetal CNS anomalies
- BPD and OFD are critical measurements for assessing fetal growth
- Early detection enables timely clinical intervention

**Approach**: Segmentation-based method
1. Train deep learning model for cranium segmentation
2. Use computer vision algorithms to extract biometry points from segmented masks
3. Apply ellipse fitting for robust measurement extraction

## Project Structure

```
.
├── dataset/                    # Original dataset
│   ├── train/
│   │   ├── images/
│   │   └── mask/
│   ├── validation/
│   │   ├── images/
│   │   └── mask/
│   └── test/
│       ├── images/
│       └── mask/
├── augmented_data/            # Augmented dataset (4x)
│   ├── train/
│   ├── validation/
│   └── test/
├── files/                     # Model checkpoints
│   ├── checkpoint_model1.pth
│   ├── checkpoint_model2.pth
│   └── checkpoint_model3.pth
├── results/                   # Test predictions & visualizations
├── model.py                   # U-Net architecture (standard)
├── model2.py                  # U-Net with reduced channels
├── model_attention.py         # Attention U-Net
├── data.py                    # Dataset class
├── loss.py                    # Loss functions (Dice, DiceBCE)
├── utils.py                   # Helper functions
├── data_aug.py                # Data augmentation pipeline
├── train.py                   # Training script
├── test.py                    # Evaluation & visualization
├── biometry.py                # BPD/OFD computation
└── README.md
```

## 🔧 Installation

### Requirements
```bash
Python >= 3.8
PyTorch >= 1.10
CUDA (for GPU training)
```

### Dependencies
```bash
pip install torch torchvision
pip install opencv-python
pip install albumentations
pip install scikit-learn
pip install tqdm
pip install numpy
```

## Models & Performance

### Model Architectures

| Model | Channels | Parameters | FPS | Jaccard | F1 Score |
|-------|----------|------------|-----|---------|----------|
| **U-Net (Standard)** | 64-128-256-512-1024 | High | 153.19 | 0.1621 | 0.2753 |
| **U-Net (Reduced)** | 32-64-128-256-512 | Medium | 260.77 | 0.1588 | 0.2698 |
| **Attention U-Net** | 64-128-256-512 | Medium | 231.92 | 0.1497 | 0.2577 |

### Best Model: U-Net (Standard)
```
✓ Jaccard:    0.1621
✓ F1 Score:   0.2753
✓ Recall:     0.5036
✓ Precision:  0.1905
✓ Accuracy:   0.9791
✓ FPS:        153.19
```

## Usage

### 1. Data Augmentation
Generate 4x augmented dataset with ultrasound-specific transformations:

```bash
python data_aug.py
```

**Augmentation Strategy** (4 combinations):
1. Original (no augmentation)
2. Horizontal Flip + CLAHE (probe orientation + contrast)
3. Brightness/Contrast + Horizontal Flip (gain/TGC variations)
4. Speckle Noise + CLAHE (noise + contrast enhancement)

### 2. Training

```bash
python train.py
```

**Hyperparameters**:
- Image Size: 512×512
- Batch Size: 2
- Epochs: 50
- Learning Rate: 1e-4
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau (patience=5)
- Loss: DiceBCELoss (Dice + Binary Cross Entropy)

**Training Features**:
- Automatic checkpoint saving (best validation loss)
- Learning rate scheduling
- Mixed Dice-BCE loss for better boundary detection

### 3. Testing & Evaluation

```bash
python test.py
```

**Outputs**:
- Segmentation metrics (Jaccard, F1, Precision, Recall, Accuracy)
- Visual results: `results/` folder
  - Original image | Ground truth | Prediction | Biometry overlay
- BPD/OFD measurements in pixels
- Inference speed (FPS)

## Biometry Computation

The `biometry.py` module implements robust measurement extraction:

### Pipeline
```
Segmentation mask → Morphological closing → Contour extraction → 
Ellipse fitting → BPD/OFD landmark extraction
```

### Key Steps
1. **Mask Preprocessing**: Binary thresholding + morphological closing (ellipse kernel 11×11)
2. **Contour Validation**: Area > 1000 pixels, points > 20
3. **Ellipse Fitting**: `cv2.fitEllipse()` for skull shape completion
4. **Measurement Extraction**:
   - **BPD**: Minor axis diameter (left-right parietal distance)
   - **OFD**: Major axis diameter (front-back frontal-occipital distance)

### Visualization
- Yellow: Fitted ellipse (completed skull contour)
- Green: BPD line and endpoints
- Blue: OFD line and endpoints

## Loss Functions

### Dice Loss
```python
Dice = 1 - (2 × |X ∩ Y|) / (|X| + |Y|)
```

### DiceBCE Loss (Used)
```python
Loss = BCE(pred, target) + DiceLoss(pred, target)
```

**Benefits**:
- BCE: Pixel-wise classification accuracy
- Dice: Overlap optimization for segmentation
- Combined: Better boundary detection + region accuracy

## Data Augmentation Details

### Training Augmentations (4x dataset)
1. **Horizontal Flip + CLAHE**
   - Simulates probe orientation variations
   - Enhances contrast (clip_limit=2.5)

2. **Brightness/Contrast + Horizontal Flip**
   - Simulates TGC (Time Gain Compensation) variations
   - Range: ±20% brightness/contrast

3. **Speckle Noise + CLAHE**
   - Adds realistic ultrasound noise (Gaussian std=0.01-0.05)
   - Enhances contrast for robustness

### Validation/Test
- No augmentation (resize to 512×512 only)
- Preserves original image characteristics for fair evaluation

## Model Architectures

### 1. Standard U-Net (`model.py`)
- **Encoder**: 64 → 128 → 256 → 512
- **Bottleneck**: 1024
- **Decoder**: 512 → 256 → 128 → 64
- **Output**: 1 channel (binary mask)

### 2. Reduced U-Net (`model2.py`)
- **Encoder**: 32 → 64 → 128 → 256
- **Bottleneck**: 512 + Dropout(0.5)
- **Decoder**: 256 → 128 → 64 → 32
- **Benefits**: Faster inference (260 FPS), lower memory

### 3. Attention U-Net (`model_attention.py`)
- Attention gates at each decoder level
- Focus on relevant cranium features
- Better feature localization

##  Training Tips

### Best Practices
1. **Monitor validation loss**: Model saves best checkpoint automatically
2. **Learning rate**: Reduces on plateau (patience=5 epochs)
3. **Early stopping**: Stop if validation loss doesn't improve for 10+ epochs
4. **Data quality**: Ensure masks are clean binary images (0/255)

### Hyperparameter Tuning
- Increase batch size (4-8) if GPU memory allows
- Try different learning rates (1e-3, 5e-4, 1e-5)
- Experiment with loss functions (Dice, BCE, Focal, Tversky)

## Troubleshooting

### Common Issues

**1. Biometry computation fails**
- Mask too small (area < 1000 pixels)
- Insufficient contour points (< 20)
- Solution: Improve segmentation quality or adjust thresholds

**2. Low Jaccard/F1 scores**
- Dataset imbalance (too much background)
- Insufficient training epochs
- Solution: Use weighted loss or train longer

**3. CUDA out of memory**
- Reduce batch size to 1
- Use reduced channel model (`model2.py`)
- Enable gradient checkpointing

## Dataset Requirements

### Expected Format
```
dataset/
├── train/
│   ├── images/        # Grayscale ultrasound images (.png, .jpg)
│   └── mask/          # Binary masks (0=background, 255=cranium)
├── validation/
└── test/
```

### Data Specifications
- **Input**: Grayscale ultrasound images (any size, resized to 512×512)
- **Masks**: Binary segmentation (0/255), same filename as images
- **Format**: PNG or JPG
- **Channels**: 1 (grayscale)

## References

### Papers
1. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
2. **Attention U-Net**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas"

### Medical Context
- BPD/OFD measurements are standard for fetal biometry
- Ellipse fitting is clinically validated for skull shape approximation
- Early detection of CNS anomalies improves neonatal outcomes




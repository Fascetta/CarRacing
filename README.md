# Image Classification for Car Racing in a Gym Environment

## Overview

This project explores the application of convolutional neural networks (CNNs) to classify car control actions in a simulated racing environment. Two CNN architectures were developed and evaluated using a dataset of 96×96 color images representing five distinct action classes.

Key goals:
- Design and implement CNN models for classifying car actions.
- Compare the performance of a baseline architecture and an expanded architecture with regularization.
- Analyze model behavior through metrics like accuracy, precision, recall, F1-score, and confusion matrices.

---

## Dataset

- **Source**: Simulated car racing in a Gym environment.
- **Structure**: Training and testing datasets organized into subdirectories for each class.
- **Resolution**: Images are 96×96 pixels.
- **Classes**: Five distinct car actions corresponding to control commands.

### Data Processing
1. **Loading**: TensorFlow's `image_dataset_from_directory` function infers class labels from subdirectory names.
2. **Normalization**: Images are normalized to the range [0, 1] to improve convergence.
3. **Batching**: Mini-batches of 16 samples optimize memory usage.

---

## Model Architectures

### Model 1: Simpler Baseline
- **Feature Extraction**:
  - Conv2D (16 filters, kernel: 3×3, ReLU), MaxPooling2D (pool: 2×2).
  - Conv2D (32 filters, kernel: 3×3, ReLU), MaxPooling2D (pool: 2×2).
- **Classification**:
  - Flatten layer to convert 2D maps to 1D vector.
  - Dense layer (64 units, ReLU), Dropout (0.3).
  - Dense output layer (5 units, Softmax).

### Model 2: Expanded Architecture with Regularization
- **Feature Extraction**:
  - Block 1: Two Conv2D layers (32 filters each, kernel: 3×3, ReLU, L2 regularization), MaxPooling2D (pool: 2×2).
  - Block 2: Two Conv2D layers (64 filters each, kernel: 3×3, ReLU, L2 regularization), MaxPooling2D (pool: 2×2).
  - Block 3: Conv2D (128 filters, kernel: 3×3, ReLU), MaxPooling2D (pool: 2×2).
- **Classification**:
  - Flatten layer to convert 2D maps to 1D vector.
  - Dense layer (256 units, ReLU, L2 regularization), Dropout (0.5).
  - Dense output layer (5 units, Softmax).

---

## Training Procedure

- **Hyperparameters**:
  - Learning Rate: `1×10^-4`.
  - Loss Function: Sparse Categorical Crossentropy.
  - Optimizer: Adam.
  - Epochs: 50.

- **Callbacks**:
  - TensorBoard: Logs training and validation metrics.
  - EarlyStopping: Stops training after 5 epochs of no improvement in validation loss.
  - ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.

---

## Results and Discussion

### Performance Metrics

| Model  | Accuracy | Loss  | Notes                                  |
|--------|----------|-------|----------------------------------------|
| Model 1 | 67%      | 0.94  | Faster convergence but limited capacity. |
| Model 2 | 68%      | 1.05  | Better accuracy but higher penalized loss. |

### Observations
- **Class Imbalance**: Both models favored the majority class, struggling with underrepresented classes.
- **Model Complexity**: Model 2 learned more nuanced features due to deeper architecture and L2 regularization but exhibited slightly higher misclassification penalties.

---

## Future Work

1. **Data Augmentation**:
   - Apply random flips, rotations, and zooms to improve class balance.
2. **Hyperparameter Tuning**:
   - Optimize learning rates, L2 regularization coefficients, and batch sizes.
3. **Advanced Architectures**:
   - Explore deeper CNNs or transfer learning (e.g., ResNet, VGG).
4. **Ensemble Techniques**:
   - Combine multiple models to address class imbalance and reduce overfitting.

---

## Citation

If you reference this work, please cite:

```bibtex
@misc{bianchi2024carracing,
  title={Image Classification for Car Racing in a Gym Environment},
  author={Christian Bianchi},
  year={2024}
}

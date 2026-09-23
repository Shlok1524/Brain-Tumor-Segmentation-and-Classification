# Comparative Analysis of Brain Tumor Detection Using MONAI and CNN with Explainable AI

A deep-learning project that compares **2D CNN-based brain tumor classification** with **3D MONAI U-Net-based tumor segmentation**, while using **SHAP, LIME, and Grad-CAM** to improve the interpretability of CNN predictions.

> **Project focus:** automated brain MRI analysis, tumor classification, tumor-region segmentation, and explainable AI.

---

## 📌 Overview

Brain tumor detection from MRI scans can require detailed manual examination by medical professionals. This project explores two complementary deep-learning approaches:

1. **CNN + XAI — Classification**
   - Classifies 2D grayscale MRI images as **tumor** or **non-tumor**.
   - Uses SHAP, LIME, and Grad-CAM to visualize regions contributing to predictions.

2. **MONAI + U-Net — Segmentation**
   - Processes 3D volumetric MRI data in NIfTI format.
   - Produces a pixel/voxel-level **tumor segmentation mask**.
   - Uses a 3D U-Net architecture implemented with MONAI.

Our experiments achieved a CNN validation accuracy of **93.33%**, a CNN test accuracy of **93.3%**, and a MONAI segmentation Dice score of **0.82** on the test set. The MONAI model achieved a Dice score of **0.87** during training.

---

## ✨ Key Features

- 🧠 Brain MRI tumor classification
- 🧩 3D brain tumor segmentation
- 🔬 MONAI-based medical image processing
- 🧠 Custom TensorFlow/Keras CNN
- 🔍 SHAP explanations
- 🗺️ LIME explanations
- 🔥 Grad-CAM visualization
- 📊 Accuracy, precision, recall, F1-score and Dice evaluation
- 📈 Training/validation loss and metric visualization
- 💾 Model checkpointing
- ⏹️ Early stopping for CNN training
- 🖥️ GPU-enabled training with CUDA/PyTorch where available

---

## 🏗️ Project Architecture

```text
                         Brain MRI Scans
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
           2D MRI Images              3D MRI Volumes
           JPEG / PNG                  NIfTI (.nii.gz)
                 │                           │
          Preprocessing                 Preprocessing
                 │                           │
        CNN Classification          MONAI 3D U-Net
                 │                           │
       Tumor / Non-Tumor              Tumor Segmentation
                 │                           │
          ┌──────┼──────┐                    │
          │      │      │                    │
        SHAP    LIME  Grad-CAM         Segmentation Mask
          │      │      │                    │
          └──────┴──────┘                    │
                 │                           │
          Explainable Output          Localized Tumor
```

The project workflow takes MRI scans through preprocessing and then into either the MONAI segmentation pipeline or CNN classification pipeline, with XAI applied to CNN predictions.

---

## 📂 Repository Structure

A recommended repository structure is:

```text
Brain-Tumor-Detection/
│
├── Classification (CNN).ipynb
├── Segmentation (MONAI).ipynb
├── README.md
│
├── models/
│   ├── brain_tumor_checkpoint.h5
│   └── best_metric_model.pth
│
├── results/
│   ├── training_history.pkl
│   ├── loss_train.npy
│   ├── metric_train.npy
│   ├── loss_test.npy
│   └── metric_test.npy
│
└── data/
    ├── classification/
    │   ├── tumor/
    │   └── non-tumor/
    │
    └── segmentation/
        ├── TrainImages/
        ├── TrainLabels/
        ├── TestImages/
        └── TestLabels/
```

> The notebooks currently contain local Windows dataset paths. Update these paths before running the notebooks on another machine.

---

# 📂 Datasets

This project uses two publicly available datasets, with a separate dataset selected for each modeling approach.

## 🧠 Classification Dataset — Br35H Brain Tumor Detection

The CNN classification pipeline uses the **Br35H Brain Tumor Detection** dataset available on Kaggle.

**Source:** Ahmed Hamada — Brain Tumor Detection (Br35H)

🔗 https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

The dataset provides brain MRI images for **tumor/non-tumor classification** and is used as the image-level classification dataset for the CNN pipeline. 

The images are processed as 2D grayscale inputs before being passed to the CNN.

### Used for

- Binary brain tumor classification
- CNN training and validation
- CNN test evaluation
- SHAP, LIME, and Grad-CAM explanations

---

## 🧩 Segmentation Dataset — Medical Segmentation Decathlon

The MONAI segmentation pipeline uses **Task01_BrainTumour** from the **Medical Segmentation Decathlon (MSD)**.

**Source:** Medical Segmentation Decathlon

🔗 http://medicaldecathlon.com/

The Brain Tumours task contains **3D multimodal MRI volumes** with segmentation annotations. The official dataset describes the task as glioma segmentation involving necrotic/active tumor and edema, with data derived from the BraTS 2016 and 2017 datasets.

### Used for

- 3D brain MRI processing
- MONAI-based segmentation
- 3D U-Net training
- Tumor-region localization
- Dice-based segmentation evaluation

### Dataset Format

The Medical Segmentation Decathlon dataset follows a structure containing image volumes and corresponding training labels:

```text
Task01_BrainTumour/
├── dataset.json
├── imagesTr/
├── imagesTs/
└── labelsTr/
```

The segmentation data is provided as volumetric medical images and is suitable for 3D semantic segmentation workflows.

> **Note:** The classification and segmentation pipelines use different datasets because they address different tasks and require different forms of ground-truth annotation: image-level labels for classification and voxel-level segmentation masks for segmentation.

---

# 🧠 1. CNN Classification Pipeline

## Dataset

The CNN pipeline uses **2D grayscale JPEG images** organized by class.

The notebook expects a directory structure compatible with Keras `flow_from_directory()`:

```text
Final Dataset/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── class_2/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

A separate prediction directory is used for inference.

## Preprocessing

The CNN notebook uses:

- Resize to **150 × 150**
- Grayscale input
- Pixel normalization using `1/255`
- 20% validation split
- Random rotation up to 15°
- Width and height shifts
- Zoom augmentation
- Horizontal flipping

The validation generator uses normalization and the validation split without the training augmentations.

## CNN Architecture

The implemented CNN contains:

```text
Input: 150 × 150 × 1
        │
   Conv2D (32)
        │
   MaxPooling
        │
   Conv2D (64)
        │
   MaxPooling
        │
      Flatten
        │
   Dense (128)
        │
    Dropout (0.5)
        │
   Sigmoid Output
```

Training configuration in the notebook:

- Optimizer: **Adam**
- Learning rate: **0.001**
- Loss: **Binary Cross-Entropy**
- Batch size: **16**
- Activation: **ReLU**
- Output activation: **Sigmoid**
- Early stopping based on validation loss
- Model checkpointing based on validation accuracy

---

# 🔍 2. Explainable AI

The CNN predictions are analyzed using three XAI methods.

### SHAP

**SHAP (Shapley Additive Explanations)** estimates the contribution of image features toward the model output.

The notebook uses a subset of training images as the SHAP background dataset.

### LIME

**LIME (Local Interpretable Model-Agnostic Explanations)** perturbs image regions and examines how those changes affect the prediction.

The implementation uses image segmentation with `quickshift` and visualizes the most influential regions.

### Grad-CAM

**Grad-CAM (Gradient-weighted Class Activation Mapping)** produces a heatmap highlighting image regions associated with the CNN prediction.

Together, these techniques provide visual explanations rather than only returning a binary prediction.

---

# 🧩 3. MONAI Segmentation Pipeline

## Dataset

The MONAI pipeline uses **3D NIfTI (`.nii.gz`) MRI volumes** with corresponding segmentation labels.

Expected structure:

```text
Final/
├── TrainImages/
│   ├── *.nii.gz
│   └── ...
├── TrainLabels/
│   ├── *.nii.gz
│   └── ...
├── TestImages/
│   ├── *.nii.gz
│   └── ...
└── TestLabels/
    ├── *.nii.gz
    └── ...
```

Each image is paired with its corresponding segmentation label.

## Preprocessing

The MONAI notebook performs:

1. Load NIfTI images and labels
2. Ensure channel-first format
3. Resample voxel spacing to:

```text
(1.5, 1.5, 1.0)
```

4. Reorient volumes to **RAS**
5. Scale image intensity from `[0, 1000]` to `[0, 1]`
6. Crop foreground
7. Resize volumes to:

```text
128 × 128 × 128
```

8. Convert the image to a single-channel representation
9. Convert data to tensors

The notebook supports both `Dataset` and `CacheDataset`.

---

## MONAI U-Net

The segmentation model is a **3D U-Net** configured as:

```text
spatial_dims = 3
in_channels  = 1
out_channels = 2

channels = (16, 32, 64, 128, 256)
strides  = (2, 2, 2, 2)
num_res_units = 2
normalization = BatchNorm
```

The model is trained using:

- Loss: **DiceCELoss**
- Optimizer: **Adam**
- Learning rate: **1e-5**
- Weight decay: **1e-5**
- AMSGrad: enabled
- Learning-rate scheduler: `ReduceLROnPlateau`

The notebook's training call is configured for **50 epochs**.

---

# 🧪 4. Inference

MONAI inference uses:

```text
Sliding Window Inference
ROI size: 128 × 128 × 128
Sliding-window batch size: 5
```

The segmentation output is passed through sigmoid activation and thresholded at **0.70** in the notebook's displayed inference workflow.

The notebook visualizes:

```text
MRI Slice | Ground Truth Label | Predicted Segmentation
```

across the volume.

---

# 📊 Results

The following results were obtained during our experiments and are documented in our research work.

## CNN Classification

| Metric | Result |
|---|---:|
| Training Accuracy | 94.13% |
| Validation Accuracy | 93.33% |
| Final Training Loss | 0.1566 |
| Final Validation Loss | 0.2088 |
| Test Accuracy | 93.3% |
| Precision | 93.4% |
| Recall / Sensitivity | 93.2% |
| F1-Score | 93.3% |

Early stopping was triggered around the **55th epoch** during CNN training to help prevent overfitting.

## MONAI Segmentation

| Metric | Result |
|---|---:|
| Reported training Dice | 0.87 |
| Reported test Dice | 0.82 |

Our comparative analysis includes CNN sensitivity, specificity, and precision alongside MONAI segmentation metrics.

---

# ⚖️ CNN vs MONAI

| Aspect | CNN + XAI | MONAI |
|---|---|---|
| Task | Classification | Segmentation |
| Input | 2D grayscale JPEG/PNG | 3D NIfTI MRI |
| Output | Tumor / non-tumor | Tumor segmentation mask |
| Architecture | Custom 2D CNN | 3D U-Net |
| Labels | Image-level labels | Pixel/voxel-level masks |
| XAI | SHAP, LIME, Grad-CAM | Segmentation visualization |
| Computational Cost | Lower | Higher |
| Training Complexity | Lower | Higher |
| Localization | Indirect through XAI | Direct segmentation |

These approaches solve different parts of the brain-tumor detection problem rather than being identical models performing the same task.

---

# 🛠️ Tech Stack

### Deep Learning

- Python
- TensorFlow
- Keras
- PyTorch
- MONAI

### Medical Imaging

- NIfTI
- NiBabel
- DICOM/NIfTI conversion utilities
- MONAI transforms

### Computer Vision

- OpenCV
- NumPy
- Matplotlib

### Explainable AI

- SHAP
- LIME
- Grad-CAM

### Model Evaluation

- Accuracy
- Precision
- Recall / Sensitivity
- F1-score
- Dice coefficient

---

# ⚙️ Installation

Create a Python environment and install the required libraries:

```bash
pip install torch torchvision torchaudio
pip install tensorflow
pip install monai
pip install nibabel
pip install dicom2nifti
pip install opencv-python
pip install numpy matplotlib
pip install shap lime scikit-image
```

Depending on your CUDA configuration, install the appropriate PyTorch build from the official PyTorch installation instructions.

---

# 🚀 How to Run

## 1. CNN Classification

Open:

```text
Classification (CNN).ipynb
```

Update:

```python
train_dir = "YOUR_PATH/Final Dataset"
test_dir = "YOUR_PATH/pred"
```

Then run the notebook sequentially.

The notebook can:

- Train the CNN
- Save/load model checkpoints
- Plot training history
- Predict individual images
- Predict batches of images
- Generate SHAP and LIME explanations

---

## 2. MONAI Segmentation

Open:

```text
Segmentation (MONAI).ipynb
```

Update:

```python
data_dir = "YOUR_PATH/Final"
model_dir = "YOUR_PATH/Results"
```

Then execute the notebook in order:

```text
Preparing data
      ↓
Preprocessing
      ↓
Training
      ↓
Testing
      ↓
Output visualization
```

For GPU training, verify:

```python
torch.cuda.is_available()
```

before starting training.

---

# 💾 Saved Models and Results

### CNN

The notebook saves the best CNN checkpoint as:

```text
brain_tumor_checkpoint.h5
```

Training history is stored as:

```text
training_history.pkl
```

### MONAI

The segmentation pipeline stores:

```text
best_metric_model.pth
loss_train.npy
metric_train.npy
loss_test.npy
metric_test.npy
```

---

# 📈 Visualizations

The project generates:

### CNN

- Training vs validation accuracy
- Training vs validation loss
- SHAP explanations
- LIME explanations
- Grad-CAM heatmaps
- Tumor/non-tumor predictions

### MONAI

- Training Dice
- Training loss
- Test Dice
- Test loss
- MRI slices
- Ground-truth segmentation
- Predicted segmentation

---

# 🔬 Research Context

This project is documented in our research paper:

**"Comparative Analysis of Brain Tumor Detection Using MONAI and CNN with Explainable AI"**

Our research focuses on combining automated brain tumor detection with model interpretability. We compare a CNN-based classification pipeline with a MONAI-based segmentation pipeline and apply XAI techniques to understand the CNN predictions.

---

# 🔮 Future Work

Based on our findings, the following extensions can be explored:

- Combine classification and segmentation into a unified system
- Explore 3D CNN architectures
- Train on larger and more diverse datasets
- Improve detection of difficult-to-identify regions
- Further integrate explainability
- Explore federated learning for improved privacy
- Incorporate cross-modality medical imaging data

---

# ⚠️ Limitations & Disclaimer

This repository is intended for **research and educational purposes**.

The models and results described here should **not be used as a standalone clinical diagnostic system**. Medical diagnosis requires qualified healthcare professionals and appropriate clinical validation.

The reported performance was obtained using the datasets and experimental setup used in this project and should not be interpreted as a guarantee of performance on other clinical populations or imaging systems.

---

# 👥 Authors

- **Shlok Swetalkumar Pastagia** — SCOPE, VIT Bhopal University
- **Yash Dharmendrabhai Patel** — SCOPE, VIT Bhopal University
- **Ishan Vijay Sharma** — SCOPE, VIT Bhopal University
- **Diya Madhavbhai Vyas** — SCOPE, VIT Bhopal University
- **Palak Narang** — SCOPE, VIT Bhopal University
- **Sasmita Padhy** — SCOPE, VIT Bhopal University
- **Naween Kumar** — SCSET, Bennett University

---

# 📚 References

Key references used in our research include:

1. Pereira et al., *Brain tumor segmentation using convolutional neural networks in MRI images*, IEEE Transactions on Medical Imaging, 2016.
2. Havaei et al., *Brain tumor segmentation with deep neural networks*, Medical Image Analysis, 2017.
3. Ronneberger, Fischer & Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI, 2015.
4. MONAI Consortium, *MONAI: Medical Open Network for AI*, 2020.
5. Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS, 2017.
6. Ribeiro, Singh & Guestrin, *Why Should I Trust You? Explaining the Predictions of Any Classifier*, KDD, 2016.
7. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*, ICCV, 2017.

---

## ⭐ Summary

This project demonstrates two complementary approaches to automated brain tumor analysis:

```text
CNN + XAI
    → Fast 2D tumor classification
    → SHAP + LIME + Grad-CAM explanations

MONAI + 3D U-Net
    → Volumetric MRI processing
    → Pixel/voxel-level tumor segmentation
```

The combination provides a practical research framework for studying **accuracy, localization, and interpretability in deep-learning-based medical image analysis**.

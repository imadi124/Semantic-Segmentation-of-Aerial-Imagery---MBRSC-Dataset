# Semantic Segmentation of Aerial Imagery

## Project Overview

This project implements **pixel-wise semantic segmentation** on high-resolution aerial imagery of Dubai captured by MBRSC satellites. The goal is to classify every pixel into one of six land-cover classes using a custom U-Net architecture trained from scratch.

**Classes and Hex Colors:**

| Class              | Hex Color  | Description                  |
|--------------------|------------|------------------------------|
| Building           | `#3C1098`  | Built-up structures          |
| Land (unpaved)     | `#8429F6`  | Bare / unpaved soil          |
| Road               | `#6EC1E4`  | Roads and paved surfaces     |
| Vegetation         | `#FEDD3A`  | Trees, grass, crops          |
| Water              | `#E2A929`  | Water bodies                 |
| Unlabeled          | `#9B9B9B`  | Background / undefined       |

**Dataset Source:** [Kaggle – Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)

The original dataset contains 8 large tiles of varying sizes. Each tile is cropped to the nearest multiple of 256 and then divided into non-overlapping 256×256 patches using `patchify`.

**Final patch count:** 1,305 patches of size `256 × 256 × 3`

---

## Project Structure
```
Semantic segmentation dataset/
├── Tile 1/
│   ├── images/
│   └── masks/
├── Tile 2/
│   ├── images/
│   └── masks/
...
└── Tile 8/
├── images/
└── masks/
```


- Images are RGB `.jpg`
- Masks are RGB `.png` (color-coded)

---

## Pipeline Summary

### 1. Data Preparation

- Walk through all `images` and `masks` folders.
- Crop each image/mask to the largest size divisible by 256.
- Extract non-overlapping 256×256 patches (`step=256`).
- Normalize image patches with `MinMaxScaler` (range [0, 1]).
- Convert mask colors to integer class labels (0–5) and one-hot encode them.

### 2. Model Architecture

A custom **U-Net** with the following characteristics:

- Encoder–decoder with skip connections
- Progressive filter sizes: 16 → 32 → 64 → 128 → 256
- Dropout layers after each convolutional block
- Transposed convolutions for upsampling
- Final 1×1 convolution with 6 filters + softmax

**Total parameters:** 1,941,190 (≈ 7.41 MB)  
All parameters are trainable.

### 3. Training Configuration

- **Loss:** Categorical Cross-Entropy + Dice Loss (or Jaccard Loss)
- **Optimizer:** Adam
- **Metrics:** Accuracy, Jaccard Coefficient (IoU)
- **Epochs:** 100
- **Batch size:** typically 8–16 (depending on GPU memory)
- **Hardware:** Google Colab T4 GPU

### 4. Evaluation Metrics

After 100 epochs the model reaches approximately:

- **Training IoU** ≈ 0.78
- **Validation IoU** ≈ 0.75
- **Training Loss** ≈ 0.88
- **Validation Loss** ≈ 0.90

**Per-class IoU (test set):**

| Class       | IoU  |
|-------------|------|
| Building    | 0.70 |
| Land        | 0.83 |
| Road        | 0.60 |
| Vegetation  | 0.65 |
| Water       | 0.90 |
| Unlabeled   | 0.04 |

**Normalized Confusion Matrix** shows strong performance on Land and Water, moderate on Building/Vegetation/Road, and poor on the rare Unlabeled class.

---

## Key Visualizations

### Training Curves
- Training vs Validation IoU (steadily increasing)
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/ee86986d-0ae4-4b0d-87ed-175451c3db61" />

<br><br>

- Training vs Validation Loss (decreasing with mild overfitting)
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/39faf71e-c30a-4109-b39e-d8beab867453" />


### Qualitative Results
- Side-by-side comparison: Original image | Ground Truth | Prediction
  <img width="993" height="339" alt="image" src="https://github.com/user-attachments/assets/36708500-4e1b-4860-94cd-a25625b1b607" />

- Error maps highlighting misclassified pixels (red = incorrect)
  <img width="1990" height="524" alt="image" src="https://github.com/user-attachments/assets/b043ca98-0ee8-4cd7-a667-8afd74e7ec55" />
  <br>

  <img width="1990" height="524" alt="image" src="https://github.com/user-attachments/assets/1e63c403-e245-4379-baaf-e84e8571a317" />
  <br>
  <img width="1990" height="524" alt="image" src="https://github.com/user-attachments/assets/70744a09-5ec9-44d1-ab30-1dc4d75b157a" />

  
---

## How to Reproduce

### 1. Environment Setup (Google Colab recommended)

```python
!pip install -q patchify segmentation-models
```

### 2. Authentication & Download
```
from google.colab import userdata
import os

os.environ["KAGGLE_API_TOKEN"] = userdata.get('KAGGLE_API_TOKEN')
!kaggle datasets download -d humansintheloop/semantic-segmentation-of-aerial-imagery --unzip
```

### 3. Core Libraries
```
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import MeanIoU
import segmentation_models as sm

os.environ['SM_FRAMEWORK'] = 'tf.keras'
```
Follow the Notebook after for full training Pipeline

## Observations & Insights

- Water and Land are the easiest classes (high IoU, distinct spectral signatures).
- Road is the most challenging among the major classes (thin linear features easily confused with Land).
- Unlabeled class has extremely low IoU because it is rare and poorly defined.
- Mild overfitting appears after ~60–70 epochs (validation curves plateau while training continues to improve).

## License & Credits

- Dataset: Humans in the Loop / MBRSC
- Code structure inspired by : https://youtu.be/jvZm8REF2KY?si=aIzXDE_fa1Oc_m1k
- Built and trained on Google Colab (T4 GPU)

# 🧠 Ensemble Learning-Based Brain Cancer Diagnosis Using MRI Images

> **A Precision-Based Approach** — Applied Machine Learning Course Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 About

This project was developed as part of the **Applied Machine Learning** course at **FAST-NUCES, Lahore**. It presents an ensemble deep learning pipeline for automated brain tumor classification from MRI scans, combining image super-resolution enhancement with multiple state-of-the-art CNN architectures.

### Authors

| Name | Email |
|------|-------|
| Muhammad Asim | l215240@lhr.nu.edu.pk |
| Murtaza Ahmad | l215173@lhr.nu.edu.pk |
| Muhammad Danyal Jawad | l215221@lhr.nu.edu.pk |
| Haseeb Zaheer | l215171@lhr.nu.edu.pk |

---

## 🎯 Problem Statement

Brain tumor diagnosis via MRI currently relies on manual analysis by radiologists, which is time-consuming and prone to human error. This project automates the classification pipeline using deep learning to improve diagnostic accuracy and efficiency.

---

## 🗂️ Dataset

**Brain Tumor MRI Dataset** — curated by Masoud Nickparvar on [Kaggle](https://www.kaggle.com/)

| Property | Details |
|----------|---------|
| Total Images | 7,023 MRI scans |
| Format | JPEG |
| Classes | Glioma, Meningioma, Pituitary Tumor, No Tumor |
| Sources | figshare, SARTAJ, Br35H |

---

## 🏗️ Pipeline Overview

```
MRI Input
    │
    ▼
┌─────────────┐
│   ESRGAN    │  ← Image Super-Resolution Enhancement
└─────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│           Base Models (Feature Extraction)│
│  ┌─────────────┐  ┌───────┐  ┌────────┐ │
│  │EfficientNetB3│  │ VGG16 │  │ VGG19  │ │
│  └─────────────┘  └───────┘  └────────┘ │
└──────────────────────────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Stacking Ensemble │  ← Meta-model (Logistic Regression)
└─────────────────────┘
    │
    ▼
Classification Output
(Glioma / Meningioma / Pituitary / No Tumor)
```

---

## 🧪 Methodology

### 1. Preprocessing
- Images resized to **224×224 pixels**
- Pixel values normalized to **[0, 1]** (divided by 255)

### 2. Image Enhancement
- **ESRGAN** (Enhanced Super-Resolution GAN) used to upscale and sharpen MRI images, improving feature extraction quality for downstream models.

### 3. Models Used

#### EfficientNetB3
- CNN with compound scaling across depth, width, and resolution
- Lightweight and optimized for real-time inference
- Achieved the highest individual accuracy

#### VGG16 & VGG19
- Deep CNNs from the Visual Geometry Group
- Use 3×3 convolutional filters and 2×2 max-pooling layers
- VGG19 (19 layers) slightly outperforms VGG16 (16 layers)

#### Stacking Ensemble
- Base models trained on the original dataset
- Their predictions serve as input features to a **meta-model**
- Meta-model learns optimal combination of base predictions

---

## 📊 Results

| Model | Training Accuracy | Validation Accuracy |
|-------|:-----------------:|:-------------------:|
| EfficientNetB3 | 99.82% | 99.77% |
| VGG19 | 99.35% | 98.86% |
| VGG16 | 98.91% | 96.49% |
| **Ensemble (Stacking)** | **99.79%** | **99.39%** |

The stacking ensemble effectively mitigates individual model weaknesses, yielding robust and generalizable classification performance.

---

## 🔮 Future Work

- **Content-Based Image Retrieval (CBIR)**: Allow clinicians to search for similar historical MRI cases using latent feature representations (autoencoders / ViTs).
- **Clustering & Similarity Matching**: Apply k-means clustering with Euclidean distance metrics to speed up similar-case retrieval.
- **FAISS Integration**: Use approximate nearest neighbor algorithms for scalable retrieval over large medical datasets.

---

## 📦 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/your-username/brain-tumor-ensemble.git
cd brain-tumor-ensemble

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset

# Run training
python train.py --model efficientnetb3

# Run ensemble
python ensemble.py
```

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow / Keras
- **Image Enhancement**: ESRGAN
- **Models**: EfficientNetB3, VGG16, VGG19
- **Ensemble**: Scikit-learn (Stacking / Logistic Regression)
- **Dataset**: Kaggle Brain Tumor MRI Dataset

---

## 📚 References

1. Ledig et al., "Photo-realistic single image super-resolution using a GAN," *IEEE TMI*, 2017.
2. Wang et al., "ESRGAN: Enhanced super-resolution generative adversarial networks," *MIA*, 2018.
3. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," *ICLR*, 2021.
4. Asiri et al., "Advancing brain tumor classification through fine-tuned vision transformers," *Sensors*, 2024.
5. Zhou et al., "Shape-scale co-awareness network for 3D brain tumor segmentation," *IEEE TMI*, 2024.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> *Developed for the Applied Machine Learning course — FAST-NUCES, Lahore, Pakistan*

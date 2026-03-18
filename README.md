🧠 Ensemble Learning-Based Brain Tumor Diagnosis using MRI Images
📌 Overview

This project presents an ensemble learning-based system for the automated diagnosis of brain tumors using MRI images. The goal is to improve diagnostic accuracy and reduce reliance on manual interpretation by radiologists.

The system leverages deep learning models and image enhancement techniques to classify MRI scans into tumor and non-tumor categories with high precision.

🎓 Academic Context

This paper was developed as part of the Applied Machine Learning course project at FAST School of Computing, FAST-NUCES Lahore.

👨‍💻 Authors

Muhammad Asim

Murtaza Ahmad

Muhammad Danyal Jawad

Haseeb Zaheer

🧪 Methodology

The proposed system follows a structured pipeline:

1. Image Enhancement

Used ESRGAN to improve MRI image quality and resolution.

2. Feature Extraction & Classification Models

Multiple deep learning models were used:

EfficientNetB3

VGG16

VGG19

Vision Transformer (ViT)

ResNet50

3. Ensemble Learning

A stacking ensemble technique was applied.

Predictions from base models were combined using a meta-model.

This improved overall classification performance and robustness.

📊 Dataset

Brain Tumor MRI Dataset (Kaggle)

Total Images: 7,023

Classes:

Glioma

Meningioma

Pituitary Tumor

No Tumor

⚙️ Preprocessing

Image resizing to 224×224

Normalization of pixel values to [0,1]

📈 Results

The system achieved very high accuracy:

Model	Training Accuracy	Validation Accuracy
EfficientNetB3	~99.82%	~99.77%
VGG16	~98.91%	~96.49%
VGG19	~99.35%	~98.86%
Ensemble Model	~99.79%	~99.39%

These results demonstrate that ensemble learning significantly improves performance compared to individual models.

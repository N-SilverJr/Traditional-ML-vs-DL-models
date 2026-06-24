# Benchmarking Deep Feature Extraction and End-to-End Deep Learning for Pediatric Pneumonia Detection in Chest Radiographs

**An academic benchmarking study comparing frozen deep features vs. end-to-end CNNs for detecting pneumonia in pediatric chest X-rays.**

## Overview

This project benchmarks two deep learning paradigms for binary classification (Normal vs. Pneumonia) on a dataset of **5,856 pediatric chest radiographs**:

- **Frozen Feature Extraction** using EfficientNetB0 + classical ML classifiers (Logistic Regression & SVM)
- **End-to-End Transfer Learning** with modern CNN architectures

The study highlights performance differences, clinical trade-offs (especially recall vs. specificity), architectural insights (e.g., Global Average Pooling), and practical implications for real-world pediatric radiology.

## Key Highlights

- **Dataset**: 5,856 images with 2.89:1 class imbalance (addressed via class weights)
- **Best Model**: MobileNetV2 — **84.1% Accuracy**, **0.884 F1**, **0.948 AUC**
- Strong emphasis on clinical usability and model interpretability

## Models Compared

### Feature Extraction Pipeline
- EfficientNetB0 (frozen, 1280 features)
- Logistic Regression (Baseline & Augmented)
- SVM (RBF Kernel)

### End-to-End Models
- Custom CNN (from scratch baseline)
- ResNet50V2
- MobileNetV2 (best overall)
- NASNetMobile

## Results Summary

| Model                  | Accuracy | F1-Score | AUC-ROC | Recall  | Specificity |
|------------------------|----------|----------|---------|---------|-------------|
| Logistic Regression    | 73.2%    | 0.822    | 0.844   | **98.7%** | 30.8%     |
| SVM (RBF)              | 77.2%    | 0.844    | 0.917   | **98.7%** | 41.5%     |
| Custom CNN             | 81.3%    | 0.865    | 0.921   | 94.2%   | -           |
| ResNet50V2             | 82.7%    | 0.871    | 0.934   | 91.5%   | -           |
| **MobileNetV2**        | **84.1%**| **0.884**| **0.948**| 90.2%   | -           |
| NASNetMobile           | 83.5%    | 0.878    | 0.941   | 90.8%   | -           |


## Features

- Class imbalance handling with `compute_class_weight`
- Comprehensive data augmentation
- Comparison of Global Average Pooling vs. Flatten
- Analysis of ImageNet pre-training benefits for medical domain adaptation
- Clinical trade-off discussion (high-recall screening vs. balanced diagnosis)

## Technologies Used

- Python, TensorFlow/Keras
- EfficientNet, ResNet, MobileNet, NASNet
- Scikit-learn (Logistic Regression, SVM)
- Matplotlib & Seaborn (learning curves)

## How to Use

1. Clone the repository
2. Download the [pediatric pneumonia dataset](https://data.mendeley.com/datasets/rscbjbr9sj/3) (Kermany et al.)
3. Install dependencies: `pip install -r requirements.txt`
4. Run notebooks in order

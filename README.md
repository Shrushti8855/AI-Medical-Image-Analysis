# AI Medical Image Analysis

An educational chest X-ray image classification project using MobileNetV2 transfer learning, TensorFlow, and Streamlit to classify images as NORMAL or PNEUMONIA.

> **Disclaimer:** This project is for educational and portfolio purposes only. It is not a validated medical diagnostic tool and must not be used to make medical or treatment decisions.

## Project Overview

This project explores how deep learning and transfer learning can be applied to medical image classification. A MobileNetV2 model pretrained on ImageNet is used as a feature extractor with a custom classification head trained on a public chest X-ray dataset.

## Features

- Chest X-ray image classification into NORMAL or PNEUMONIA
- MobileNetV2 transfer learning
- Fine-tuning of selected pretrained layers
- Class weighting to help address class imbalance
- Evaluation using precision, recall, F1-score, and a confusion matrix
- Streamlit interface for uploading and testing images

## Technology Stack

- Python
- TensorFlow and Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit

## Project Structure

```text
AI-Medical-Image-Analysis/
|-- app.py
|-- train_model.py
|-- evaluate_model.py
|-- check_data.py
|-- check_model.py
|-- check_models.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- src/
|   |-- __init__.py
|   |-- data.py
|   |-- model.py
|   |-- train.py
|-- data/
|   |-- raw/                 # Downloaded dataset; not tracked by Git
|-- models/                  # Trained models; not tracked by Git
|-- outputs/                 # Generated reports; not tracked by Git
```

## Dataset

This project uses the Chest X-Ray Images (Pneumonia) dataset available on Kaggle:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Download the dataset and extract it so that the images are available under:

`data/raw/chest_xray/`

The expected folders are `train`, `test`, and `val`, each containing `NORMAL` and `PNEUMONIA` subfolders.

The training pipeline creates a stratified training and validation split from the original training directory. The separate test directory is used for final evaluation.

Dataset counts used in the current project:

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,073 | 3,099 | 4,172 |
| Validation | 268 | 776 | 1,044 |
| Test | 234 | 390 | 624 |

## Model Architecture

The model uses MobileNetV2 pretrained on ImageNet as its base, followed by:

- Image preprocessing to the expected input range
- Global average pooling
- Dropout regularization
- A sigmoid output layer for binary classification

The initial training phase freezes the pretrained base. Fine-tuning then unfreezes selected layers while keeping Batch Normalization layers frozen.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download and extract the dataset into the expected directory before training or evaluating.

## Train the Model

```powershell
python train_model.py
```

Training creates model files under `models/` and a training log under `outputs/reports/`.

## Evaluate the Model

```powershell
python evaluate_model.py
```

## Run the Streamlit App

```powershell
python -m streamlit run app.py
```

Upload a supported chest X-ray image to view the model's prediction score. Predictions are model outputs, not medical conclusions.

## Current Test Results

The current model was evaluated on 624 images at a classification threshold of 0.50.

| Metric | NORMAL | PNEUMONIA |
|---|---:|---:|
| Precision | 0.98 | 0.77 |
| Recall | 0.50 | 0.99 |
| F1-score | 0.66 | 0.87 |
| Support | 234 | 390 |

- Overall test accuracy: **80.77%**
- Confusion matrix, rows = actual labels and columns = predicted labels:

```text
                 Predicted NORMAL  Predicted PNEUMONIA
Actual NORMAL              116                  118
Actual PNEUMONIA              2                  388
```

These results show that the current model detects most pneumonia cases in this test set but incorrectly labels many NORMAL images as PNEUMONIA. Accuracy alone does not describe model safety or clinical usefulness.

## Limitations and Future Improvements

- The dataset may not represent all patient populations, hospitals, scanners, or image conditions.
- The current test results show a substantial false-positive rate for NORMAL images.
- Further validation, data-quality checks, calibration, and independent testing would be needed before any clinical consideration.
- This project is not intended for diagnosis, treatment decisions, or real-world clinical use.

Potential future improvements include threshold analysis on validation data, more detailed error analysis, and testing on an independent dataset.

## Author

Shrushti Shewale

This project was developed as an educational machine learning and portfolio project.

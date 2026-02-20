# 🧠 Face Demographic Analyzer  
### Age Group • Gender • Ethnicity Prediction using Deep Learning

A multi-output deep learning system that predicts **Age Group, Gender, and Ethnicity** from facial images using Transfer Learning (MobileNetV2) and deploys predictions via a Streamlit web application.

---

## 🚀 Project Overview

This project implements a multi-task Convolutional Neural Network (CNN) to classify:

- 👶 Age Group (5 Classes)
- 🚻 Gender (Binary Classification)
- 🌍 Ethnicity (5 Classes)

The model is trained on the UTKFace dataset and deployed using Streamlit for real-time predictions.

---

## 🏗️ Architecture

Transfer Learning using:

- **MobileNetV2 (Pretrained on ImageNet)**
- Custom Multi-Output Classification Head
- Dropout Regularization
- Two-Phase Training:
  - Phase 1: Frozen Backbone
  - Phase 2: Fine-Tuning Last Layers

---

## 📊 Model Performance (Validation)

| Task        | Accuracy |
|------------|----------|
| Age Group  | ~81%     |
| Gender     | ~78–85%  |
| Ethnicity  | ~45–50%  |

> Ethnicity classification is inherently challenging due to dataset imbalance and visual similarity across groups.

---

## 📂 Dataset

Dataset Used: **UTKFace**

Each image filename encodes:

age_gender_ethnicity_date.jpg

25_0_2_201701161745.jpg


- Age → Converted to 5 Age Groups
- Gender → 0 (Male), 1 (Female)
- Ethnicity → 5 categories

---

## 🛠️ Tech Stack

- Python 3.10
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

---

## 🧠 Training Strategy

### ✔ Preprocessing
- Resize images to 96×96
- MobileNetV2 `preprocess_input()`

### ✔ Regularization
- Dropout (0.5)
- EarlyStopping

### ✔ Optimization
- Adam Optimizer
- Reduced Learning Rate during fine-tuning

---

## 🌐 Streamlit Web App

Users can:

1. Upload an image
2. Get predictions for:
   - Gender
   - Age Group
   - Ethnicity

Run locally:

```bash
streamlit run app.py

face-demographic-analyzer/
│
├── app.py
├── face_multi_output_model.keras
├── requirements.txt
├── README.md
└── notebook.ipynb

⚠️ Ethical Disclaimer

This project is built for educational and research purposes only.

Ethnicity prediction from facial images may be biased.

Results should NOT be used for real-world decision-making.

The dataset may contain imbalances and labeling noise.

🚀 Future Improvements

Add face detection before prediction

Improve ethnicity accuracy using EfficientNet

Deploy on Streamlit Cloud

Add confidence score visualization

Add multi-face detection support

Perform bias analysis across demographic groups

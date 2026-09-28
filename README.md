# Fingerprint-Liveness-Detection
Lightweight Deep learning Framework On fingerprint Liveness Detection using Genetic Algorithm
<div align="center">

# 🔍 Fingerprint Liveness Detection

### CNN + Genetic Algorithm Feature Selection + Decision Tree

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

**A deep learning web application that detects whether a fingerprint is live or spoofed using a hybrid CNN + GA + Decision Tree pipeline achieving 98.21% accuracy.**

</div>

---

## 📊 Model Performance

<div align="center">

![Model Comparison](https://raw.githubusercontent.com/Hajiraunnisa/Fingerprint-Liveness-Detection/static/static/analytics/model_comparison.png)

</div>

| Stage | Method | Accuracy | Precision | Recall | F1-Score |
|:-----:|--------|:--------:|:---------:|:------:|:--------:|
| 1 | CNN alone | 96.37% | 93.22% | 100.00% | 96.49% |
| 2 | CNN + GA + CNN head | 93.68% | 91.48% | 96.31% | 93.83% |
| ✅ 3 | **CNN + GA + Decision Tree** | **98.21%** | **97.54%** | **98.90%** | **98.22%** |

---

## 🧠 How It Works

```
Fingerprint Image
       ↓
  CNN Feature Extractor  (leaky_re_lu_5 layer → 256-dim features)
       ↓
  Genetic Algorithm      (selects optimal discriminative features)
       ↓
  Decision Tree          (Live / Fake classification)
       ↓
  Web App Result         (Prediction + Confidence Score)
```

---

## 📈 Analytics & Results

<div align="center">

### Confusion Matrix
![Confusion Matrix](https://raw.githubusercontent.com/Hajiraunnisa/Fingerprint-Liveness-Detection/static/static/analytics/confusion_matrix_dt.png)

### ROC Curve
![ROC Curve](https://raw.githubusercontent.com/Hajiraunnisa/Fingerprint-Liveness-Detection/static/static/analytics/roc_dt.png)

</div>

<div align="center">

| Accuracy | Precision | Recall | F1-Score |
|:--------:|:---------:|:------:|:--------:|
| ![Accuracy](https://raw.githubusercontent.com/Hajiraunnisa/Fingerprint-Liveness-Detection/static/static/analytics/accuracy_comparison.png) | ![Precision](https://raw.githubusercontent.com/Hajiraunnisa/Fingerprint-Liveness-Detection/static/static/analytics/precision_comparison.png) | ![Recall](https://raw.githubusercontent.com/Hajiraunnisa/Fingerprint-Liveness-Detection/static/static/analytics/recall_comparison.png) | ![F1](https://raw.githubusercontent.com/Hajiraunnisa/Fingerprint-Liveness-Detection/static/static/analytics/f1_comparison.png) |

</div>

---

## 🏗️ CNN Architecture

```
Input (224×224×3)
 ├─ Conv2D(32)  + BatchNorm + LeakyReLU + MaxPool
 ├─ Conv2D(64)  + BatchNorm + LeakyReLU + MaxPool
 ├─ Conv2D(128) + BatchNorm + LeakyReLU + MaxPool
 ├─ Conv2D(256) + BatchNorm + LeakyReLU
 ├─ GlobalAveragePooling2D
 ├─ Dense(256)  + LeakyReLU + Dropout(0.5)
 ├─ Dense(128)  + LeakyReLU + Dropout(0.3)
 └─ Dense(1, sigmoid)  →  Live / Fake
```

---

## 🧬 Genetic Algorithm

| Parameter | Value |
|-----------|:-----:|
| Population Size | 30 |
| Generations | 20 |
| Mutation Rate | 5% |
| Elite Size | 2 |
| Selection | Tournament (k=3) |
| Fitness Function | 0.7 × Accuracy + 0.3 × F1 |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow / Keras |
| Machine Learning | scikit-learn |
| Image Processing | OpenCV |
| Web Framework | Flask |
| Data & Visualization | NumPy, Pandas, Matplotlib, Seaborn |

---

## 📁 Project Structure

```
├── app.py                        ← Flask web application
├── Cnn.py                        ← CNN architecture
├── train.py                      ← CNN training script
├── feature_extraction.py         ← Extract CNN features
├── ga_feature_selection.py       ← Genetic Algorithm feature selection
├── decision_tree.py              ← Decision Tree training & evaluation
├── comparision.py                ← Model comparison charts
├── selected_feature_indices.txt  ← GA-selected feature indices
├── models/
│   ├── best_cnn.keras            ← Trained CNN
│   └── decision_tree.pkl         ← Trained Decision Tree
├── dataset/
│   ├── train/  (live/ + fake/)
│   └── val/    (live/ + fake/)
├── static/analytics/             ← Charts and metrics
└── templates/
    ├── index.html                ← Detection page
    └── analytics.html            ← Analytics dashboard
```

---

## 🚀 Setup & Run

### 1. Clone
```bash
git clone https://github.com/Hajiraunnisa/Fingerprint-Liveness-Detection.git
cd Fingerprint-Liveness-Detection
```

### 2. Install dependencies
```bash
pip install tensorflow flask opencv-python scikit-learn numpy matplotlib seaborn pandas joblib
```

### 3. Prepare dataset
Place the [LivDet dataset](https://livdet.org/) into:
```
dataset/train/live/   ← live fingerprints
dataset/train/fake/   ← spoof fingerprints
dataset/val/live/
dataset/val/fake/
```

### 4. Train the pipeline (in order)
```bash
python train.py                  # Step 1: Train CNN
python feature_extraction.py     # Step 2: Extract features
python ga_feature_selection.py   # Step 3: Run Genetic Algorithm
python decision_tree.py          # Step 4: Train Decision Tree
```

### 5. Run the web app
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

---

## 📦 Dataset

This project uses the **LivDet** fingerprint liveness detection dataset (~10,000 `.BMP` images).
The dataset is **not included** in this repo due to size. Download from [livdet.org](https://livdet.org/).

---

<div align="center">

Made with ❤️ by **Hajira** — Final Year Major Project

</div>

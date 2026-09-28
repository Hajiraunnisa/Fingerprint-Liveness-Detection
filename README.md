# Fingerprint-Liveness-Detection
Lightweight Deep learning Framework On fingerprint Liveness Detection using Genetic Algorithm
# Fingerprint Liveness Detection
### CNN + Genetic Algorithm Feature Selection + Decision Tree

A deep learning web application that detects whether a fingerprint is **live** or **fake (spoof)** using a hybrid pipeline: a custom CNN for feature extraction, a Genetic Algorithm (GA) for optimal feature selection, and a Decision Tree classifier for final prediction.

---

## Demo

Upload a fingerprint image through the web interface and get an instant prediction with confidence scores.

---

## Project Overview

Fingerprint spoofing is a serious threat to biometric security systems. This project addresses it with a three-stage pipeline:

| Stage | Method | Accuracy |
|-------|--------|----------|
| 1 | CNN alone | 96.37% |
| 2 | CNN + GA + CNN head | 93.68% |
| 3 | CNN + GA + Decision Tree ✅ | **98.21%** |

The deployed model (Stage 3) achieves **98.21% accuracy**, **97.54% precision**, **98.90% recall**, and **98.22% F1-score**.

---

## Features

- Custom 4-block CNN with LeakyReLU, BatchNorm, and L2 regularization
- Genetic Algorithm (GA) for selecting the most discriminative CNN features
- Decision Tree classifier trained on GA-selected features
- Flask web app with drag-and-drop image upload
- Analytics dashboard with model comparison charts, confusion matrix, and ROC curve
- Supports BMP, PNG, JPG, JPEG fingerprint images

---

## Tech Stack

- **Backend:** Python, Flask
- **Deep Learning:** TensorFlow / Keras
- **Machine Learning:** scikit-learn
- **Image Processing:** OpenCV
- **Data & Visualization:** NumPy, Matplotlib, Seaborn, Pandas

---

## Project Structure

```
├── app.py                        # Flask web application
├── Cnn.py                        # CNN architecture definition
├── train.py                      # CNN training script
├── feature_extraction.py         # Extract CNN features from dataset
├── ga_feature_selection.py       # Genetic Algorithm for feature selection
├── decision_tree.py              # Train & evaluate Decision Tree
├── comparision.py                # Generate model comparison charts
├── selected_feature_indices.txt  # GA-selected feature indices
├── best_chromosome.npy           # Best GA chromosome
├── best_features.npy             # Selected feature indices (numpy)
├── models/
│   ├── best_cnn.keras            # Trained CNN model
│   └── decision_tree.pkl         # Trained Decision Tree model
├── dataset/
│   ├── train/
│   │   ├── live/                 # Live fingerprint images
│   │   └── fake/                 # Spoofed fingerprint images
│   └── val/
│       ├── live/
│       └── fake/
├── static/
│   ├── uploads/                  # Uploaded images (runtime)
│   └── analytics/                # Charts and metrics
├── templates/
│   ├── index.html                # Main detection page
│   └── analytics.html            # Analytics dashboard
└── graphs/                       # Training accuracy/loss plots
```

---

## Dataset

This project uses the **LivDet** fingerprint liveness detection dataset, which contains live and spoofed fingerprint images in `.BMP` format. Images are organized into `train/live`, `train/fake`, `val/live`, and `val/fake` folders.

> The dataset is **not included** in this repository due to its size (~10,000+ images). Download it from [LivDet](https://livdet.org/) and place it in the `dataset/` folder.

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fingerprint-liveness-detection.git
cd fingerprint-liveness-detection
```

### 2. Install dependencies

```bash
pip install tensorflow keras flask opencv-python scikit-learn numpy matplotlib seaborn pandas joblib
```

### 3. Prepare the dataset

Place the dataset in the following structure:
```
dataset/
  train/
    live/   ← live fingerprint images
    fake/   ← spoof fingerprint images
  val/
    live/
    fake/
```

---

## Training Pipeline

Run these scripts **in order** to train the full pipeline from scratch:

```bash
# Step 1: Train the CNN
python train.py

# Step 2: Extract CNN features from the dataset
python feature_extraction.py

# Step 3: Run Genetic Algorithm for feature selection
python ga_feature_selection.py

# Step 4: Train the Decision Tree on selected features
python decision_tree.py

# Step 5: Generate model comparison charts
python comparision.py
```

---

## Running the Web App

Once the models are trained (or you have pre-trained models in `models/`):

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## Model Architecture

The CNN consists of 4 convolutional blocks followed by fully connected layers:

```
Input (224×224×3)
 → Conv2D(32) + BN + LeakyReLU + MaxPool
 → Conv2D(64) + BN + LeakyReLU + MaxPool
 → Conv2D(128) + BN + LeakyReLU + MaxPool
 → Conv2D(256) + BN + LeakyReLU
 → GlobalAveragePooling2D
 → Dense(256) + LeakyReLU + Dropout(0.5)
 → Dense(128) + LeakyReLU + Dropout(0.3)
 → Dense(1, sigmoid)
```

Features are extracted from the `leaky_re_lu_5` layer, reduced by the GA, then fed to the Decision Tree.

---

## Genetic Algorithm

| Parameter | Value |
|-----------|-------|
| Population size | 30 |
| Generations | 20 |
| Mutation rate | 5% |
| Elite size | 2 |
| Selection | Tournament (k=3) |
| Fitness | 0.7 × Accuracy + 0.3 × F1 |

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 98.21% |
| Precision | 97.54% |
| Recall | 98.90% |
| F1-Score | 98.22% |

---

## Files to Add to `.gitignore`

```
dataset/
models/
static/uploads/
*.npy
__pycache__/
*.pyc
*.docx
History/
```

---

## Author

**Hajira**  
Final Year Major Project — Fingerprint Liveness Detection using Deep Learning and Evolutionary Computation

---

## License

This project is for academic purposes.

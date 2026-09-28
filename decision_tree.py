# ============================================================
# Decision Tree Training
# CNN + GA Feature Selection + Decision Tree
#
# NOTE: Features are extracted in inference mode (single-image
# style) to match app.py. This avoids BatchNormalization
# train/inference mismatch that caused wrong predictions.
# ============================================================

import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model, Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle as sk_shuffle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ============================================================
# CONFIG
# ============================================================

TRAIN_FAKE    = "dataset/train/fake"
TRAIN_LIVE    = "dataset/train/live"
VAL_FAKE      = "dataset/val/fake"
VAL_LIVE      = "dataset/val/live"

FEATURE_LAYER = "leaky_re_lu_5"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 64

os.makedirs("static/analytics", exist_ok=True)

# ============================================================
# LOAD CNN + BUILD FEATURE EXTRACTOR
# ============================================================

print("Loading CNN model...")
cnn = load_model("models/best_cnn.keras")
extractor = Model(inputs=cnn.inputs, outputs=cnn.get_layer(FEATURE_LAYER).output)
print(f"Feature extractor: {FEATURE_LAYER}, output shape: {extractor.output.shape}")

# ============================================================
# EXTRACT FEATURES
# Images are processed in small batches in inference mode —
# same as app.py does at runtime, so distributions match.
# ============================================================

def extract_folder(folder, label):
    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))
    ])
    print(f"  {folder}: {len(files)} images, label={label}")

    all_feats  = []
    batch_imgs = []

    for i, fname in enumerate(files):
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE).astype("float32") / 255.0
        batch_imgs.append(img)

        if len(batch_imgs) == BATCH_SIZE or i == len(files) - 1:
            feats = extractor.predict(np.array(batch_imgs), verbose=0)
            all_feats.extend(feats)
            batch_imgs = []
            print(f"    {i+1}/{len(files)}", end="\r")

    print()
    return np.array(all_feats), np.full(len(all_feats), label, dtype=np.int32)

# ============================================================
# EXTRACT TRAIN FEATURES
# ============================================================

print("\nExtracting TRAIN features...")
tf_feats, tf_labs = extract_folder(TRAIN_FAKE, label=0)
tl_feats, tl_labs = extract_folder(TRAIN_LIVE, label=1)

X_train = np.vstack([tf_feats, tl_feats])
y_train = np.concatenate([tf_labs, tl_labs])
X_train, y_train = sk_shuffle(X_train, y_train, random_state=42)
print(f"Train: {X_train.shape} | mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")

# ============================================================
# EXTRACT VAL FEATURES
# ============================================================

print("\nExtracting VAL features...")
vf_feats, vf_labs = extract_folder(VAL_FAKE, label=0)
vl_feats, vl_labs = extract_folder(VAL_LIVE, label=1)

X_val = np.vstack([vf_feats, vl_feats])
y_val = np.concatenate([vf_labs, vl_labs])
print(f"Val:   {X_val.shape} | mean: {X_val.mean():.4f}, std: {X_val.std():.4f}")

# ============================================================
# LOAD GA SELECTED INDICES
# ============================================================

indices = []
with open("selected_feature_indices.txt") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        for p in line.replace(",", " ").split():
            if p.isdigit():
                indices.append(int(p))
indices = np.array(indices, dtype=int)
print(f"\nGA indices: {len(indices)}, max={indices.max()}, feature_dim={X_train.shape[1]}")

X_train_sel = X_train[:, indices]
X_val_sel   = X_val[:, indices]

# ============================================================
# TRAIN DECISION TREE
# ============================================================

print("\nTraining Decision Tree...")
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=10,
    random_state=42
)
dt_model.fit(X_train_sel, y_train)

# ============================================================
# PREDICTIONS & METRICS
# ============================================================

y_pred = dt_model.predict(X_val_sel)
y_prob = dt_model.predict_proba(X_val_sel)[:, 1]

accuracy  = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall    = recall_score(y_val, y_pred)
f1        = f1_score(y_val, y_pred)

print("\n==============================")
print("DECISION TREE PERFORMANCE")
print("==============================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print("\nClassification Report\n")
print(classification_report(y_val, y_pred, target_names=["Fake", "Live"]))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=["Fake", "Live"],
    yticklabels=["Fake", "Live"]
)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("static/analytics/confusion_matrix_dt.png", dpi=300)
plt.close()

# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Decision Tree ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/analytics/roc_dt.png", dpi=300)
plt.close()

# ============================================================
# SAVE METRICS TO FILE
# ============================================================

with open("static/analytics/dt_metrics.txt", "w") as f:
    f.write("Decision Tree Performance\n")
    f.write("=========================\n\n")
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-score  : {f1:.4f}\n")
    f.write(f"AUC       : {roc_auc:.4f}\n\n")
    f.write(classification_report(y_val, y_pred, target_names=["Fake", "Live"]))

# ============================================================
# SAVE MODEL + FEATURES
# ============================================================

joblib.dump(dt_model, "models/decision_tree.pkl")
np.save("train_selected_features.npy", X_train_sel)
np.save("val_selected_features.npy",   X_val_sel)
np.save("train_selected_labels.npy",   y_train)
np.save("val_selected_labels.npy",     y_val)

print("\nDecision Tree saved to models/decision_tree.pkl")
print("Done!")

# ============================================================
# Fingerprint Liveness Detection
# CNN + GA Feature Selection + Decision Tree
# Flask Application
# ============================================================

import os
import cv2
import uuid
import numpy as np

# Suppress TensorFlow/oneDNN noise before importing tf
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

from flask import Flask, render_template, request, jsonify

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import load_model

import joblib

# ============================================================
# Flask Setup
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE         = (224, 224)
CNN_MODEL_PATH     = "models/best_cnn.keras"
DT_MODEL_PATH      = "models/decision_tree.pkl"
FEATURE_INDEX_PATH = "selected_feature_indices.txt"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
CLASSIFIER_NAME    = "CNN + GA + Decision Tree"

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")
cnn_model = load_model(CNN_MODEL_PATH)
decision_tree = joblib.load(DT_MODEL_PATH)

# Feature extractor uses leaky_re_lu_5 — matches inference-mode
# distribution the Decision Tree was trained on.
feature_extractor = tf.keras.Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.get_layer("leaky_re_lu_5").output
)

# ============================================================
# LOAD GA INDICES
# ============================================================

selected_indices = []
with open(FEATURE_INDEX_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        for p in line.replace(",", " ").split():
            if p.isdigit():
                selected_indices.append(int(p))
selected_indices = np.array(selected_indices, dtype=int)

print(f"Ready. CNN + Decision Tree loaded | GA features: {len(selected_indices)}")

# ============================================================
# HELPERS
# ============================================================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be read. Check format/path.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def extract_features(image_path):
    img = preprocess_image(image_path)
    features = feature_extractor.predict(img, verbose=0)
    return features.flatten()

def select_features(features):
    if max(selected_indices) >= len(features):
        raise ValueError(f"GA index error: max {max(selected_indices)} but features {len(features)}")
    return features[selected_indices].reshape(1, -1)

def prediction_to_label(pred):
    return "Live" if int(pred) == 1 else "Fake"

def calculate_scores(x):
    if hasattr(decision_tree, "predict_proba"):
        probs   = decision_tree.predict_proba(x)[0]
        classes = list(decision_tree.classes_)
        live_score = float(probs[classes.index(1)]) if 1 in classes else 0.0
        fake_score = float(probs[classes.index(0)]) if 0 in classes else 0.0
    else:
        pred = decision_tree.predict(x)[0]
        live_score = float(pred == 1)
        fake_score = float(pred == 0)
    return max(live_score, fake_score), live_score, fake_score

def predict_image(path):
    features = extract_features(path)
    selected = select_features(features)
    pred     = decision_tree.predict(selected)[0]
    label    = prediction_to_label(pred)
    conf, live, fake = calculate_scores(selected)
    return {
        "prediction": label,
        "confidence": round(conf * 100, 2),
        "liveScore":  round(live * 100, 2),
        "fakeScore":  round(fake * 100, 2),
        "classifier": CLASSIFIER_NAME
    }

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "message": "No image"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"success": False, "message": "Empty file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "Invalid format"}), 400

        ext      = os.path.splitext(file.filename)[1]
        filename = str(uuid.uuid4()) + ext
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to read image. Unsupported or corrupted format.")

        result = predict_image(filepath)

        print(f"[{result['prediction']}] {os.path.basename(filepath)} | confidence: {result['confidence']}%")

        return jsonify({
            "success":       True,
            "prediction":    result["prediction"],
            "confidence":    result["confidence"],
            "liveScore":     result["liveScore"],
            "fakeScore":     result["fakeScore"],
            "uploadedImage": "/" + filepath.replace("\\", "/"),
            "classifier":    result["classifier"]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=False)

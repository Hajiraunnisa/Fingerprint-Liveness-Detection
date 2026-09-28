import numpy as np
import os
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# 1. LOAD MODEL
# =========================
model = load_model(r"C:\Users\Hajira\OneDrive\Documents\Major P\models\best_cnn.keras")

# =========================
# 2. FIND FEATURE LAYER SAFELY
# =========================
# print model summary first (optional debug)
model.summary()

# take second last layer output (BEFORE final sigmoid/dense(1))
feature_layer = model.layers[-2].output

feature_extractor = Model(inputs=model.inputs, outputs=feature_layer)

# =========================
# 3. LOAD DATA
# =========================
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    "dataset/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_generator = datagen.flow_from_directory(
    "dataset/val",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
print("Class indices:", train_generator.class_indices)
print("Validation class indices:", test_generator.class_indices)
# =========================
# 4. EXTRACT FEATURES
# =========================
print("Extracting train features...")
train_features = feature_extractor.predict(train_generator)

print("Extracting test features...")
test_features = feature_extractor.predict(test_generator)

# =========================
# 5. LABELS
# =========================
train_labels = train_generator.classes
test_labels = test_generator.classes

# =========================
# 6. SAVE FEATURES
# =========================
np.save("train_features.npy", train_features)
np.save("test_features.npy", test_features)
np.save("train_labels.npy", train_labels)
np.save("test_labels.npy", test_labels)

print("✅ Feature extraction completed successfully!")
print("Train shape:", train_features.shape)
print("Test shape:", test_features.shape)
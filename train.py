import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from Cnn import model

# ==========================================================
# Paths
# ==========================================================

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

MODEL_DIR = "models"
GRAPH_DIR = "graphs"
HISTORY_DIR = "history"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# ==========================================================
# Parameters
# ==========================================================

IMG_SIZE = (224,224)

BATCH_SIZE = 32

EPOCHS = 30

# ==========================================================
# Data Augmentation
# ==========================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("\nClass Indices")
print(train_generator.class_indices)

# ==========================================================
# Compile Model
# ==========================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================================
# Callbacks
# ==========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath="models/best_cnn.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

callbacks = [
    early_stop,
    checkpoint,
    reduce_lr
]

# ==========================================================
# Train
# ==========================================================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================================
# Save Final Model
# ==========================================================

model.save("models/final_cnn.keras")

print("\nModel Saved Successfully.")

# ==========================================================
# Save History
# ==========================================================

history_dict = history.history

with open("history/history.json","w") as f:
    json.dump(history_dict,f)

print("History Saved.")

# ==========================================================
# Accuracy Plot
# ==========================================================

plt.figure(figsize=(8,6))

plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("graphs/accuracy.png")

# ==========================================================
# Loss Plot
# ==========================================================

plt.figure(figsize=(8,6))

plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")

plt.legend()

plt.grid(True)

plt.savefig("graphs/loss.png")

plt.show()

print("\nTraining Completed Successfully.")
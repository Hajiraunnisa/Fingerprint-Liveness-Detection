from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    LeakyReLU
)
from tensorflow.keras.regularizers import l2

IMG_SIZE = 224

model = Sequential([

    # ==============================
    # Input Layer
    # ==============================
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # ==============================
    # Block 1
    # ==============================
    Conv2D(
        32,
        (3,3),
        padding='same',
        kernel_regularizer=l2(0.0001)
    ),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    MaxPooling2D(pool_size=(2,2)),

    # ==============================
    # Block 2
    # ==============================
    Conv2D(
        64,
        (3,3),
        padding='same',
        kernel_regularizer=l2(0.0001)
    ),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    MaxPooling2D(pool_size=(2,2)),

    # ==============================
    # Block 3
    # ==============================
    Conv2D(
        128,
        (3,3),
        padding='same',
        kernel_regularizer=l2(0.0001)
    ),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    MaxPooling2D(pool_size=(2,2)),

    # ==============================
    # Block 4
    # ==============================
    Conv2D(
        256,
        (3,3),
        padding='same',
        kernel_regularizer=l2(0.0001)
    ),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),

    # ==============================
    # Global Average Pooling
    # ==============================
    GlobalAveragePooling2D(),

    # ==============================
    # Fully Connected Layer
    # ==============================
    Dense(
        256,
        kernel_regularizer=l2(0.0001)
    ),
    LeakyReLU(negative_slope=0.1),
    Dropout(0.5),

    Dense(
        128,
        kernel_regularizer=l2(0.0001)
    ),
    LeakyReLU(negative_slope=0.1),
    Dropout(0.3),

    # ==============================
    # Output Layer
    # ==============================
    Dense(1, activation='sigmoid')

])

model.summary()
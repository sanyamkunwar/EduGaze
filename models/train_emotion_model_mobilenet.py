import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

print("\n=== EDU-GAZE UPGRADED EMOTION MODEL (MobileNetV2) ===\n")

# -------------------------------------------------------------------------
# 1. SET PATHS
# -------------------------------------------------------------------------
BASE_PATH = "."
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    raise FileNotFoundError("train/ or test/ folders missing!")

print("Train path:", TRAIN_DIR)
print("Test path :", TEST_DIR)

# -------------------------------------------------------------------------
# 2. DATA GENERATORS
# -------------------------------------------------------------------------
IMG_SIZE = (96, 96)   # MobileNetV2 prefers larger input
BATCH_SIZE = 32

train_gen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1/255.0)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

print("Class mapping:", train_data.class_indices)

# -------------------------------------------------------------------------
# 3. BUILD MODEL (TRANSFER LEARNING)
# -------------------------------------------------------------------------
base_model = MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(5, activation="softmax")   # 5-class output
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# -------------------------------------------------------------------------
# 4. TRAIN TOP LAYERS
# -------------------------------------------------------------------------
print("\nTraining top layers...")
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# -------------------------------------------------------------------------
# 5. FINE-TUNE LOWER LAYERS
# -------------------------------------------------------------------------
print("\nFine-tuning MobileNetV2...")

base_model.trainable = True
for layer in base_model.layers[:-30]:   # Unfreeze only last 30 layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# -------------------------------------------------------------------------
# 6. SAVE FINAL MODEL
# -------------------------------------------------------------------------
SAVE_PATH = "../emotion_model.h5"
model.save(SAVE_PATH)

print(f"\nMobileNetV2 emotion model saved to: {SAVE_PATH}")
print("\n=== TRAINING COMPLETE ===\n")

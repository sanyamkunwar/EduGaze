import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

print("\n=== EDU-GAZE EMOTION MODEL TRAINING (IMAGE FOLDERS) ===\n")

# -------------------------------------------------------------------------
# 1. SET PATHS
# -------------------------------------------------------------------------
BASE_PATH = "."
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError("train/ folder not found. Please check extraction.")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError("test/ folder not found.")

print("Train path:", TRAIN_DIR)
print("Test path :", TEST_DIR)

# -------------------------------------------------------------------------
# 2. DATA GENERATORS
# -------------------------------------------------------------------------
IMG_SIZE = (48, 48)
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1/255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

print("\nClass mapping:", train_data.class_indices)

# -------------------------------------------------------------------------
# 3. BUILD CNN MODEL
# -------------------------------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(48,48,1)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    
    layers.Dense(5, activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print(model.summary())

# -------------------------------------------------------------------------
# 4. TRAIN
# -------------------------------------------------------------------------
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=20
)

# -------------------------------------------------------------------------
# 5. SAVE MODEL
# -------------------------------------------------------------------------
SAVE_PATH = "../emotion_model.h5"
model.save(SAVE_PATH)

print(f"\nModel saved to {SAVE_PATH}")
print("\n=== TRAINING COMPLETE ===")

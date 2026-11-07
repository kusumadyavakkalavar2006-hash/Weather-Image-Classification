import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# CLOUDY AND SUNNY WEATHER IMAGE CLASSIFICATION
# ============================================================

print("\n==============================================")
print("     WEATHER IMAGE CLASSIFICATION")
print("          CLOUDY AND SUNNY")
print("==============================================")


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "weather_model.h5"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "labels.txt"
)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 30

print("\nImage Size :", IMAGE_SIZE)
print("Batch Size :", BATCH_SIZE)
print("Epochs     :", EPOCHS)

print("\nDataset Path:")
print(DATASET_PATH)


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR!")
    print("Dataset folder was not found.")
    print("\nExpected location:")
    print(DATASET_PATH)

    input("\nPress Enter to close...")
    exit()


cloudy_path = os.path.join(
    DATASET_PATH,
    "cloudy"
)

sunny_path = os.path.join(
    DATASET_PATH,
    "sunny"
)


if not os.path.exists(cloudy_path):

    print("\nERROR!")
    print("cloudy folder was not found.")
    print("\nPlease create:")
    print(cloudy_path)

    input("\nPress Enter to close...")
    exit()


if not os.path.exists(sunny_path):

    print("\nERROR!")
    print("sunny folder was not found.")
    print("\nPlease create:")
    print(sunny_path)

    input("\nPress Enter to close...")
    exit()


# ============================================================
# 4. COUNT IMAGES
# ============================================================

cloudy_images = [
    file
    for file in os.listdir(cloudy_path)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]


sunny_images = [
    file
    for file in os.listdir(sunny_path)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]


print("\n==============================================")
print("             DATASET INFORMATION")
print("==============================================")

print("Cloudy images :", len(cloudy_images))
print("Sunny images  :", len(sunny_images))

print(
    "Total images  :",
    len(cloudy_images) + len(sunny_images)
)


if len(cloudy_images) == 0:

    print("\nERROR: No Cloudy images found.")

    input("\nPress Enter to close...")
    exit()


if len(sunny_images) == 0:

    print("\nERROR: No Sunny images found.")

    input("\nPress Enter to close...")
    exit()


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    validation_split=0.20,

    rotation_range=10,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.10,

    horizontal_flip=True
)


# ============================================================
# 6. TRAINING DATA
# ============================================================

print("\n==============================================")
print("          LOADING TRAINING DATA")
print("==============================================")


train_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True
)


# ============================================================
# 7. VALIDATION DATA
# ============================================================

print("\n==============================================")
print("         LOADING VALIDATION DATA")
print("==============================================")


validation_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False
)


# ============================================================
# 8. CLASS MAPPING
# ============================================================

print("\n==============================================")
print("              CLASS MAPPING")
print("==============================================")


print(
    train_generator.class_indices
)


# Expected:
# {'cloudy': 0, 'sunny': 1}


labels = [
    None
] * len(
    train_generator.class_indices
)


for class_name, index in train_generator.class_indices.items():

    labels[index] = class_name


print("\nLabels:")

for index, label in enumerate(labels):

    print(
        index,
        "=",
        label
    )


# ============================================================
# 9. CREATE CNN MODEL
# ============================================================

print("\n==============================================")
print("             CREATING CNN MODEL")
print("==============================================")


model = Sequential([

    # ------------------------------------------
    # First Convolution Layer
    # ------------------------------------------

    Conv2D(
        16,
        (3, 3),
        activation="relu",
        input_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )
    ),

    MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------------------
    # Second Convolution Layer
    # ------------------------------------------

    Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------------------
    # Third Convolution Layer
    # ------------------------------------------

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------------------
    # Feature Reduction
    # ------------------------------------------

    GlobalAveragePooling2D(),


    # ------------------------------------------
    # Fully Connected Layer
    # ------------------------------------------

    Dense(
        64,
        activation="relu"
    ),


    # ------------------------------------------
    # Dropout
    # ------------------------------------------

    Dropout(
        0.30
    ),


    # ------------------------------------------
    # Output Layer
    # ------------------------------------------

    Dense(
        2,
        activation="softmax"
    )

])


# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# 11. MODEL SUMMARY
# ============================================================

print("\n==============================================")
print("                MODEL SUMMARY")
print("==============================================")


model.summary()


# ============================================================
# 12. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=7,

    restore_best_weights=True
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n==============================================")
print("             STARTING TRAINING")
print("==============================================")

print("\nPlease wait...")
print("The model is learning Cloudy and Sunny images.\n")


history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[
        early_stopping
    ]
)


# ============================================================
# 14. EVALUATE MODEL
# ============================================================

print("\n==============================================")
print("              MODEL EVALUATION")
print("==============================================")


loss, accuracy = model.evaluate(

    validation_generator,

    verbose=0
)


print(
    "\nValidation Loss     :",
    round(loss, 4)
)


print(
    "Validation Accuracy :",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# 15. SAVE MODEL
# ============================================================

print("\n==============================================")
print("                SAVING MODEL")
print("==============================================")


model.save(
    MODEL_PATH
)


print(
    "\nModel saved successfully!"
)

print(
    "Location:",
    MODEL_PATH
)


# ============================================================
# 16. SAVE LABELS
# ============================================================

with open(
    LABEL_PATH,
    "w"
) as file:

    for label in labels:

        file.write(
            label + "\n"
        )


print(
    "\nLabels saved successfully!"
)

print(
    "Location:",
    LABEL_PATH
)


# ============================================================
# 17. FINAL RESULT
# ============================================================

print("\n==============================================")
print("           TRAINING COMPLETED")
print("==============================================")


print("\nClasses:")

for index, label in enumerate(labels):

    print(
        index,
        "=",
        label
    )


print(
    "\nFinal Validation Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print("\nModel file:")
print("weather_model.h5")


print("\nLabel file:")
print("labels.txt")


print("\n==============================================")
print("        NOW RUN weather_detection.py")
print("==============================================")


input("\nPress Enter to close...")
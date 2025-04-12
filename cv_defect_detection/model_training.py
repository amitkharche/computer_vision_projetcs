
"""
Train a CNN model to classify images as defective or non-defective.
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

IMG_SIZE = (64, 64)
BATCH_SIZE = 4
EPOCHS = 5
DATA_CSV = "data/image_labels.csv"
IMAGE_DIR = "."

def load_data():
    df = pd.read_csv(DATA_CSV)
    return df

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    df = load_data()

    # ✅ Keep labels as strings for class_mode='binary'
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_dataframe(
        train_df, directory=IMAGE_DIR, x_col="image_path", y_col="label",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
    )

    val_data = val_gen.flow_from_dataframe(
        val_df, directory=IMAGE_DIR, x_col="image_path", y_col="label",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
    )

    model = build_model()
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

    os.makedirs("model", exist_ok=True)
    model.save("model/cnn_defect_model.h5")
    with open("model/class_map.pkl", "wb") as f:
        pickle.dump(train_data.class_indices, f)

    val_data.reset()
    y_true = val_data.classes
    y_pred = model.predict(val_data).flatten()
    y_pred = np.where(y_pred > 0.5, 1, 0)

    logging.info("\n" + classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()

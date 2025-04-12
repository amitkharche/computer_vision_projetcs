"""
Train a CNN model to classify facial emotions using image data.
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pickle

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configs
IMG_SIZE = (48, 48)
DATA_PATH = "data/fer_labels.csv"
IMAGE_FOLDER = "."
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "emotion_cnn_model.h5")
LABEL_MAP_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")

def load_data():
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    df["emotion_label"] = le.fit_transform(df["emotion"])
    with open(LABEL_MAP_FILE, "wb") as f:
        pickle.dump(le, f)
    return df, le

def load_images(df):
    images = []
    labels = df["emotion_label"].values
    for path in df["image_path"]:
        img = load_img(os.path.join(IMAGE_FOLDER, path), target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    return np.array(images), to_categorical(labels)

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df, le = load_data()
    X, y = load_images(df)
    #X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model((*IMG_SIZE, 3), y.shape[1])
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8, callbacks=[early_stop])
    model.save(MODEL_FILE)
    logging.info(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()

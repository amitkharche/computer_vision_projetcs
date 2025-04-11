"""
Streamlit app to classify uploaded images as defective or non-defective.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

st.set_page_config(page_title="🧪 Defect Detection", layout="centered")
st.title("🧪 Upload an Image for Defect Classification")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/cnn_defect_model.h5")
    with open("model/class_map.pkl", "rb") as f:
        class_map = pickle.load(f)
    return model, {v: k for k, v in class_map.items()}

def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

model, class_labels = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("")

    img_tensor = preprocess_image(img)
    prediction = model.predict(img_tensor)[0][0]
    label = "defective" if prediction > 0.5 else "non-defective"

    st.markdown(f"### 🔍 Prediction: **{label.upper()}**")
    st.progress(int(prediction * 100)) if label == "defective" else st.progress(int((1 - prediction) * 100))

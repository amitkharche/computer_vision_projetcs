"""
Streamlit UI to classify uploaded facial images into emotions.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

st.set_page_config(page_title="😊 Facial Emotion Recognition", layout="centered")
st.title("😊 Facial Emotion Classifier")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/emotion_cnn_model.h5")
    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

def preprocess_image(image, target_size=(48, 48)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

model, label_encoder = load_model()

uploaded_file = st.file_uploader("Upload a facial image (48x48 or larger)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    #st.image(image, caption="Uploaded Image", use_column_width=False, width=250)
    st.image(image, caption="Uploaded Image", use_container_width=False, width=250)

    input_tensor = preprocess_image(image)
    prediction = model.predict(input_tensor)[0]
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    st.markdown("### 🎯 Predicted Emotion:")
    st.markdown(f"## `{predicted_label.upper()}`")
    st.bar_chart(prediction)

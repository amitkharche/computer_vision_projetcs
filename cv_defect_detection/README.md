# 🧪 Defect Detection in Manufacturing (Computer Vision)

## 📌 Use Case

This project uses computer vision to classify product images as **defective** or **non-defective**. It's ideal for manufacturing and quality control.

## 🧠 Model Overview

- **Model**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow / Keras
- **Preprocessing**: Resizing, normalization
- **Evaluation**: Binary accuracy, classification report

## 📁 Files

- `model_training.py`: Loads labeled images, trains and saves CNN
- `app.py`: Streamlit app to upload and classify images
- `requirements.txt`: Python dependencies

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python model_training.py
```

### 3. Launch the App

```bash
streamlit run app.py
```

📌 Add your own JPG images under `data/images/` and label them in `data/image_labels.csv`.

## 📄 License

MIT License

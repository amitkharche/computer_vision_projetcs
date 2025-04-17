
# 🖼️ Computer Vision Projects

This repository contains a collection of real-world computer vision projects powered by **Convolutional Neural Networks (CNNs)** using **TensorFlow/Keras**. Each project includes a training pipeline, image preprocessing, and a deployable **Streamlit app** for inference.

---

## 📦 Included Projects

### 🧪 1. Defect Detection in Manufacturing
Classifies product images as **defective or non-defective** for manufacturing quality control.

**Use Case**:
- Assembly line monitoring
- Automated defect inspection

**Tech Stack**:
- TensorFlow/Keras
- Binary classification CNN
- Streamlit

📁 Folder: `defect_detection_project/`

---

### 😊 2. Facial Emotion Recognition
Classifies facial images into emotion categories such as **Happy, Sad, Angry, Surprised, Neutral**.

**Use Case**:
- Retail shopper engagement
- Mental health monitoring
- Workplace stress detection

**Tech Stack**:
- CNN (48x48 grayscale images)
- LabelEncoder
- Streamlit + Docker

📁 Folder: `facial_emotion_recognition_project/`

---

### 🛍️ 3. Product Image Categorization
Automatically classifies product images into predefined categories like **Phone, Bag, Book, Shoe**.

**Use Case**:
- E-commerce catalog tagging
- Product discovery enhancement

**Tech Stack**:
- CNN classifier
- Label encoding
- Streamlit + Docker

📁 Folder: `product_image_categorization_project/`

---

### 🚗 4. Vehicle Classification from CCTV
Classifies vehicle images from CCTV footage into types like **Sedan, SUV, Bus, Truck, Motorbike**.

**Use Case**:
- Smart city surveillance
- Urban traffic planning
- Violation monitoring

**Tech Stack**:
- CNN + Dropout
- Image resize to 64x64
- Streamlit + Docker

📁 Folder: `vehicle_classification_project/`

---

## 🚀 How to Run Any Project

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Train the Model**
```bash
python model_training.py
```

3. **Launch the App**
```bash
streamlit run app.py
```

---

## 🐳 Docker Usage (Optional)

1. **Build Docker Image**
```bash
docker build -t project-name .
```

2. **Run Docker Container**
```bash
docker run -p 8501:8501 project-name
```

Then open [http://localhost:8501](http://localhost:8501)

---

## 📁 Common Folder Structure

```
project_folder/
├── data/
│   ├── images/                # Input images
│   └── labels.csv             # Labels or metadata
├── model/                     # Saved model and encoders
├── app.py                     # Streamlit app
├── model_training.py          # CNN training script
├── requirements.txt           # Dependencies
├── Dockerfile                 # Docker container setup
├── README.md                  # Documentation
```

---

## 🧰 Tech Stack

- Python
- TensorFlow/Keras
- Streamlit
- pandas, numpy
- LabelEncoder
- Docker

---

## 📄 License

All projects in this repository are licensed under the **MIT License**.

---

## 👤 Author

**Amit Kharche**  
🔗 [LinkedIn](https://www.linkedin.com/in/amitkharche)

---

## ⭐ Contributions Welcome!

If you find these projects helpful, feel free to ⭐ the repo.  
Fork, enhance, and submit a PR!


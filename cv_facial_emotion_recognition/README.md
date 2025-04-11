# 😊 Facial Emotion Recognition with CNN

## 📌 Business Use Case

In today's customer-centric world, understanding emotional reactions is critical in areas such as:
- **Retail**: Track shopper reactions to products or displays
- **Mental Health**: Monitor mood trends in patients
- **Workplace Wellness**: Gauge employee stress and engagement

This project demonstrates a complete pipeline for building, training, and deploying a deep learning model for **facial emotion classification** using simulated image data.

---

## 📁 Project Structure

```
facial_emotion_recognition_project/
├── data/
│   ├── images/                # Simulated facial images
│   └── fer_labels.csv         # Image paths and emotion labels
├── model/                     # Saved trained CNN and label encoder
├── model_training.py          # Model training pipeline
├── app.py                     # Streamlit web app
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Containerization setup
├── README.md                  # Documentation
└── .gitignore                 # Git ignore rules
```

---

## ⚙️ Features Used

- CNN trained on facial images (48x48 px)
- Labels: Happy, Sad, Angry, Surprised, Neutral
- TensorFlow, Keras, Streamlit

---

## 🚀 How to Run Locally

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

---

## 🐳 How to Use with Docker

### 1. Build the Image
```bash
docker build -t fer-app .
```

### 2. Run the Container
```bash
docker run -p 8501:8501 fer-app
```

Then open [http://localhost:8501](http://localhost:8501)

---

## 📄 License
MIT License

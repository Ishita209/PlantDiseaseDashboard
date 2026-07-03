# 🌿 Plant Disease Detection using Federated Learning

An interactive Streamlit dashboard for detecting apple leaf diseases using a deep learning model based on **InceptionV3** and comparing the performance of multiple **Federated Learning algorithms**.

---
## Live Demo
https://plantdiseasedashboard-apple.streamlit.app/

## 📌 Overview

This project demonstrates how Federated Learning can be used for plant disease detection while preserving data privacy. The dashboard allows users to explore model performance, compare different federated learning algorithms, and predict diseases from uploaded apple leaf images.

---

## ✨ Features

- Interactive Streamlit dashboard
- Image-based disease prediction
- Performance comparison of Federated Learning algorithms
- Centralized vs Federated Learning analysis
- Confusion Matrix visualization
- Interactive analytics using Plotly
- Multi-page dashboard

---

## 🧠 Deep Learning Model

- **Architecture:** InceptionV3 (Transfer Learning)
- **Framework:** TensorFlow / Keras
- **Input Image Size:** 299 × 299
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy

---

## 🤝 Federated Learning Algorithms

The project compares the following algorithms:

- FedAvg
- FedProx
- FedDyn
- FedAdam
- FedSGD

Experiments were conducted using both **3-client** and **9-client** federated settings.

---

## 📂 Dataset

**PlantVillage Dataset (Apple Leaves)**

Disease classes:

- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

---

## 📊 Dashboard Pages

### Home
Project overview and introduction.

### Analytics
Interactive charts showing algorithm performance.

### Model Performance
- Centralized vs Federated Learning comparison
- Individual algorithm comparison
- Confusion Matrix

### Prediction
Upload an apple leaf image and predict the disease using the trained InceptionV3 model.

### About
Project summary, dataset details, and technologies used.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- InceptionV3
- Plotly
- Pandas
- NumPy
- Pillow

---

## 📁 Project Structure

```
PlantDiseaseDashboard/
│
├── assets/
│   └── images/
├── data/
├── models/
├── pages/
├── utils/
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone <repository-url>
```

Move to the project directory

```bash
cd PlantDiseaseDashboard
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📈 Results

- Centralized InceptionV3 achieved the highest overall accuracy.
- FedProx provided the best Federated Learning performance.
- Increasing the number of clients from 3 to 9 resulted in only a slight decrease in model accuracy.

---

## 👩‍💻 Author

**Ishita Chaple**

B.Tech, Computer Science and Engineering

Maulana Azad National Institute of Technology (MANIT), Bhopal

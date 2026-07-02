import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="📄",
    layout="wide"
)

st.title("📄 About the Project")

st.markdown("""
## 🌿 Plant Disease Detection using Federated Learning

This project presents a privacy-aware deep learning framework for detecting apple leaf diseases using Federated Learning.

Instead of collecting all training data on a central server, Federated Learning enables multiple clients to train locally and share only model updates. This helps preserve data privacy while maintaining high prediction performance.

---

## Dataset

- **Dataset:** PlantVillage (Apple)
- **Total Images:** 3171
- **Classes:** 4

### Disease Classes

- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

---

## Deep Learning Model

- Architecture: **InceptionV3**
- Framework: TensorFlow / Keras
- Image Size: **299 × 299**
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

---

## Federated Learning Algorithms Compared

- FedAvg
- FedProx
- FedDyn
- FedAdam
- FedSGD

The dashboard compares these algorithms under both **3-client** and **9-client** configurations.

---

## Key Results

- Centralized Model Accuracy: **96.34%**
- Best Federated Model: **FedProx**
- Best Federated Accuracy: **93.51%**
- Communication Rounds: **8**

---

## Dashboard Features

- Interactive analytics
- Model performance comparison
- Plant disease prediction
- Project overview

---

## 👩‍💻 Developed By

**Ishita Chaple**

B.Tech, Computer Science & Engineering

Maulana Azad National Institute of Technology (MANIT), Bhopal
""")
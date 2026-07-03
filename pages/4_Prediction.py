import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Page Configuration

st.set_page_config(
    page_title="Prediction",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Plant Disease Prediction")
st.write(
    "Upload an image of an apple leaf to identify whether it is healthy or affected by Apple Scab, Black Rot, or Cedar Apple Rust using the trained InceptionV3 model."
)

# Load Model 

@st.cache_resource
def load_my_model():
    return load_model("models/crop_classifier_inception_new.h5")

model = load_my_model()

# Class Names

classes = [
    "Apple Scab",
    "Black Rot",
    "Cedar Apple Rust",
    "Healthy"
]

# Upload Image

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # Preprocess

    from tensorflow.keras.utils import load_img, img_to_array

    img = load_img(uploaded_file, target_size=(299, 299))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Prediction

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    with col2:

      st.subheader("Prediction")

      if classes[predicted_index] == "Healthy":
        st.success(f"🌿 {classes[predicted_index]}")
      else:
        st.error(f"🦠 {classes[predicted_index]}")

      st.info(f"Confidence: **{confidence:.2f}%**")

      st.subheader("Prediction Probabilities")

      for i, cls in enumerate(classes):
        st.write(f"**{cls}**")
        st.progress(float(prediction[0][i]))
        st.write(f"{prediction[0][i] * 100:.2f}%")

    st.divider()

    st.subheader("Model Information")

    st.markdown("""
- **Model:** InceptionV3
- **Input Size:** 299 × 299
- **Dataset:** PlantVillage (Apple)
- **Classes:** 4
- **Framework:** TensorFlow / Keras
- **Training Method:** Federated Learning
""")
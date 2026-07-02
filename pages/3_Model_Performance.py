import streamlit as st

st.set_page_config(
    page_title="Model Performance",
    layout="wide"
)

st.title("📈 Model Performance")

st.markdown("""
This page compares the performance of the centralized InceptionV3 model with different Federated Learning algorithms using 3-client and 9-client configurations.
""")

st.divider()

# Centralized Training 

st.subheader("Centralized InceptionV3 Training")

st.image(
    "assets/images/centralized_training.png",
    use_container_width=True
)

st.divider()

# Overall Comparison

st.subheader("Overall Performance Comparison")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "assets/images/centralized_vs_Federated_3_Clients.png",
        caption="Centralized vs Federated Learning (3 Clients)",
        use_container_width=True
    )

with col2:
    st.image(
        "assets/images/centralized_vs_Federated_9_Clients.png",
        caption="Centralized vs Federated Learning (9 Clients)",
        use_container_width=True
    )

st.divider()

# Algorithm Comparison

st.subheader("Individual Algorithm Comparison (3 Clients vs 9 Clients)")

algorithm = st.selectbox(
    "Select Federated Learning Algorithm",
    [
        "FedAvg",
        "FedProx",
        "FedDyn",
        "FedAdam",
        "FedSGD"
    ]
)

image_paths = {
    "FedAvg": "assets/images/fedavg_3_vs_9.png",
    "FedProx": "assets/images/fedprox_3_vs_9.png",
    "FedDyn": "assets/images/feddyn_3_vs_9.png",
    "FedAdam": "assets/images/fedadam_3_vs_9.png",
    "FedSGD": "assets/images/fedsgd_3_vs_9.png"
}

analysis = {
    "FedAvg":
    """
    **FedAvg**

    - Stable convergence over communication rounds.
    - Maintains comparable performance with both 3 and 9 clients.
    - Performs close to the centralized model.
    """,

    "FedProx":
    """
    **FedProx**

    - Best-performing Federated Learning algorithm.
    - Handles client heterogeneity effectively.
    - Shows minimal accuracy drop with increased clients.
    """,

    "FedDyn":
    """
    **FedDyn**

    - Faster convergence than FedAvg.
    - Good overall stability.
    - Maintains competitive accuracy.
    """,

    "FedAdam":
    """
    **FedAdam**

    - Adaptive optimization strategy.
    - Slightly slower convergence.
    - Moderate performance compared to FedAvg and FedProx.
    """,

    "FedSGD":
    """
    **FedSGD**

    - Basic federated optimization approach.
    - Lower convergence speed.
    - Most affected by increasing client count.
    """
}

st.image(
    image_paths[algorithm],
    caption=f"{algorithm}: Comparison of 3 Clients vs 9 Clients",
    use_container_width=True
)

st.info(analysis[algorithm])

st.divider()

# Confusion Matrix

st.subheader("Confusion Matrix")

st.image(
    "assets/images/confusion_matrix.png",
    use_container_width=False
)

st.divider()

st.success("""
### Key Observations

- The centralized InceptionV3 achieved the highest overall accuracy.
- FedProx consistently produced the best federated performance.
- FedAvg also performed competitively with minimal accuracy loss.
- Increasing the number of clients from 3 to 9 caused only a slight reduction in accuracy.
- The confusion matrix demonstrates effective classification across all four disease categories.
""")
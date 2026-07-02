import streamlit as st

# Page Config
st.set_page_config(
    page_title="Plant Disease Analytics",
    page_icon="🌿",
    layout="wide"
)

# Header
st.title("🌿 Federated Learning Analytics Dashboard")

st.markdown("### Apple Disease Detection using InceptionV3")

st.caption("Interactive dashboard for analyzing Federated Learning experiments.")

st.divider()

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset", "3171 Images")

with col2:
    st.metric("Classes", "4")

with col3:
    st.metric("Centralized Accuracy", "96.34%")

with col4:
    st.metric("Best FL Model", "FedProx")

st.divider()

# Overview
st.subheader("📌 Project Overview")

st.write("""
This dashboard presents the results of a Federated Learning based
Apple Plant Disease Detection system.

The experiments compare multiple Federated Learning algorithms
across different client configurations and evaluate their
classification performance.
""")

st.divider()

# Algorithms
col1, col2 = st.columns(2)

with col1:

    st.subheader("🤖 Algorithms")

    st.success("FedAvg")
    st.success("FedProx")
    st.success("FedDyn")
    st.success("FedAdam")
    st.success("FedSGD")

with col2:

    st.subheader("🌿 Disease Classes")

    st.info("Apple Scab")
    st.info("Black Rot")
    st.info("Cedar Apple Rust")
    st.info("Healthy")

st.divider()

st.info("👈 Use the sidebar to navigate through the dashboard.")
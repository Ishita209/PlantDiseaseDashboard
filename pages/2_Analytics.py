import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Analytics", layout="wide")

# Load Data
df = pd.read_csv("data/algorithm_results.csv")
df["Clients"] = pd.to_numeric(df["Clients"], errors="coerce")
# Remove the centralized model for FL comparison
fl_df = df[df["Algorithm"] != "Centralized"]

# Find the best-performing FL experiment
best_row = fl_df.loc[fl_df["Best Accuracy"].idxmax()]
# Title 
st.title("📊 Federated Learning Analytics")
st.caption("The charts below summarize the accuracy achieved by different federated learning algorithms across various client configurations.")

st.divider()

# KPI Cards 
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Best Algorithm", best_row["Algorithm"])

with col2:
    st.metric("Highest FL Accuracy", f"{best_row['Best Accuracy']:.2f}%")

with col3:
    st.metric("Best Client Setup", f"{int(best_row['Clients'])} Clients")

with col4:
    st.metric("Algorithms Compared", fl_df["Algorithm"].nunique())

st.divider()

# Client Selection
client = st.radio(
    "Select Client Configuration",
    [3, 9],
    horizontal=True
)

filtered = df[df["Clients"] == client]

# Chart
fig = px.bar(
    filtered,
    x="Algorithm",
    y="Best Accuracy",
    color="Algorithm",
    text="Best Accuracy",
    title=f"Best Accuracy Comparison ({client} Clients)"
)

fig.add_hline(
    y=96.34,
    line_dash="dash",
    line_color="red",
    annotation_text="Centralized Accuracy (96.34%)",
    annotation_position="top left"
)

fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

fig.update_layout(
    showlegend=False,
    yaxis_title="Accuracy (%)",
    xaxis_title="Algorithm",
    yaxis_range=[0,100],
    height=500
)

st.plotly_chart(fig, use_container_width=True)

st.divider()
# Key Insights
st.subheader("📌 Key Insights")

best = filtered.loc[filtered["Best Accuracy"].idxmax()]
worst = filtered.loc[filtered["Best Accuracy"].idxmin()]

gap = 96.34 - best["Best Accuracy"]

st.success(
    f"{best['Algorithm']} achieved the highest accuracy "
    f"({best['Best Accuracy']:.2f}%) using {client} clients."
)

st.info(
    f"The best federated model is only {gap:.2f}% behind "
    f"the centralized model."
)

st.warning(
    f"{worst['Algorithm']} recorded the lowest accuracy "
    f"({worst['Best Accuracy']:.2f}%)."
)
# Ranking 
st.subheader("🏆 Algorithm Ranking")

ranking = (
    filtered.sort_values("Best Accuracy", ascending=False)
    [["Algorithm", "Best Accuracy"]]
)

for i, row in enumerate(ranking.itertuples(), start=1):
    st.write(
        f"**{i}. {row.Algorithm}** — {row._2:.2f}%"
    )

st.divider()

# Raw Data
with st.expander("View Experimental Results"):
    st.dataframe(df, use_container_width=True)
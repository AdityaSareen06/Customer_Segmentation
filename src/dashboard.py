import streamlit as st
import pandas as pd
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("📊 Customer Segmentation: Mismatched Clusters Analysis")

# Load Data
mismatch_analysis = pd.read_csv("data/mismatched_customers_analysis.csv", index_col=0)
spending_behavior = pd.read_csv("data/mismatched_cluster_vs_spending_behavior.csv")
overall_metrics = pd.read_csv("data/cluster_comparison_synthetic.csv")
raw_data = pd.read_csv("data/mismatched_customers_synthetic.csv")

# 🔹 Sidebar Filters
st.sidebar.header("🔍 Filter Mismatched Customers")

# Cluster Filter
selected_cluster = st.sidebar.multiselect(
    "Filter by K-Means Cluster:",
    options=raw_data["KMeans_Cluster"].unique(),
    default=raw_data["KMeans_Cluster"].unique()
)

# Spending Behavior Filter
selected_behavior = st.sidebar.multiselect(
    "Filter by Spending Behavior:",
    options=raw_data["Spending_Behavior_kmeans"].unique(),
    default=raw_data["Spending_Behavior_kmeans"].unique()
)

# Apply Filters
filtered_data = raw_data[
    (raw_data["KMeans_Cluster"].isin(selected_cluster)) &
    (raw_data["Spending_Behavior_kmeans"].isin(selected_behavior))
]

# 🔍 Mismatched Customers Overview
st.header("🔍 Overview of Mismatched Customers")
st.dataframe(mismatch_analysis, use_container_width=True)


# 📈 Select Feature for Boxplot Analysis
st.header("📊 Feature Distribution of Filtered Customers")
feature = st.selectbox("Select Feature:", ['Age_kmeans', 'Annual_Income_k_kmeans', 'Spending_Score_1100_kmeans'])

# Interactive Boxplot
fig = px.box(filtered_data, y=feature, title=f"{feature} Distribution of Mismatched Customers", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# 💰 Spending Behavior Analysis (Fixed)
st.header("💰 Spending Behavior Distribution")
fig_spending = px.bar(spending_behavior, x="Spending_Behavior_kmeans", y="Count",
                      title="Spending Behavior of Mismatched Customers", color="Spending_Behavior_kmeans",
                      template="plotly_dark")
st.plotly_chart(fig_spending, use_container_width=True)

# 📌 Model Performance Comparison Table
st.header("📌 Model Performance Metrics Comparison")
st.markdown("Comparison of **K-Means (Cluster 1)** and **Hierarchical Clustering (Cluster 2)** models")
overall_metrics.columns = ['Cluster', 'K-Means', 'Hierarchical']
st.table(overall_metrics)

# 🔑 Key Insights Section
st.header("🔑 Key Insights")
st.markdown("""
- Most mismatched customers fall into the **low spending category**.
- A significant portion are from **senior and middle-aged groups**.
- A small subset of customers shows **extremely high spending scores**, indicating potential sub-segments or unique customer behaviors (outliers).
- Income levels among mismatched customers are **relatively uniform**, suggesting income isn't the main driver of mismatch.
- Hierarchical clustering grouped more customers into cluster 0 compared to K-Means, indicating **varying sensitivity** of models in cluster assignment.
- Age alone appears **insufficient** for clear customer segmentation given the broad range (20-69 years) among mismatched customers.
- Outliers in Annual Income and Spending Scores reveal customers whose unique profiles might require personalized marketing strategies.
""")

# 🔎 Show Raw Data Option
st.header("🔎 Explore Mismatched Customers Data")
if st.checkbox("Show raw mismatched customer data"):
    st.dataframe(raw_data, use_container_width=True)

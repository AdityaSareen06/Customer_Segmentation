import pandas as pd

# Load clustered data
df_kmeans = pd.read_csv("data/customer_segmentation_clustered_synthetic.csv")
df_hierarchical = pd.read_csv("data/customer_segmentation_hierarchical_synthetic.csv")

# Rename cluster columns for clarity
df_kmeans.rename(columns={"Cluster": "KMeans_Cluster"}, inplace=True)
df_hierarchical.rename(columns={"Hierarchical_Cluster": "Hierarchical_Cluster"}, inplace=True)

# Merge datasets on CustomerID
df_comparison = df_kmeans.merge(df_hierarchical, on="CustomerID", suffixes=("_kmeans", "_hierarchical"))

# 🔹 Extract only the 39 mismatched customers (Cluster 1 in K-Means, Cluster 2 in Hierarchical)
df_mismatched_39 = df_comparison[
    (df_comparison["KMeans_Cluster"] == 1) & (df_comparison["Hierarchical_Cluster"] == 2)
]

# ✅ Save these 39 mismatched customers
df_mismatched_39.to_csv("data/mismatched_customers_synthetic.csv", index=False)
print(f"✅ {len(df_mismatched_39)} mismatched customers saved to 'data/mismatched_customers_synthetic.csv'.")

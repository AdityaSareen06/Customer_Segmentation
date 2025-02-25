import pandas as pd

# Load the datasets
df_kmeans = pd.read_csv("data/customer_segmentation_clustered_synthetic.csv")
df_hierarchical = pd.read_csv("data/customer_segmentation_hierarchical_synthetic.csv")

# Print available columns to verify correctness
print("K-Means Columns:", df_kmeans.columns)
print("Hierarchical Columns:", df_hierarchical.columns)

# Ensure 'Cluster' is the correct name for KMeans and 'Hierarchical_Cluster' for hierarchical
df_comparison = df_kmeans.merge(df_hierarchical, on="CustomerID", suffixes=("_KMeans", "_Hierarchical"))

# Generate the comparison table
cluster_comparison = pd.crosstab(df_comparison["Cluster"], df_comparison["Hierarchical_Cluster"])

# Display comparison results
print("\n🔹 **Cluster Comparison Table** 🔹")
print(cluster_comparison)

# Save to CSV for further analysis
cluster_comparison.to_csv("data/cluster_comparison_synthetic.csv")
print("✅ Cluster comparison saved successfully!")

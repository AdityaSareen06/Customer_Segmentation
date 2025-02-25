import pandas as pd

# Load clustered datasets
df_kmeans = pd.read_csv("data/customer_segmentation_clustered.csv")
df_hierarchical = pd.read_csv("data/customer_segmentation_hierarchical.csv")

# Print unique cluster values for both methods
print("K-Means Clusters:", df_kmeans["Cluster"].unique())
print("Hierarchical Clusters:", df_hierarchical["Hierarchical_Cluster"].unique())

# Ensure Hierarchical Clusters are in the same range (0,1 instead of 1,2)
if df_hierarchical["Hierarchical_Cluster"].min() == 1:
    df_hierarchical["Hierarchical_Cluster"] -= 1  # Shift values to match K-Means (0,1)
    df_hierarchical.to_csv("data/customer_segmentation_hierarchical.csv", index=False)
    print("✅ Hierarchical clusters adjusted to match K-Means (Shifted by -1).")
else:
    print("✅ Hierarchical clusters already aligned with K-Means.")

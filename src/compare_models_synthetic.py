from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd

# Load clustered data
df_kmeans = pd.read_csv("data/customer_segmentation_clustered_synthetic.csv")
df_hierarchical = pd.read_csv("data/customer_segmentation_hierarchical_synthetic.csv")

# Feature columns used for clustering (excluding CustomerID and Cluster labels)
feature_columns = ["Age", "Annual_Income_k", "Spending_Score_1100", "Income_to_Spending_Ratio"]

# Compute evaluation metrics for K-Means
silhouette_kmeans = silhouette_score(df_kmeans[feature_columns], df_kmeans["Cluster"])
davies_bouldin_kmeans = davies_bouldin_score(df_kmeans[feature_columns], df_kmeans["Cluster"])
calinski_harabasz_kmeans = calinski_harabasz_score(df_kmeans[feature_columns], df_kmeans["Cluster"])

# Compute evaluation metrics for Hierarchical Clustering
silhouette_hierarchical = silhouette_score(df_hierarchical[feature_columns], df_hierarchical["Hierarchical_Cluster"])
davies_bouldin_hierarchical = davies_bouldin_score(df_hierarchical[feature_columns], df_hierarchical["Hierarchical_Cluster"])
calinski_harabasz_hierarchical = calinski_harabasz_score(df_hierarchical[feature_columns], df_hierarchical["Hierarchical_Cluster"])

# Display results
results = pd.DataFrame({
    "Metric": ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"],
    "K-Means": [silhouette_kmeans, davies_bouldin_kmeans, calinski_harabasz_kmeans],
    "Hierarchical Clustering": [silhouette_hierarchical, davies_bouldin_hierarchical, calinski_harabasz_hierarchical]
})

print(results)

import pandas as pd

# Load clustered synthetic datasets
df_kmeans = pd.read_csv("data/customer_segmentation_clustered_synthetic.csv")  
df_hierarchical = pd.read_csv("data/customer_segmentation_hierarchical_synthetic.csv")  

# Standardize cluster column names
df_kmeans.rename(columns={'Cluster': 'KMeans_Cluster'}, inplace=True)
df_hierarchical.rename(columns={'Hierarchical_Cluster': 'Hierarchical_Cluster'}, inplace=True)

# Ensure Hierarchical Cluster IDs start from 0 (match K-Means)
if df_hierarchical["Hierarchical_Cluster"].min() == 1:
    df_hierarchical["Hierarchical_Cluster"] -= 1  

# Select Numeric and Categorical Features
numeric_features = ['Age', 'Annual_Income_k', 'Spending_Score_1100', 'Income_to_Spending_Ratio']
categorical_features = ['Age_Group', 'Spending_Behavior', 'Savings_Potential', 'Spending_Type']

# Function to Summarize Clusters
def summarize_clusters(df, cluster_col):
    numeric_summary = df.groupby(cluster_col)[numeric_features].agg(['mean', 'std'])
    numeric_summary.columns = ['_'.join(col) for col in numeric_summary.columns]  # Flatten multi-index

    categorical_summary = df.groupby(cluster_col)[categorical_features].agg(lambda x: x.mode()[0])
    return pd.concat([numeric_summary, categorical_summary], axis=1)  # Combine both

# Generate Profiles
kmeans_summary = summarize_clusters(df_kmeans, "KMeans_Cluster")
hierarchical_summary = summarize_clusters(df_hierarchical, "Hierarchical_Cluster")

# Save Cleaned Profile Data
kmeans_summary.to_csv("data/kmeans_cluster_profile_synthetic.csv")
hierarchical_summary.to_csv("data/hierarchical_cluster_profile_synthetic.csv")

print("✅ Cluster profiling (synthetic) completed & saved with proper formatting!")

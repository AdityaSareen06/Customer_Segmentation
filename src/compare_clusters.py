import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load clustered datasets
df_kmeans = pd.read_csv("data/customer_segmentation_clustered.csv")
df_hierarchical = pd.read_csv("data/customer_segmentation_hierarchical.csv")

# Standardize column names
df_kmeans.columns = df_kmeans.columns.str.replace(' ', '_').str.replace(r'\W', '', regex=True)
df_hierarchical.columns = df_hierarchical.columns.str.replace(' ', '_').str.replace(r'\W', '', regex=True)

# Create subplots for side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# K-Means Clustering Scatter Plot
sns.scatterplot(
    x=df_kmeans["Annual_Income_k"], y=df_kmeans["Spending_Score_1100"],
    hue=df_kmeans["Cluster"], palette="viridis", ax=axes[0]
)
axes[0].set_title("K-Means Clustering", fontsize=14)
axes[0].set_xlabel("Annual Income (k$)")
axes[0].set_ylabel("Spending Score (1-100)")

# Hierarchical Clustering Scatter Plot
sns.scatterplot(
    x=df_hierarchical["Annual_Income_k"], y=df_hierarchical["Spending_Score_1100"],
    hue=df_hierarchical["Hierarchical_Cluster"], palette="Set1", ax=axes[1]
)
axes[1].set_title("Hierarchical Clustering", fontsize=14)
axes[1].set_xlabel("Annual Income (k$)")
axes[1].set_ylabel("Spending Score (1-100)")

# Adjust layout
plt.tight_layout()
plt.show()

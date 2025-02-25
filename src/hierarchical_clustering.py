import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
import numpy as np

# Load dataset with synthetic feature
df = pd.read_csv("data/customer_segmentation_synthetic.csv")

# Standardize column names
df.columns = df.columns.str.replace(' ', '_').str.replace(r'\W', '', regex=True)

# Selecting features for clustering
X = df[['Annual_Income_k', 'Spending_Score_1100', 'Income_to_Spending_Ratio']]

# Create linkage matrix
linkage_matrix = sch.linkage(X, method='ward')

# Compute the largest jump in distances for optimal cut
distances = linkage_matrix[:, 2]
sorted_jumps = np.diff(distances)
largest_jump_idx = np.argmax(sorted_jumps)
optimal_distance = distances[largest_jump_idx]

# Determine number of clusters
num_clusters = len(set(fcluster(linkage_matrix, optimal_distance, criterion='distance')))

# Assign clusters
df["Hierarchical_Cluster"] = fcluster(linkage_matrix, optimal_distance, criterion='distance')

# Save new hierarchical clustered dataset
df.to_csv("data/customer_segmentation_hierarchical_synthetic.csv", index=False)

# Plot dendrogram with new cutoff
plt.figure(figsize=(16, 7))
dendrogram = sch.dendrogram(
    linkage_matrix,
    leaf_rotation=90,  
    leaf_font_size=10,  
    color_threshold=optimal_distance,  
    truncate_mode='level',  
    p=5  
)

plt.axhline(y=optimal_distance, color='red', linestyle='--', label=f'Optimal k Cutoff (y={optimal_distance:.2f})')

plt.title('Dendrogram for Hierarchical Clustering', fontsize=16)
plt.xlabel('Customer Groups', fontsize=14)
plt.ylabel('Euclidean Distance', fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

print(f"🔹 Detected Optimal k: {num_clusters} (Based on largest jump in distances)")
print("✅ Hierarchical clustering completed and saved!")

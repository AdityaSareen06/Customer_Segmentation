import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Load dataset with new feature
df = pd.read_csv("data/customer_segmentation_synthetic.csv")

# Selecting features for clustering
X = df[['Annual_Income_k', 'Spending_Score_1100', 'Income_to_Spending_Ratio']]


# Finding the optimal number of clusters
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    
    # Compute silhouette score
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

# Determine optimal k
inertia_diff = np.diff(inertia, 2)  # Second derivative for elbow method
optimal_k_elbow = K_range[np.argmin(inertia_diff) + 1]  # +1 to align with K_range
optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
optimal_k = optimal_k_silhouette

print(f"Optimal k (Elbow Method): {optimal_k_elbow}")
print(f"Optimal k (Silhouette Score): {optimal_k_silhouette}")
print(f"Final Chosen k: {optimal_k}")

# Plot Elbow & Silhouette
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o', color='r')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')

plt.tight_layout()
plt.show()

# Apply K-Means using the new optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Save new clustered dataset
df.to_csv("data/customer_segmentation_clustered_synthetic.csv", index=False)

print("✅ K-Means clustering completed and saved!")

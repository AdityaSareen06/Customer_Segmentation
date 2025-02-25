import pandas as pd
import os

# File paths
kmeans_file = "data/kmeans_cluster_profile.csv"
hierarchical_file = "data/hierarchical_cluster_profile.csv"

# Check if files exist
if not os.path.exists(kmeans_file) or not os.path.exists(hierarchical_file):
    print("❌ Cluster profile files not found! Please run `profile_clusters.py` first.")
    exit()

# Load profiling results
kmeans_profile = pd.read_csv(kmeans_file, index_col=0, header=[0, 1])
hierarchical_profile = pd.read_csv(hierarchical_file, index_col=0, header=[0, 1])

# Standardize column names (Remove spaces and special characters)
kmeans_profile.columns = [col[0].strip().replace(" ", "_").replace("(", "").replace(")", "").replace("$", "") + "_" + col[1].strip() for col in kmeans_profile.columns]
hierarchical_profile.columns = [col[0].strip().replace(" ", "_").replace("(", "").replace(")", "").replace("$", "") + "_" + col[1].strip() for col in hierarchical_profile.columns]

# Print column names for debugging
print("K-Means Profile Columns:", kmeans_profile.columns.tolist())
print("Hierarchical Profile Columns:", hierarchical_profile.columns.tolist())

### **🔹 Step 1: Force Correct Cluster Alignment**
# Get cluster means
kmeans_income = kmeans_profile["Annual_Income_k_mean"]
hierarchical_income = hierarchical_profile["Annual_Income_k_mean"]

# Check if clusters need to be swapped
if hierarchical_income[0] > hierarchical_income[1]:  
    # Hierarchical Cluster 0 has higher income, swap needed
    hierarchical_profile = hierarchical_profile.rename(index={0: 1, 1: 0})
    hierarchical_profile = hierarchical_profile.sort_index()
    print("✅ Hierarchical clusters swapped to match K-Means.")

### **🔹 Step 2: Generate the Summary**
def generate_summary(profile, method_name):
    summary = f"🔹 **{method_name} Clustering Summary** 🔹\n"

    for cluster in profile.index:
        # Dynamically find correct columns
        age_col = [col for col in profile.columns if "Age_mean" in col][0]
        income_col = [col for col in profile.columns if "Annual_Income" in col and "mean" in col][0]
        spending_col = [col for col in profile.columns if "Spending_Score" in col and "mean" in col][0]

        age_mean = profile.loc[cluster, age_col]
        income_mean = profile.loc[cluster, income_col]
        spending_mean = profile.loc[cluster, spending_col]

        summary += f"\n**Cluster {cluster}:**\n"
        summary += f"- 📌 Average Age: {age_mean:.1f} years\n"
        summary += f"- 💰 Average Annual Income: {income_mean:.1f} k$\n"
        summary += f"- 🛍️ Average Spending Score: {spending_mean:.1f}\n"

        # Add possible interpretations
        if spending_mean > 50:
            summary += "  - 🟢 **High Spending Customers**\n"
        else:
            summary += "  - 🔵 **Moderate Spending Customers**\n"

        if income_mean > 100:
            summary += "  - 💳 **High-Income Group**\n"
        else:
            summary += "  - 🏡 **Middle/Lower Income Group**\n"

    return summary

# Generate summaries
kmeans_summary = generate_summary(kmeans_profile, "K-Means")
hierarchical_summary = generate_summary(hierarchical_profile, "Hierarchical")

# Save summaries to a text file
with open("data/cluster_summaries.txt", "w") as f:
    f.write(kmeans_summary + "\n\n" + hierarchical_summary)

# Print output
print("\n✅ Cluster summaries updated and saved to `data/cluster_summaries.txt`.")

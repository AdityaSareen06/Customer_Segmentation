import pandas as pd

# Load the 39 mismatched customers
df_mismatched = pd.read_csv("data/mismatched_customers_synthetic.csv")

# Select relevant numerical columns for analysis
numerical_cols = ["Age_kmeans", "Annual_Income_k_kmeans", "Spending_Score_1100_kmeans"]

# Generate summary statistics
summary_stats = df_mismatched[numerical_cols].describe()

# Save the summary statistics
summary_stats.to_csv("data/mismatched_customers_analysis.csv")
print(f"✅ Summary statistics saved to 'data/mismatched_customers_analysis.csv'.")

# Count distribution by age group
age_group_counts = df_mismatched["Age_Group_kmeans"].value_counts()
print("\n📊 Age Group Distribution:\n", age_group_counts)

# Count distribution by spending behavior
spending_behavior_counts = df_mismatched["Spending_Behavior_kmeans"].value_counts()
print("\n💰 Spending Behavior Distribution:\n", spending_behavior_counts)

# Identify why mismatches occur
# Compare cluster assignments with spending patterns
comparison = df_mismatched.groupby(["KMeans_Cluster", "Spending_Behavior_kmeans"]).size().reset_index(name="Count")

# Save cluster vs spending behavior comparison
comparison.to_csv("data/mismatched_cluster_vs_spending_behavior.csv", index=False)
print(f"✅ Cluster-Spending Behavior analysis saved to 'data/mismatched_cluster_vs_spending_behavior.csv'.")

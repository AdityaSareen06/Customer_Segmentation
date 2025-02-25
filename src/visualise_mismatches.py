import pandas as pd
import matplotlib.pyplot as plt

# 📌 Load mismatched customers analysis
file_path = "data/mismatched_customers_analysis.csv"
mismatched_analysis_df = pd.read_csv(file_path, index_col=0)

# Keep only statistical rows and convert to numeric
mismatched_analysis_df = mismatched_analysis_df.loc[["mean", "min", "25%", "50%", "75%", "max"]]
mismatched_analysis_df = mismatched_analysis_df.apply(pd.to_numeric)

# 📊 Plot Age Distribution
plt.figure(figsize=(8, 5))
plt.plot(mismatched_analysis_df.index, mismatched_analysis_df["Age_kmeans"], marker='o', label="Age")
plt.xlabel("Statistic")
plt.ylabel("Age")
plt.title("Age Distribution of Mismatched Customers")
plt.legend()
plt.grid()
plt.show()

# 📊 Plot Annual Income Distribution
plt.figure(figsize=(8, 5))
plt.plot(mismatched_analysis_df.index, mismatched_analysis_df["Annual_Income_k_kmeans"], marker='o', label="Annual Income")
plt.xlabel("Statistic")
plt.ylabel("Annual Income (in k)")
plt.title("Annual Income Distribution of Mismatched Customers")
plt.legend()
plt.grid()
plt.show()

# 📊 Plot Spending Score Distribution
plt.figure(figsize=(8, 5))
plt.plot(mismatched_analysis_df.index, mismatched_analysis_df["Spending_Score_1100_kmeans"], marker='o', label="Spending Score")
plt.xlabel("Statistic")
plt.ylabel("Spending Score (out of 100)")
plt.title("Spending Score Distribution of Mismatched Customers")
plt.legend()
plt.grid()
plt.show()

# 📌 Load mismatched spending behavior distribution
spending_behavior_file = "data/mismatched_cluster_vs_spending_behavior.csv"
spending_behavior_df = pd.read_csv(spending_behavior_file)

# 📊 Plot Spending Behavior Distribution
plt.figure(figsize=(8, 5))
plt.bar(spending_behavior_df["Spending_Behavior_kmeans"], spending_behavior_df["Count"], color=['blue', 'orange', 'green'])
plt.xlabel("Spending Behavior")
plt.ylabel("Number of Customers")
plt.title("Spending Behavior Distribution of Mismatched Customers")
plt.grid(axis='y')
plt.show()

print("✅ All visualizations generated successfully.")

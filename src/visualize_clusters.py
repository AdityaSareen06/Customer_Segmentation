import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load clustered dataset
df = pd.read_csv("data/customer_segmentation_clustered.csv")

# Rename columns if needed
df.rename(columns={"Annual Income (k$)": "Annual_Income", "Spending Score (1-100)": "Spending_Score"}, inplace=True)

# Scatter plot of clusters with distinct colors
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Annual_Income'], y=df['Spending_Score'], hue=df['Cluster'], palette="tab10")  # Change palette

# Adjust title and legend
plt.title('Customer Segmentation')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')  # Moves legend outside

# Show plot
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/customer_segmentation_basic.csv")

# Check for missing values
print(df.isnull().sum())

# Rename columns to standardized format
df.rename(columns={'Annual Income (k$)': 'Annual_Income', 'Spending Score (1-100)': 'Spending_Score'}, inplace=True)

# Income distribution
plt.figure(figsize=(12, 5))
sns.histplot(df['Annual_Income'], bins=30, kde=True)
plt.title("Distribution of Annual Income")
plt.show()

# Income vs Spending Score
sns.scatterplot(x=df['Annual_Income'], y=df['Spending_Score'])
plt.title("Income vs Spending Score")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

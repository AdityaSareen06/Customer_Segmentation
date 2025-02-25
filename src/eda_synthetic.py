import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset with synthetic features
df = pd.read_csv("data/customer_segmentation_synthetic.csv")

# Standardize column names (if necessary)
df.columns = df.columns.str.replace(' ', '_').str.replace(r'\W', '', regex=True)

# ✅ Check for missing values
print("\n🔍 Missing Values in Each Column:\n")
print(df.isnull().sum())

# ✅ Summary statistics
print("\n📊 Summary Statistics of Synthetic Features:\n")
print(df.describe())

# ✅ Visualizing distributions of new features
synthetic_columns = [col for col in df.columns if col not in ['CustomerID', 'Age', 'Annual_Income_k', 'Spending_Score_1100']]

for col in synthetic_columns:
    plt.figure(figsize=(7, 5))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# ✅ Checking correlation between synthetic and original features
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

print("\n✅ EDA on synthetic features completed successfully!\n")

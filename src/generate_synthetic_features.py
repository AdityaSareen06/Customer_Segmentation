import pandas as pd
import numpy as np

# Load existing dataset
df = pd.read_csv("data/customer_segmentation_basic.csv")

# Standardize column names
df.columns = df.columns.str.replace(' ', '_').str.replace(r'\W', '', regex=True)

# Generate Income-to-Spending Ratio
df['Income_to_Spending_Ratio'] = df['Annual_Income_k'] / df['Spending_Score_1100']

# Generate Age Groups
def categorize_age(age):
    if age <= 30:
        return "Young"
    elif 31 <= age <= 50:
        return "Middle-Aged"
    else:
        return "Senior"

df['Age_Group'] = df['Age'].apply(categorize_age)

# Generate Spending Behavior Score
def spending_behavior(score):
    if score <= 33:
        return "Low"
    elif score <= 66:
        return "Medium"
    else:
        return "High"

df['Spending_Behavior'] = df['Spending_Score_1100'].apply(spending_behavior)

# Generate Savings Potential
df['Savings_Potential'] = df['Annual_Income_k'] - df['Spending_Score_1100']

# Generate Luxury vs. Necessity Spending
def categorize_spending_ratio(ratio):
    if ratio >= 4:
        return "High Savings"
    elif 2 <= ratio < 4:
        return "Balanced"
    else:
        return "High Spending"

df['Spending_Type'] = df['Income_to_Spending_Ratio'].apply(categorize_spending_ratio)

# Save the enhanced dataset
df.to_csv("data/customer_segmentation_advanced.csv", index=False)

print("✅ Synthetic features generated and dataset saved successfully!")

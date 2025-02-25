import pandas as pd

# Load dataset
df = pd.read_csv("data/customer_segmentation_basic.csv")

# Display basic info
print(df.head())  # First few rows
print(df.describe())  # Summary statistics
print(df.info())  # Data types and null values

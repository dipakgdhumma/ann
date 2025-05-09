# Data Wrangling I

# 1. Import Libraries
import pandas as pd
import numpy as np

# 2. Load Dataset from URL
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# 3. Initial Overview
print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Source: https://github.com/mwaskom/seaborn-data/blob/master/tips.csv")

# 4. Data Preprocessing
print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nData Description:")
print(df.describe(include='all'))

print("\nDataFrame Shape:")
print(df.shape)

# 5. Data Formatting / Normalization
print("\nData Types:")
print(df.dtypes)

# Convert 'sex', 'smoker', 'day', and 'time' to category
df['sex'] = df['sex'].astype('category')
df['smoker'] = df['smoker'].astype('category')
df['day'] = df['day'].astype('category')
df['time'] = df['time'].astype('category')

# 6. Convert Categorical to Numeric
df_encoded = pd.get_dummies(df, drop_first=True)

print("\nEncoded DataFrame:")
print(df_encoded.head())

##Part 1
import pandas as pd
import seaborn as sns

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Show first few rows
print("Dataset Preview:")
print(df[['sex', 'age', 'fare']].head())

# Drop rows where age or fare is missing
df_clean = df[['sex', 'age', 'fare']].dropna()

# Group by 'sex' and calculate statistics
summary_stats = df_clean.groupby('sex').agg(['mean', 'median', 'min', 'max', 'std'])

print("\nSummary Statistics (Grouped by Sex):")
print(summary_stats)

# Creating numeric codes for 'sex' categorical values
df_clean['sex_code'] = df_clean['sex'].map({'male': 0, 'female': 1})
print("\nList of Numeric Values for 'sex':")
print(df_clean['sex_code'].tolist())


# part 2
from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['species'] = [iris.target_names[i] for i in iris.target]

# Show preview
print("\nIris Dataset Preview:")
print(df_iris.head())

# Group by species and get descriptive stats
stats_by_species = df_iris.groupby('species').describe()

print("\nDescriptive Statistics Grouped by Species:")
print(stats_by_species)

# Optional: Display percentiles manually
percentiles = df_iris.groupby('species').quantile([0.25, 0.5, 0.75])
print("\nPercentiles (25%, 50%, 75%) for each species:")
print(percentiles)

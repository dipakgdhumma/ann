# Importing required libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the Iris dataset from seaborn
iris_data = sns.load_dataset('iris')

# 1. List down the features and their types
print("Features and their Types:")
print(iris_data.dtypes)

# 2. Create a histogram for each feature in the dataset
plt.figure(figsize=(12, 8))

# Creating histograms for each feature
for i, feature in enumerate(iris_data.columns[:-1]):  # Excluding the target variable 'species'
    plt.subplot(2, 2, i+1)
    sns.histplot(iris_data[feature], kde=True, color='skyblue', bins=20)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 3. Create a boxplot for each feature in the dataset
plt.figure(figsize=(12, 8))

# Creating boxplots for each feature
for i, feature in enumerate(iris_data.columns[:-1]):  # Excluding the target variable 'species'
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=iris_data, palette='Set2')
    plt.title(f"Boxplot of {feature} by Species")
    plt.xlabel('Species')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()

# 4. Compare distributions and identify outliers
print("\nOutliers and Distribution Observations:")
for feature in iris_data.columns[:-1]:  # Excluding 'species'
    q1 = iris_data[feature].quantile(0.25)
    q3 = iris_data[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identifying outliers
    outliers = iris_data[(iris_data[feature] < lower_bound) | (iris_data[feature] > upper_bound)]
    print(f"\nOutliers for {feature}: {outliers.shape[0]}")


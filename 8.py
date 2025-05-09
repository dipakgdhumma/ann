# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the Titanic dataset (available in seaborn)
titanic = sns.load_dataset('titanic')

# Show the first 5 rows
print(titanic.head())

# Check the shape of the dataset
print("\nShape of dataset:", titanic.shape)

# Check for null values
print("\nMissing values:\n", titanic.isnull().sum())

# Summary of dataset
print("\nDataset Info:")
print(titanic.info())


# 1. Count of survivors vs non-survivors
sns.countplot(x='survived', data=titanic)
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# 2. Survival rate by gender
sns.countplot(x='sex', hue='survived', data=titanic)
plt.title("Survival by Gender")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# 3. Survival by Passenger Class
sns.countplot(x='pclass', hue='survived', data=titanic)
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# 4. Age distribution by survival
sns.kdeplot(data=titanic, x="age", hue="survived", fill=True)
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.show()


# Plot histogram for 'fare'
sns.histplot(data=titanic, x='fare', bins=30, kde=True)
plt.title("Distribution of Ticket Fare")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()

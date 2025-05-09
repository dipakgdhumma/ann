# Importing required libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from seaborn
titanic_data = sns.load_dataset('titanic')
print(titanic_data.head())

# Checking for missing values in the dataset
missing_data = titanic_data.isnull().sum()
print("Missing values in each column:\n", missing_data)

# Dropping rows with missing age values for clarity
titanic_data = titanic_data.dropna(subset=['age'])

# Plotting a boxplot for distribution of 'age' with respect to 'sex' and 'survived'
plt.figure(figsize=(10, 6))

sns.boxplot(x='sex', y='age', hue='survived', data=titanic_data, 
            palette='Set2', showfliers=False)

# Adding labels and title
plt.title('Box Plot of Age Distribution by Gender and Survival', fontsize=14)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Age', fontsize=12)

# Adding a grid to the plot for better readability
plt.grid(True)

# Display the plot
plt.show()

# Additional Analysis: Displaying summary statistics (mean, median, IQR) for age by gender and survival status
summary_stats = titanic_data.groupby(['sex', 'survived'])['age'].describe()
print("\nSummary Statistics (Age by Gender and Survival Status):\n", summary_stats)


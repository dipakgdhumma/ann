# Data Wrangling II – Academic Performance Dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 1. Create Dataset
data = {
    'StudentID': range(1, 11),
    'Math_Score': [78, 85, 92, 88, np.nan, 95, 105, 60, 77, 999],  # 999 is an outlier
    'Science_Score': [82, 90, 94, 86, 89, 87, 88, 85, np.nan, 91],
    'Attendance (%)': [90, 85, 88, 92, 80, 95, 60, 75, 85, 99],
    'Study_Hours': [2, 3, 5, 4, 1, 6, 7, 5, 3, 4]
}

df = pd.DataFrame(data)

# 2. Handle Missing Values
df['Math_Score'].fillna(df['Math_Score'].mean(), inplace=True)
df['Science_Score'].fillna(df['Science_Score'].median(), inplace=True)

# 3. Detect Outliers using Z-score
z_scores = np.abs(stats.zscore(df[['Math_Score', 'Science_Score']]))
threshold = 3
outliers = (z_scores > threshold).any(axis=1)

# Cap Math_Score if it's an outlier
math_mean = df['Math_Score'].mean()
math_std = df['Math_Score'].std()
upper_limit = math_mean + 3 * math_std
df.loc[df['Math_Score'] > upper_limit, 'Math_Score'] = upper_limit

# 4. Data Transformation: Log Transform Study_Hours
# (Before Transformation)
sns.histplot(df['Study_Hours'], kde=True)
plt.title("Study Hours (Before Log Transformation)")
plt.show()

# Apply log transform
df['Log_Study_Hours'] = np.log(df['Study_Hours'] + 1)

# (After Transformation)
sns.histplot(df['Log_Study_Hours'], kde=True)
plt.title("Study Hours (After Log Transformation)")
plt.show()

# Final Dataset
print("\nFinal Cleaned Dataset:")
print(df)

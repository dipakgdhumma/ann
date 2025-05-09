# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
url = "https://raw.githubusercontent.com/krishnaik06/Social-Network-Ads/master/Social_Network_Ads.csv"
df = pd.read_csv("Social_Network_Ads.csv")

# Display basic info
print("Dataset preview:")
print(df.head())

# Select features (Age, EstimatedSalary) and target (Purchased)
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Split dataset (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling (very important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model
classifier = LogisticRegression()
classifier.fit(X_train_scaled, y_train)

# Predicting
y_pred = classifier.predict(X_test_scaled)

# Confusion Matrix and Evaluation
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("\nConfusion Matrix:")
print(cm)

print(f"\nTrue Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

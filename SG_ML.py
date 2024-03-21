#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc

# Load the dataset
df = pd.read_csv("bank-additional-full.csv", delimiter=';')

# Drop columns with high correlation and irrelevant columns
df = df.drop(columns=['euribor3m', 'nr.employed', 'emp.var.rate', 'contact', 'month', 'day_of_week', 'poutcome'])

# Remove rows with unknown marital status and housing
df = df[(df['marital'] != 'unknown') & (df['housing'] != 'unknown')]

# Group basic education levels together
df['education'] = df['education'].replace(['basic.4y', 'basic.6y', 'basic.9y'], 'basic.education')

# Bin age into groups
num_bins = 5
df['age_group'] = pd.cut(df['age'], bins=num_bins, labels=[f'Group {i+1}' for i in range(num_bins)])
df = df.drop(columns=['age'])

# Encode categorical features
cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'age_group', 'y']
label_encoder = LabelEncoder()
for column in cat_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Scale numerical features
num_columns = ['duration', 'campaign', 'pdays', 'previous', 'cons.price.idx', 'cons.conf.idx']
scaler = StandardScaler()
df[num_columns] = scaler.fit_transform(df[num_columns])

# Split data into features and target variable
X = df.drop(columns=['y'])
y = df['y']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine (SVM)": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Evaluate and select best performing classifier
best_classifier = None
best_accuracy = 0
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = clf

# Save the best model
joblib.dump(best_classifier, 'best_model.pkl')

# Generate ROC curve for best classifier
y_prob = best_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

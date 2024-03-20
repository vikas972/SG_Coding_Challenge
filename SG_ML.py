#!/usr/bin/env python
# coding: utf-8

# <H2> Bank Marketing Model Performance Analysis

# In[37]:


# Libraries used 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# <h2>Exploratory Data Analysis (EDA)

# In[2]:


df = pd.read_csv("bank-additional-full.csv", delimiter=';')
df.head(5)


#  <p><h4>What's the Goal?</h4></p>
# Build a Machine Learning pipeline 
# to identify customers that should be called within the weekly cohort and provide
# recommendations on how to reduce the number of cal.ls

# In[3]:


df.info()


# <h3>Univariable Check</h3>

# In[4]:


# Numerical Columns
numerical_columns = ['age', 'duration', 'campaign', 'pdays', 'previous',
                     'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                     'euribor3m', 'nr.employed']

plt.figure(figsize=(12, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 4, i)
    plt.hist(df[column], bins=20, color='skyblue', edgecolor='black', linewidth=1.5)
    plt.title(column)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[5]:


# Categorical columns
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                       'contact', 'month', 'day_of_week', 'poutcome', 'y']

plt.figure(figsize=(20, 30))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(6, 3, i)
    sns.countplot(data=df, x=column, palette='viridis')
    plt.title(column, fontsize=14, fontweight='bold') 
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xticks(rotation=45)  

plt.tight_layout()
plt.show()


# In[6]:


summary_stats = df.describe()
summary_stats


# In[7]:


# checking missing values 
df.isnull().sum()


# <h3> Bivariable Check </h3>

# In[8]:


# Bivariable check

# Categorical
plt.figure(figsize=(20, 30))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(6, 3, i)
    sns.countplot(data=df, x=column, hue='y', palette='viridis')
    plt.title(f'{column} vs. y')
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[9]:


# Numerical 

plt.figure(figsize=(15, 20))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(5, 4, i)
    sns.boxplot(data=df, x='y', y=column, palette='viridis')
    plt.title(f'{column} vs. y')
    plt.xlabel(None)
    plt.ylabel(None)

plt.tight_layout()
plt.show()


# <h3> Correlation 

# In[10]:


# correlation 

# Compute the correlation matrix
correlation_matrix = df[numerical_columns].corr()


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()


# <h2> Data Processing

# In[11]:


# Removing columns with high correlation ( More than 0.95)
variables_to_remove = ['euribor3m', 'nr.employed','emp.var.rate']  
#drop
df_filtered = df.drop(variables_to_remove, axis=1)

print(df_filtered.head(5))


# In[12]:


# checking the correlation again

# Numerical Columns
numerical_columns2 = ['age', 'duration', 'campaign', 'pdays', 'previous',
                     'cons.price.idx', 'cons.conf.idx'
                     ]

# Compute the correlation matrix
correlation_matrix = df_filtered[numerical_columns2].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()


# In[13]:


# Keeping the column I think is relevant to the model 
variables_to_remove2 = ['contact', 'month','day_of_week','poutcome']  
#drop
df_filtered = df_filtered.drop(variables_to_remove2, axis=1)
df_filtered.columns


# <h3> Categorical Treatment 

# In[14]:


df_filtered['marital'].value_counts()


# <h4> Marital treatment decision: Remove category "Unknown"

# In[15]:


df_filtered = df_filtered[df_filtered['marital'] != 'unknown']
df_filtered['marital'].value_counts()


# <h4>-- Housing 

# In[16]:


df_filtered['housing'].value_counts()


# <h4> Housing treatment decision: Remove category "Unknown"

# In[17]:


df_filtered = df_filtered[df_filtered['housing'] != 'unknown']
df_filtered['housing'].value_counts()


# <h4> -- Loan

# <h4> Loan column doesn't need to remove the unkown category because as removing from the others it also removed from the Loan column

# <h4> -- Education

# <h4> Create a new category called " basic.education" by replacing the values 'basic.4y', 'basic.6y', and 'basic.9y' 

# In[18]:


df_filtered['education'] = df_filtered['education'].replace(['basic.4y', 'basic.6y', 'basic.9y'], 'basic.education')
df_filtered['education'].value_counts()


# <h4> -- Age

# <h4>Age grouping by Equal-wigth Binning :  Divide the range of ages into a specified number of equal-width intervals. This approach ensures that each interval has the same width, but it may not capture variations in the distribution of ages.

# In[19]:


num_bins = 5
# Create equal-width bins for ages
df_filtered['age_group'] = pd.cut(df_filtered['age'], bins=num_bins, labels=[f'Group {i+1}' for i in range(num_bins)])

df_filtered['age_group'].value_counts()


# In[20]:


# Print the boundaries of each age group
print("Age Group Boundaries:")
print(df_filtered.groupby('age_group')['age'].min())
print(df_filtered.groupby('age_group')['age'].max())


# In[21]:


# removing age from df 
df_filtered = df_filtered.drop(columns=['age'])
df_filtered.head()


# <h4> Categorical Treatment 

# In[22]:


cat_columns = ['job', 'marital', 'education', 'default', 'housing',
                     'loan', 'age_group','y']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode each categorical column
for column in cat_columns:
    df_filtered[column] = label_encoder.fit_transform(df_filtered[column])

df_filtered.head()


# <h3> Numerical Treatment 

# <h4> -- Feature Scaling 

# In[23]:


num_columns = ['duration', 'campaign', 'pdays', 'previous', 'cons.price.idx', 'cons.conf.idx']

# Initialize StandardScaler
scaler = StandardScaler()

df_filtered[num_columns] = scaler.fit_transform(df_filtered[num_columns])
df_filtered.head()


# <h2> ML Model

# <h4> -- Split the data

# In[24]:


X = df_filtered.drop(columns=['y'])  # Features
y = df_filtered['y']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# <h3> Classifiers: Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting, KNN, Naive Bayes

# <h4> -- Logistic Regression

# In[25]:


from sklearn.linear_model import LogisticRegression # Library

model = LogisticRegression(random_state=42)

model.fit(X_train, y_train) # Train the model on the training set

y_pred_lr = model.predict(X_test) # Make predictions on the testing set

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred_lr)
print("Accuracy:", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(4, 2))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"fontsize":8})
plt.title('Confusion Matrix - LR')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# <h4> Note: Logistic Regression achieved an accuracy of 90.0% and demonstrated good performance in classifying the target variable. It correctly classified 6908 instances of the negative class (no) and 314 instances of the positive class (yes).

# <h4> -- Decision Tree

# In[26]:


from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier()

dt_classifier.fit(X_train, y_train) # Train the decision tree classifier

y_pred_dt = dt_classifier.predict(X_test) # Make predictions on the testing set


accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Accuracy:", accuracy_dt)

# Calculate confusion matrix
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(4, 2))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"fontsize":8})
plt.title('Confusion Matrix - DT')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# <h4> Decision Tree achieved an accuracy of 88.4%. It correctly classified 6609 instances of the negative class and 485 instances of the positive class.

# <h4> -- Random Forest

# In[27]:


from sklearn.ensemble import RandomForestClassifier # Library

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train) # Train classifier


y_pred_rf = rf_classifier.predict(X_test) # Predict on the test set

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy:", accuracy_rf)

# Calculate confusion matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(4, 2))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"fontsize":8})
plt.title('Confusion Matrix - RF')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# <h4> Random Forest achieved an accuracy of 90.2% and demonstrated robust performance. It correctly classified 6818 instances of the negative class and 420 instances of the positive class.

# <h4> -- Support Vector Machines (SVM)

# In[28]:


from sklearn.svm import SVC # Library


svm_classifier = SVC()
svm_classifier.fit(X_train, y_train) # Train classifier


y_pred_svm = svm_classifier.predict(X_test) # Predict on the test set

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy:", accuracy_svm)

# Calculate confusion matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(4, 2))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"fontsize":8})
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# <h4> SVM achieved an accuracy of 89.9%. It correctly classified 6984 instances of the negative class and 234 instances of the positive class.

# <h4> -- Gradient Boosting

# In[29]:


from sklearn.ensemble import GradientBoostingClassifier # Library


gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train, y_train) # Train classifier

y_pred_gb = gb_classifier.predict(X_test) # Predict on the test set


# Calculate accuracy
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("Accuracy:", accuracy_gb)

# Calculate confusion matrix
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
plt.figure(figsize=(4, 2))
sns.heatmap(conf_matrix_gb, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"fontsize":8})
plt.title('Confusion Matrix - NB')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# <h4> Gradient Boosting achieved an accuracy of 90.8% and demonstrated excellent performance. It correctly classified 6839 instances of the negative class and 452 instances of the positive class.

# <h4> -- KNN

# In[30]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train) # Train the KNN classifier

y_pred_knn = knn.predict(X_test) # Predict on the testing set

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)


cm_knn = confusion_matrix(y_test, y_pred_knn)

# Confusion matrix 
plt.figure(figsize=(4, 2))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"fontsize":8})
plt.title('Confusion Matrix - KNN')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# <h4>KNN achieved an accuracy of 89.2%. It correctly classified 6823 instances of the negative class and 336 instances of the positive class.

# <h4> -- Naive Bayes

# In[31]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()


nb.fit(X_train, y_train) # Train the Naive Bayes classifier
y_pred_nb = nb.predict(X_test) # Predict on the testing set

accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Accuracy:", accuracy_nb)

cm_nb = confusion_matrix(y_test, y_pred_nb)

# Confusion matrix for Naive Bayes
plt.figure(figsize=(4, 2))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"fontsize":8})
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# <h4> Naive Bayes achieved an accuracy of 88.9%. It correctly classified 6748 instances of the negative class and 389 instances of the positive class.

# <h4>Summarizing this part: Gradient Boosting achieved the highest accuracy among the classifiers tested, followed closely by Random Forest. These models demonstrated robust performance in predicting the target variable, Let's check it further 

# <h4> Double Checking if the Gradient Boosting and Random Forest are the best among others classifiers comparing through Precision, Recall, and F1 score

# In[32]:


# Define the evaluation function
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    return accuracy, precision, recall, f1

# Evaluate each classifier
classifiers = {
    "Logistic Regression": y_pred_lr,
    "Decision Tree": y_pred_dt,
    "Gradient Boosting": y_pred_gb,
    "Random Forest": y_pred_rf,
    "Support Vector Machine (SVM)": y_pred_svm,
    "K-Nearest Neighbors (KNN)": y_pred_knn,
    "Naive Bayes": y_pred_nb
}

for clf_name, y_pred in classifiers.items():
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
    print(f"{clf_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")


# <h4> Yeah, Gradient and Random Forest still the best among other classifiers.

# <h2> Feature Importance
#     
# <h4>Feature importance analysis is a technique used to determine the relative importance of each feature in predicting the target variable. 
# 
# <h3>Extracting feature importance from Gradient Boosting and Random Forest classifiers, considering they exhibited the best performance

# <h4> -- Gradient Boosting Feature Importance 

# In[33]:


feature_importances_gb = gb_classifier.feature_importances_

# Display feature importances
feature_importance_df_gb = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances_gb})
feature_importance_df_gb = feature_importance_df_gb.sort_values(by='Importance', ascending=False)
print("Gradient Boosting Feature Importance:")
print(feature_importance_df_gb)


# <h4> -- Random Forest Feature Importance

# In[34]:


feature_importances_rf = rf_classifier.feature_importances_

# Display feature importances
feature_importance_df_rf = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances_rf})
feature_importance_df_rf = feature_importance_df_rf.sort_values(by='Importance', ascending=False)
print("Random Forest Feature Importance:")
print(feature_importance_df_rf)


# <h3>  Gradient Boosting:</h3>
# 
# <p>Duration: This feature has the highest importance, indicating that the duration of the call has a significant impact on the outcome.</p>
# <p> Pdays: The number of days that passed after the client was last contacted from a previous campaign is also a crucial factor.</p>
# <p> Cons.conf.idx and Cons.price.idx: These are economic indicators, suggesting that the overall economic context plays a role. </p>
# <p> Age Group and Previous Contacts: These features have relatively lower importance but still contribute to the model. </p>
# 
# <h3>Random Forest: </h3>
# <p> Duration: Similarly, the duration of the call is the most critical predictor in the Random Forest model.
# <p> Cons.conf.idx and Cons.price.idx: Economic indicators remain significant in this model as well.
# <p> Job and Campaign: Job type and number of contacts during this campaign also have notable importance.
# <p> Education and Age Group: These features also contribute significantly to the model's predictions.

# <h3> Conclusion to this part:</h3>
# <h4>
# Both models highlight the importance of the call duration and economic indicators (cons.conf.idx and cons.price.idx).
# Other factors such as job type, education level, and age group also play essential roles in predicting the outcome of the marketing campaign.
# Overall, these insights can guide marketing strategies to focus on specific customer demographics and tailor communication strategies based on economic conditions and call duration.

# <h2> ROC Curve form Gradient Boosting and Random Forest 

# In[35]:


from sklearn.metrics import roc_curve, auc

# Compute predicted probabilities for Gradient Boosting and Random Forest
y_prob_gb = gb_classifier.predict_proba(X_test)[:, 1]
y_prob_rf = rf_classifier.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area for Gradient Boosting
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

# Compute ROC curve and ROC area for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_gb, tpr_gb, color='blue', lw=2, label=f'Gradient Boosting (AUC = {roc_auc_gb:.2f})')
plt.plot(fpr_rf, tpr_rf, color='red', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


## Since Gradient Boosting performing better let save the model to deploy.


# In[38]:


# Save the trained model to a file
joblib.dump(gb_classifier, 'gradient_boosting_model.pkl')


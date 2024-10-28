import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.datasets import make_classification
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from imblearn.under_sampling import ClusterCentroids
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('creditcard.csv')

# df.describe() # Display the data

# df.isnull().sum().max() # Check the Null value -> No Null Values!

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset') 
"""

The classes are heavily skewed we need to solve this issue later.
'Class' is the column with 0 (No Fraud) and 1 (Fraud)

"""


"""
Now I am trying to solve the imblanace problem, I will use three different methods to solve this issue.
Oversampling:
1. SMOTE & ADASYN
2. Bootstrap & Bagging

Undersampling:
1. Cluster Centriod (Logistic Regression using Cluster Centroid Resampled Data)

I will plot the class distribution after using two methods and 
also print the classiciation report which containing the precision, recall, f1-score and support
to compare two methods.
"""
# Setup
    # Features (X): All columns except 'Class'
X = df.drop(columns=['Class'])

    # Target (y): The 'Class' column
y = df['Class']

    # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Method1
    # Initialize SMOTE

smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check the class distribution after SMOTE
print("Class distribution after SMOTE:", dict(zip(*np.unique(y_train_smote, return_counts=True))))


    # Initialize ADASYN

adasyn = ADASYN(random_state=42)

# Apply ADASYN to the training data
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

# Check the class distribution after ADASYN
print("Class distribution after ADASYN:", dict(zip(*np.unique(y_train_adasyn, return_counts=True))))

# Initialize a classifier (Random Forest in this example)
clf = RandomForestClassifier(random_state=42)

# Train on the SMOTE-resampled dataset
clf.fit(X_train_smote, y_train_smote)

# Predict on the test set
y_pred_smote = clf.predict(X_test)

# Evaluate the classifier performance
conf_matrix = confusion_matrix(y_test, y_pred_smote)
ConfusionMatrixDisplay(conf_matrix).plot()
plt.title("SMOTE Confusion Matrix")
plt.show()

print("SMOTE Classification Report:")
print(classification_report(y_test, y_pred_smote))

# Similarly, train and evaluate the classifier using the ADASYN-resampled dataset
clf.fit(X_train_adasyn, y_train_adasyn)
y_pred_adasyn = clf.predict(X_test)

# Evaluate the classifier performance
conf_matrix = confusion_matrix(y_test, y_pred_adasyn)
ConfusionMatrixDisplay(conf_matrix).plot()
plt.title("ADASYN Confusion Matrix")
plt.show()

print("ADASYN Classification Report:")
print(classification_report(y_test, y_pred_adasyn))


# Plot the distribution of the target variable before and after resampling
plt.figure(figsize=(18, 12))

# Original Dataset Distribution
plt.subplot(3, 1, 1)
sns.countplot(x=y, palette='Blues')
plt.title("Original Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

# SMOTE-Resampled Dataset Distribution
plt.subplot(3, 1, 2)
sns.countplot(x=y_train_smote, palette='Greens')
plt.title("SMOTE-Resampled Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

# ADASYN-Resampled Dataset Distribution
plt.subplot(3, 1, 3)
sns.countplot(x=y_train_adasyn, palette='Reds')
plt.title("ADASYN-Resampled Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.tight_layout()
plt.show()




# Method 2
    # Bootstrap & Bagging
# Bootstrap Sampling to balance the classes in the training set
X_minority = X_train[y_train == 1]
y_minority = y_train[y_train == 1]

X_minority_bootstrap, y_minority_bootstrap = resample(X_minority, y_minority, replace=True, n_samples=len(y_train[y_train == 0]), random_state=42)

# Combine the majority class with the upsampled minority class
X_train_balanced = pd.concat([X_train[y_train == 0], X_minority_bootstrap])
y_train_balanced = pd.concat([y_train[y_train == 0], y_minority_bootstrap])

# Bagging Classifier
bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
bagging_clf.fit(X_train_balanced, y_train_balanced)

# Predict on the test set
y_pred = bagging_clf.predict(X_test)

# Plot the distribution of the target variable before and after bootstrap sampling
plt.figure(figsize=(12, 8))

# Original Dataset Distribution
plt.subplot(2, 1, 1)
sns.countplot(x=y, palette='Blues')
plt.title("Original Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

# Bootstrap-Resampled Dataset Distribution
plt.subplot(2, 1, 2)
sns.countplot(x=y_train_balanced, palette='Oranges')
plt.title("Balanced Training Set Class Distribution after Bootstrap Sampling")
plt.xlabel("Class")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_matrix).plot()
plt.title("Bootstrap Confusion Matrix")
plt.show()

# Classification Report
print("Bootstrap Classification Report:")
print(classification_report(y_test, y_pred))



# Method 3:
    #Cluster Centroid
# Cluster Centroid Method to balance the classes in the training set
cc = ClusterCentroids(random_state=42)
X_train_cc, y_train_cc = cc.fit_resample(X_train, y_train)

# Plot Cluster Centroid Resampled Dataset Distribution
plt.subplot(3, 1, 3)
sns.countplot(x=y_train_cc, palette='Purples')
plt.title("Balanced Training Set Class Distribution after Cluster Centroid Method")
plt.xlabel("Class")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# Logistic Regression using Cluster Centroid Resampled Data
logistic_clf = LogisticRegression(random_state=42)
logistic_clf.fit(X_train_cc, y_train_cc)

# Predict on the test set
y_pred_cc = logistic_clf.predict(X_test)

# Confusion Matrix for Cluster Centroid Method
conf_matrix_cc = confusion_matrix(y_test, y_pred_cc)
ConfusionMatrixDisplay(conf_matrix_cc).plot()
plt.title("Confusion Matrix (Cluster Centroid Method)")
plt.show()

# Classification Report for Cluster Centroid Method
print("Classification Report (Cluster Centroid Method):")
print(classification_report(y_test, y_pred_cc))


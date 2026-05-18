# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import all necessary packages and dataset: Import the required libraries (like pandas, numpy, and sklearn) and load the dataset needed to implement Logistic Regression.
2.Clean the dataset: Create a copy of the actual dataset and remove any fields that are unnecessary for the analysis.

3.Define variables: Select the dependent variable (target) and independent variables (features) from the processed dataset.

4.Perform Logistic Regression: Execute the Logistic Regression algorithm on the prepared data.

5.Evaluate results: Print the values for the confusion matrix, accuracy score, and classification report to determine whether the student is placed or not.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by:MONISHA.A.K 
RegisterNumber: 212225230187 
*/
```import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load iris dataset
iris = load_iris()

# Create dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SGD Classifier model
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# Train the model
sgd_clf.fit(X_train, y_train)

# Predict using test data
y_pred = sgd_clf.predict(X_test)

# Find accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix using heatmap
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt='d',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()
```

## Output:
<img width="1210" height="863" alt="Screenshot 2026-05-18 153028" src="https://github.com/user-attachments/assets/f3296a65-b7c6-49c4-8ff1-bc143ff49aef" />
<img width="841" height="632" alt="Screenshot 2026-05-18 153042" src="https://github.com/user-attachments/assets/42f5f05d-7ad2-4d3c-86a0-fff721b0707c" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

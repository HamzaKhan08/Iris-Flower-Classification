# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import datasets

print("All imports successful!")

# Loading the Iris dataset
iris = datasets.load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['target'])

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Explore data using descriptive statistics
print("\nDescriptive statistics:")
print(data.describe())

# Visualize data using pairplot
sns.pairplot(data, hue='target', palette='Set1', diag_kind='kde')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Splitting the data into training and testing sets
X = data.iloc[:, :-1]  # Features: sepal length, sepal width, petal length, petal width
y = data['target']     # Target: species
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the model:", accuracy)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris['target_names'], yticklabels=iris['target_names'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

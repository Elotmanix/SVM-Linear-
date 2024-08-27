import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from SVM import LinearSVM



# Create the dataset with 2 features, all informative
X, y = datasets.make_classification(
    n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, random_state=42, class_sep=2)

# Adjust labels to be -1 and 1 for SVM
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Create the Logistic Regression model
model = LinearSVM(lr=0.01, iterations=1000)

# Fit the model on the data
model.fit(X_train, y_train)

# Output final parameters
print(f"Final parameters: W = {model.W}, b = {model.b}")

# Predict and classify the data
predictions = model.predict(X_test)

print("Accuracy: ",model.accuracy(y_test, predictions))  




# Plotting
plt.figure(figsize=(8, 4))

# Plot the training points
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', marker='o', label='Class 1')
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='red', marker='x', label='Class -1')

# Plot the decision boundary
x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_values = - (model.W[0] * x_values + model.b) / model.W[1]
plt.plot(x_values, y_values, color='green', label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Linear Decision Boundary')
plt.legend()
plt.savefig("svm_decision_boundary.png", dpi=2000)
plt.show()


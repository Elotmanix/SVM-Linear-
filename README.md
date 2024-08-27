# Linear SVM Implementation

This project provides an implementation of a Linear Support Vector Machine (SVM) from scratch using Python and NumPy. The SVM model is designed to perform binary classification tasks.

## Files

1. **`SVM.py`**: Contains the `LinearSVM` class with methods to train, predict, and evaluate a linear SVM model.

2. **`implementation.py`**: Another implementation of the `LinearSVM` class with similar functionality as `SVM.py`.

3. **`image_svm.png`**: An image illustrating the concept of Support Vector Machines (SVM).

## LinearSVM Class

### Methods

- **`__init__(self, lr=0.01, lambda_parametre=0.01, iterations=100)`**: Initializes the Linear SVM model with learning rate (`lr`), regularization parameter (`lambda_parametre`), and number of iterations (`iterations`).

- **`DecisionFunction(self, X)`**: Computes the decision function for given input data `X`.

- **`predict(self, X)`**: Predicts the class labels for the input data `X` based on the decision function.

- **`LossFunction(self, X, y)`**: Calculates the hinge loss for the given data `X` and labels `y`.

- **`gradient_descent(self, X, y)`**: Performs gradient descent to optimize the weights and bias.

- **`fit(self, X, y)`**: Fits the model to the data `X` and labels `y` using gradient descent.

- **`accuracy(self, y_true, y_pred)`**: Computes the accuracy of the model by comparing true labels `y_true` with predicted labels `y_pred`.

## Usage

1. **Train the Model**: Use the `fit` method to train the model with your data.
    ```python
    from SVM import LinearSVM
    # or
    # from implementation import LinearSVM

    # Create an instance of the LinearSVM class
    model = LinearSVM(lr=0.01, lambda_parametre=0.01, iterations=1000)

    # Train the model with your data
    model.fit(X_train, y_train)
    ```

2. **Make Predictions**: Use the `predict` method to make predictions on new data.
    ```python
    predictions = model.predict(X_test)
    ```

3. **Evaluate the Model**: Use the `accuracy` method to evaluate the performance of the model.
    ```python
    accuracy = model.accuracy(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
    ```

## Image

- **`image_svm.png`**: Provides a visual representation of how Support Vector Machines work.

## Requirements

- Python 3.7
- NumPy


# SVM-Linear-

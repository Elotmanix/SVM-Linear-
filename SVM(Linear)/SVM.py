import numpy as np



class LinearSVM:
    def __init__(self, lr=0.01, lambda_parametre=0.01, iterations=100):
        self.lr = lr
        self.lambda_parametre = lambda_parametre
        self.iterations = iterations
        self.W = None
        self.b = None

    def DecisionFunction(self, X):
        return np.dot(X, self.W) - self.b
    
    def predict(self, X):
        decision_values = self.DecisionFunction(X)
        return np.sign(decision_values)
    
    def LossFunction(self, X, y):
        distances = 1 - y * self.DecisionFunction(X)
        hinge_loss = np.maximum(0, distances)
        loss = (1 / 2) * np.linalg.norm(self.W) ** 2 + self.lambda_parametre * np.sum(hinge_loss)
        return loss
    
    def gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0
        
        for i in range(self.iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * self.DecisionFunction(x_i) >= 1
                if condition:
                    self.W -= self.lr * (2 * self.lambda_parametre * self.W)
                else:
                    self.W -= self.lr * (2 * self.lambda_parametre * self.W - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

            if i % 100 == 0:
                cost = self.LossFunction(X, y)
                print(f"Iteration {i}: Cost {cost}")

    def fit(self, X, y):
        self.gradient_descent(X, y)

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy



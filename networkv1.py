# Version 1 of Neural Network Implementation
import numpy as np
import pandas as pd


def load_data(train_csv, test_csv):
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    X_train = train_data.drop(columns=["label"]).values / 255.0
    y_train = train_data["label"].values

    X_test = test_data.drop(columns=["label"]).values / 255.0
    y_test = test_data["label"].values

    return (X_train, y_train, X_test, y_test)


def one_hot_encode(y, num_classes=10):
    return np.eye(N=num_classes)[y]


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(z.dtype)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y, y_pred):
    return -np.mean(np.sum(y * np.log(y_pred + 1e-9), axis=1))


def accuracy(y, y_pred):
    return np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1))


class Network:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.W1 = np.random.rand(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.lr = lr

    def forward(self, X):
        """
        Z^l = W^l * A^[l-1] + b^l
        A^l = σ(Z^l)
        """
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y, y_pred):
        m = X.shape[0]

        dZ2 = y_pred - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)

            loss = cross_entropy_loss(y, y_pred)

            self.backward(X, y, y_pred)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        np.savez

    def predict(self, X):
        y_pred = self.forward(X)
        print(f"Predictions: {y_pred}")
        return np.argmax(y_pred, axis=1)

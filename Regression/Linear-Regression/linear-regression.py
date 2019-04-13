# -*- coding: utf-8 -*-
"""
Linear Regression

The dataset for this model has been downloaded from people.sc.fsu.edu - https://people.sc.fsu.edu/~jburkardt/datasets/regression/x06.txt

@author: Pradeesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, alpha=0.05, epochs=5000):
        self.alpha = alpha
		self.epochs = epochs

    def __init_params(self, n_weights):
        self.w = np.random.rand(1, n_weights)
        self.b = 0

    def __cost(self, y_true, y_pred):
        return np.square(y_pred - y_true).sum() / (2 * len(y_true))

    def fit(self, x, y):
        self.__init_params(x.shape[1])
        n = x.shape[0]
        costs = np.zeros(self.epochs)
        for i in range(self.epochs):
            # Compute Cost and loss
            y_pred = self.predict(x)
            costs[i] = self.__cost(y, y_pred)
            loss = y_pred - y

            # Update parameters
            self.w = self.w - (self.alpha / (2 * n)) * np.sum(x * loss, axis=0, keepdims=True)
            self.b = self.b - (self.alpha / (2 * n)) * loss.sum()

        return costs, self.w, self.b

    def predict(self, x):
        return np.dot(x, self.w.T) + self.b



# Load dataset
dataset = pd.read_csv('lenght_of_fish.csv')
x = dataset.iloc[:, 1:3].values
y = dataset.iloc[:, 3:4].values

# Standardize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# Split data into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fit the training data into the model
regressor = LinearRegression()
cost, w, b = regressor.fit(x_train, y_train, alpha = 0.05, epochs=10000)

# Make prediction on the test set
y_pred = regressor.predict(x_test)


# R-Squared
def r2_score(y_true, y_pred):
    mean = np.mean(y_true)
    ss_tot = np.square(y_true - mean).sum()
    ss_res = np.square(y_true - y_pred).sum()
    r2 = 1 - (ss_res / ss_tot)
    return r2

# Adjusted R-Squared
def adjusted_r2(y_true, y_pred, n, k):
    r2 = r2_score(y_true, y_pred)
    a = (1 - r2)
    b = (n - 1) / (n - (k + 1))
    return 1 - (a * b)

# Model Accuracy
print("R2 Score: ", r2_score(y_test, y_pred))
print("Adjusted R2 Score: ", adjusted_r2(y_test, y_pred, len(x_test), x_test.shape[1]))

# Plot Cost History
plt.plot(range(len(cost)), cost)

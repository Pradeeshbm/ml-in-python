# -*- coding: utf-8 -*-
"""
Logistic Regression

The dataset for this model has been downloaded from archive.ics.uci.edu - https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival

@author: Pradeesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Logistic_Regression:

    def __init__(self, alpha = 0.05, epochs = 10000):
        self.alpha = alpha
        self.epochs = epochs

    def __init_params(self, n_weights):
        self.w = np.random.rand(1, n_weights)
        self.b = 0

    def __cost(self, y_true, y_pred):
        return (-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)).mean()

    def fit(self, x, y):
        self.__init_params(x.shape[1])
        costs = np.zeros(self.epochs)
        n = x.shape[0]

        for i in range(self.epochs):
            # Compute Cost
            y_pred = self.__hypothesis(x)
            costs[i] = self.__cost(y, y_pred)
            loss = y_pred - y

            # Update Parameters
            self.w = self.w - (self.alpha / (2 * n)) * np.sum(x * loss, axis=0, keepdims=True)
            self.b = self.b - (self.alpha / (2 * n)) * loss.sum()

        return costs, self.w, self.b

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __hypothesis(self, x):
        z = np.dot(x, self.w.T) + self.b
        h = self.__sigmoid(z)
        return h

    def predict(self, x, threshold = 0.5):
        return self.__hypothesis(x) > threshold


# Load dataset
dataset = pd.read_csv('haberman.csv')
x = dataset.iloc[:, 0:3].values
y = dataset.survival_status.replace([1, 2], [1, 0]).values.reshape(x.shape[0], 1)

# Standardize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# Split the data into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fit the training set into model
classifier = Logistic_Regression()
costs, w, b = classifier.fit(x_train, y_train)

# Make Prediction on Test set
y_pred = classifier.predict(x_test)

# Model Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test, y_pred))

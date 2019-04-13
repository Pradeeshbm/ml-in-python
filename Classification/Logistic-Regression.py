# -*- coding: utf-8 -*-
"""
Logistic Regression

The dataset for this model has been downloaded from

@author: Pradeesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, alpha = 0.05, epochs = 1000):
        self.alpha = alpha
        self.epochs = epochs

    def __init_params(self, n_weights):
        self.w = np.random.randn(n_weights)
        self.b = 0

    def __cost(self, y_true, y_pred):
        return (-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)).mean()

    def fit(self, x, y):
        self.__init_params(x.shape[1])
        costs = np.zeros(self.epochs)

        for i in range(self.epochs):
            y_pred = self.predict(x)
            costs[i] = self.__cost(y, y_pred)

        return costs, self.w, self.b

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        z = np.dot(x, self.w.T) + self.b
        h = self.sigmoid(z)
        return h


# Load dataset
dataset = pd.read_csv('')

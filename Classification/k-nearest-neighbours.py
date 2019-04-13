# -*- coding: utf-8 -*-
"""
K Nearest Neighbours

The dataset for this model has been downloaded from Kaggle - https://www.kaggle.com/selinraja/irish-data

@author: Pradeesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

class KNearestNeighbours:

    def __init__(self, k = 5):
        self.k = 5

    def fit(self, x, y):
        self.ds = np.column_stack((x, y))

    def __euclidean_distance(self, x1, x2):
        np.sqrt(sum([np.square(x1i - x2i) for x1i, x2i in zip(x1, x2)]))

    def predict(self, x):
        distance_arr = np.zeros(shape = (self.ds.shape[0], 2))
        for i, dsi in enumerate(self.ds):
            distance = self.__euclidean_distance(x, dsi[:-1])
            distance_arr[i][0], distance_arr[i][1] = distance, dsi[-1]

        # Sort the euclidean distances using merge sort
        distance_arr = distance_arr[distance_arr[:, 0].argsort(kind = 'mergesort')]
        k_neighbours = distance_arr[self.k, :]
        return self.__get_majority(k_neighbours)

    def __get_majority(self, neighbours):
        return Counter(neighbours[1]).most_common()[0][1]

# Load dataset
dataset = pd.read_csv('data/iris_data.csv')
x = dataset.iloc[:, 0:3].values
y = dataset.species.replace([1, 2], [1, 0]).values.reshape(x.shape[0], 1)

# Standardize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# Split the data into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Fit the training set into model
classifier = KNearestNeighbours()
classifier.fit(x_train, y_train)

# Make Prediction on Test set
y_pred = classifier.predict(x_test[0])

# Model Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
c = KNeighborsClassifier()
c.fit(x_train, y_train)
y_pred_c = c.predict(x_test)

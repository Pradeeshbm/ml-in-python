# -*- coding: utf-8 -*-
"""
K Nearest Neighbours

The dataset for this model has been downloaded from Kaggle - https://www.kaggle.com/selinraja/irish-data

@author: Pradeesh
"""
import numpy as np
import pandas as pd


class GaussianNBClassifier:

    def __init__(self):
        pass

    def __cost(self, y_true, y_pred):
        pass

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass


# Load dataset
dataset = pd.read_csv('data/irish_data.csv')
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
classifier = GaussianNBClassifier()
classifier.fit(x_train, y_train)

# Make Prediction on Test set
y_pred = classifier.predict(x_test)

# Model Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Compare our model accuracy with Scikit Learn library
from sklearn.naive_bayes import  GaussianNB
c = GaussianNB()
c.fit(x_train, y_train)
y_pred_c = c.predict(x_test)
print("Accuracy Using Scikit Learn: ", accuracy_score(y_test, y_pred_c))

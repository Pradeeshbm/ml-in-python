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

    '''
    Method returns mean, variance for each class and for each independent features.
        example: param_map = {
            'class_1': ([c1_mean_x1, c1_mean_x2, c1_mean_x3, ..., c1_mean_xj], [c1_var_x1, c1_var_x2, c1_var_x3, ..., c1_var_xj]),
            'class_2': ([c2_mean_x1, c2_mean_x2, c2_mean_x3, ..., c2_mean_xj], [c2_var_x1, c2_var_x2, c2_var_x3, ..., c2_var_xj]),
            .
            .
            'class_n': ([cn_mean_x1, cn_mean_x2, cn_mean_x3, ..., cn_mean_xj], [cn_var_x1, cn_var_x2, cn_var_x3, ..., cn_var_xj]),
    '''
    def __populate_param_map(self, ds, unique_classes):
        param_map = {}
        for c in unique_classes:
            class_filtered = (ds[ds[-1] == c][:, :-1])
            param_map[c] = (np.mean(class_filtered, axis = 0), np.var(class_filtered.var(), axis = 0))

        return param_map

    def fit(self, x, y):
        unique_classes = np.unique(y)
        ds = np.column_stack((x, y))
        param_map = self.__populate_param_map(ds, unique_classes)

    def predict(self, x):
        


# Load dataset
dataset = pd.read_csv('data/iris_data.csv')
x = dataset.iloc[:, 0:3].values
y_labeled = dataset.iloc[:, 4].values

# Encode target variable
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y_labeled)

# Standardize the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x = sc.fit_transform(x)

# Split the data into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fit the training set into model
classifier = GaussianNBClassifier()
ds = classifier.fit(x_train, y_train)

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

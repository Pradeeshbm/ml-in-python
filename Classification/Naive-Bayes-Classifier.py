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
            # Filter dataset by class, remove the class column from filtered data
            class_filtered = ds[ds[:, -1] == c][:, :-1]
            param_map[c] = (np.mean(class_filtered, axis=0), np.var(class_filtered, axis=0))

        return param_map

    # Method concatenates x and y in column axis in order to compute parameter easily.
    def fit(self, x, y):
        self.unique_classes = np.unique(y)
        ds = np.column_stack((x, y))
        self.param_map = self.__populate_param_map(ds, self.unique_classes)

    # Method calculates gaussian probability distribution on given x with mean and variance of calculated parameter in fit method
    def predict(self, x):
        class_probability_dict = {}
        for c in self.unique_classes:
            class_probability_dict[c] = np.prod(self.__gaussian_distribution(self.param_map[c][0], self.param_map[c][1], x), axis=1)


        return pd.DataFrame(class_probability_dict)

    # Function to calculate probability density function
    def __gaussian_distribution(self, mean, var, x):
        a = 1 / np.sqrt(2 * np.pi * var)
        numerator = -np.square(x - mean)
        denominator = 2 * var
        b = np.exp(numerator / denominator)

        return a * b

def g(mean, var, x):
    a = 1 / np.sqrt(2 * np.pi * var)
    numerator = -np.square(x - mean)
    denominator = 2 * var
    b = np.exp(numerator / denominator)

    return a * b

def f(m, v, a):
    return m + v + a

# Load dataset
dataset = pd.read_csv('data/iris_data.csv')
x = dataset.iloc[:, 0:4].values
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

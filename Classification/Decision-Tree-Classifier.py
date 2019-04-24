# -*- coding: utf-8 -*-
"""
K Nearest Neighbours

The dataset for this model has been downloaded from Kaggle - https://www.kaggle.com/selinraja/irish-data

@author: Pradeesh
"""
import numpy as np
import pandas as pd


class Leaf:

    def __init__(self, rows):
        self.predictions = class_count(rows)


class Criteria:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        return value == self.value

    def __repr__(self):
        condition = '=='
        if is_numeric(self.value):
            condition = '>='
        return ('is %s %s %s?' % header[self.value], condition, str(self.value))


def partition(dataset, criteria):
    true_rows, false_rows = [], []
    for example in dataset:
        if criteria.match(example):
            true_rows.append(example)
        else:
            false_rows.append(example)

    return true_rows, false_rows


class DecisionNode:

    def __init__(self, criteria, true_rows, false_rows):
        self.criteria = criteria
        self.true_rows = true_rows
        self.false_rows = false_rows


def class_count(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] = counts[label] + 1
    return counts


def gini(dataset):
    class_count = class_count(dataset)
    prob_of_label = 0
    for lbl in class_count:
        prob_of_label = prob_of_label + (class_count[lbl] / float(len(dataset))) ** 2

    return prob_of_label

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

def find_best_criteria(dataset):
    col_count = len(dataset[0]) - 1
    best_criteria = None
    best_score = 0

    for col in range(col_count):
        distinct_value = set([ex[col] for ex in dataset])
        for val in distinct_value:
            criteria = Criteria(col, val)
            true_rows, false_rows = partition(dataset, criteria)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            partition_1_score = gini(true_rows)
            partition_2_score = gini(false_rows)
            weighted_score = len(true_rows) / len(dataset) * partition_1_score + len(false_rows) /  len(dataset) * partition_2_score
            if weighted_score > best_score:
                best_criteria = criteria
                best_score = weighted_score

    return best_score, best_criteria

def build_tree(dataset):
    best_score, best_criteria = find_best_criteria(dataset)
    if best_score == 0:
        return Leaf(dataset)

    true_rows, false_rows = partition(dataset, best_criteria)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return DecisionNode(best_criteria, true_branch, false_branch)


class DecisionTreeClassifier:

    def __init__(self):
        pass

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def __gaussian_distribution(self, mean, var, x):
        pass


# Load dataset
dataset = pd.read_csv('data/iris_data.csv')
x = dataset.iloc[:, 0:4].values
y_labeled = dataset.iloc[:, 4].values

# Encode target variable
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y_labeled)

# Standardize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# Split the data into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fit the training set into model
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Make Prediction on Test set
y_pred = classifier.predict(x_test)

# Model Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Compare our model accuracy with Scikit Learn library

c.fit(x_train, y_train)
y_pred_c = c.predict(x_test)
print("Accuracy Using Scikit Learn: ", accuracy_score(y_test, y_pred_c))

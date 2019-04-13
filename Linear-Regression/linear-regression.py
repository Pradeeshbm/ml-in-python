# -*- coding: utf-8 -*-
"""
Linear Regression

The dataset for this model has been downloaded from Kaggle - https://www.kaggle.com/harlfoxem/housesalesprediction

@author: Pradeesh
"""

import numpy as np
import pandas as pd

# Load dataset
dataset = pd.read_csv('kc_house_data.csv')
x = dataset[['sqft_living15', 'sqft_basement', 'sqft_above', 'grade', 'view', 'floors', 'sqft_living', 'bathrooms', 'bedrooms']].values
y = dataset.iloc[:, 2:3].values

# Standardize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# Split data into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fit the training data into model

# Evaluate the model

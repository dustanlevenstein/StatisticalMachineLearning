#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:46:30 2020

@author: dustan
"""

import numpy as np
class BasicPCA(object):
    def __init__(self, n_components=None, data=None):
        self.data = (np.array(data) if data is not None else None)
        self.numcomponents = n_components
    def fit(self, X=None):
        if X is None:
            X = self.data
        else:
            self.data = X
        
        # standardize columns.
        self.mean = X.mean(axis=0)
        self.data_cent = X-self.mean
        # optional
        # self.data_cent = self.data_cent/self.data_cent.std(axis=0)

        eigenvalues, eigenvectors = np.linalg.eigh(
            self.data_cent.T@self.data_cent)
        eigenvectors = np.flip(eigenvectors, axis=1)
        eigenvalues = np.flip(eigenvalues)
        
        # An adjustment solely for the purpose of making this match the results
        # of section_05_01_03.py.
        eigenvectors[:,0] = -eigenvectors[:,0]
        
        self.explained_variance_ratio_ = list(eigenvalues/eigenvalues.sum())
        self.pc_directions = eigenvectors
    def transform(self, X=None, num_components=None):
        if num_components is None:
            num_components = self.numcomponents
            if num_components is None:
                num_components = 2
        if X is None:
            X = self.data
        return (X-self.mean) @ self.pc_directions[:num_components]

import pandas as pd
data = pd.read_csv("iris.csv")
print(data.describe())

# =============================================================================
# 
#        sepal_length  sepal_width  petal_length  petal_width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.057333      3.758000     1.199333
# std        0.828066     0.435866      1.765298     0.762238
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.800000     3.000000      4.350000     1.300000
# 75%        6.400000     3.300000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000
# 
# Perhaps it doesn't need to be standardized, but it does need to be centered.
# =============================================================================

X = data[['sepal_length', 'sepal_width', 'petal_length',
          'petal_width']].to_numpy()
correlations = np.corrcoef(X.T)
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap((1-correlations)/2, xticklabels=['sepal_length', 'sepal_width',
                                         'petal_length', 'petal_width'],
            yticklabels=['sepal_length', 'sepal_width',
                         'petal_length', 'petal_width'])
plt.show()

# =============================================================================
# Observation: sepal_length, petal_length, and petal_width are strongly
# correlated together, while sepal_width is somewhat negatively correlated with
# the others.
# =============================================================================

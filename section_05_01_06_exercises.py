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
        
        self.explained_variance_ratio_ = list(eigenvalues/eigenvalues.sum())
        self.pc_directions = eigenvectors
    def transform(self, X=None, num_components=None):
        if num_components is None:
            num_components = self.numcomponents
            if num_components is None:
                num_components = 2
        if X is None:
            X = self.data
        return (X-self.mean) @ self.pc_directions[:,:num_components]

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
pca = BasicPCA()
pca.fit(X)
ev = pca.explained_variance_ratio_
print(pca.explained_variance_ratio_)
print([ev[0], ev[0]+ev[1], ev[0]+ev[1]+ev[2], ev[0]+ev[1]+ev[2]+ev[3]])

# To capture 95% of the explained variance, use two principal components.

K = 2

print("First {K} princpal components:".format(K=K))
print(pca.pc_directions[:,:K])
PC = pca.transform(num_components=K)
datum = np.concatenate([X, PC], axis=1)

correlations = np.corrcoef(datum.T)
sns.heatmap((1-correlations)/2,
            xticklabels=['sepal_length', 'sepal_width', 'petal_length',
                         'petal_width', 'PC1', 'PC2'],
            yticklabels=['sepal_length', 'sepal_width', 'petal_length',
                         'petal_width', 'PC1', 'PC2'])
plt.show()

# =============================================================================
# It looks like PC1 is weakly correlated with sepal_width, and strongly
# anticorrelated with the other attributes, while PC2 is strongly correlated
# with sepal_width, and not correlated with the other attributes.
# =============================================================================

colors = list(map(
    lambda x: {'setosa':'r', 'versicolor':'g', 'virginica':'b'}[x],
    data['species']))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.xlabel("x1"); plt.ylabel("x2")

plt.subplot(122)
plt.scatter(PC[:, 0], PC[:, 1], c=colors)
plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
plt.axis('equal')
plt.tight_layout()
plt.show()

from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3)

yhat = model.fit_predict(X)
ac_colors = list(map(lambda x: ['g', 'r', 'b'][x], yhat))

plt.subplot(121)
plt.scatter(PC[:, 0], PC[:, 1], c=colors)
plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
plt.axis('equal')
plt.tight_layout()
plt.title("True classifications")

plt.subplot(122)
plt.scatter(PC[:, 0], PC[:, 1], c=ac_colors)
plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
plt.axis('equal')
plt.tight_layout()
plt.title("AC classifications")

plt.show()


PC_all = pca.transform(num_components=4)
yhat = model.fit_predict(PC_all)
ac_colors = list(map(lambda x: ['g', 'r', 'b'][x], yhat))

plt.subplot(121)
plt.scatter(PC[:, 0], PC[:, 1], c=colors)
plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
plt.axis('equal')
plt.tight_layout()
plt.title("True classifications")

plt.subplot(122)
plt.scatter(PC[:, 0], PC[:, 1], c=ac_colors)
plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
plt.axis('equal')
plt.tight_layout()
plt.title("PCA AC classifications")

plt.show()

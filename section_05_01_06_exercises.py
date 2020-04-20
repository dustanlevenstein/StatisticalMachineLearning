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
    def transform(self, num_components=None):
        if num_components is None:
            num_components = self.numcomponents
            if num_components is None:
                num_components = 2
        else:
            self.data = X
            self.fit()
        
        # TODO

import matplotlib.pyplot as plt

np.random.seed(42)
 
# dataset
n_samples = 100
experience = np.random.normal(size=n_samples)
salary = 1500 + experience + np.random.normal(size=n_samples, scale=.5)
X = np.column_stack([experience, salary])

pca = BasicPCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)

# PC = pca.transform(X)

# plt.subplot(121)
# plt.scatter(X[:, 0], X[:, 1])
# plt.xlabel("x1"); plt.ylabel("x2")

# plt.subplot(122)
# plt.scatter(PC[:, 0], PC[:, 1])
# plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
# plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
# plt.axis('equal')
# plt.tight_layout()
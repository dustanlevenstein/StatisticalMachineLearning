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
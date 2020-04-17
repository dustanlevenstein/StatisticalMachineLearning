#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:46:30 2020

@author: dustan
"""

import numpy as np
class BasicPCA(object):
    def __init__(self, data=None):
        self.data = (np.array(data) if data is not None else None)
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
        
        self.explained_variances = list(eigenvalues/eigenvalues.sum())
        self.pc_directions = eigenvectors
    def transform(self, X=None, num_components=2):
        if X is None:
            X = self.data
        else:
            self.data = X
            self.fit()
        
        # TODO
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:42:24 2020

@author: dustan
"""

import numpy as np
X = np.random.randn(4, 2)
mins = X.argmin(axis=0)
print(X)
print(mins)
def standardize(X):
    return (X-X.mean(axis=0))/X.std(axis=0)
print(standardize(X))
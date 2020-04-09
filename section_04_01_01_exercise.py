#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:43:15 2020

@author: dustan
"""

# • Generate 2 random samples: x ∼ N (1.78, 0.1) and y ∼ N (1.66, 0.1), both
#   of size 10.

import numpy as np

size = 10
x = np.random.normal(1.78, .1, size=size)
y = np.random.normal(1.66, .1, size=size)

xbar = np.sum(x)/size
xstd = np.sqrt(np.sum((x-xbar)*(x-xbar))/(size-1))
ybar = np.sum(y)/size
xycov = np.sum((x-xbar)*(y-ybar))/(size-1)

xbar_auto = np.mean(x)
xstd_auto = np.std(x, ddof=1)
xycov_matrix = np.cov([x,y], ddof=1)
xycov_auto = xycov_matrix[0][1]

epsilon = 10**(-10)
assert abs(xbar - xbar_auto) < epsilon
assert abs(xstd - xstd_auto) < epsilon
assert abs(xycov - xycov_auto) < epsilon

w, v = np.linalg.eigh(xycov_matrix)
print(w) # eigenvalues in ascending order
print(v) # columns are eigenvectors
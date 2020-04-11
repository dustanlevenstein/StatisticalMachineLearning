#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:51:09 2020

@author: dustan
"""

import numpy as np

a = np.array([[2], [1]])
b = np.array([[1], [1]])

def euclidean(x):
    return np.sqrt(np.sum(x*x))

print(euclidean(a)) # should be sqrt(5)
print(euclidean(a-b)) # should be 1


print(((a.T @ b)/euclidean(a))[0,0])

import matplotlib.pyplot as plt
import seaborn as sns
colors = sns.color_palette()

plt.arrow(0, 0, a[0,0], a[1, 0], head_width=.1, color=colors[0], label="a")
plt.arrow(0, 0, b[0,0], b[1, 0], head_width=.1, color=colors[1], label="b")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend()
plt.show()

np.random.seed(42)

X = np.random.random((100, 2, 1))*4-2

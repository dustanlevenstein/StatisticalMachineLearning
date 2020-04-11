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
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:22:42 2020

@author: dustan
"""

def numpy_and_scipy_1_1_4():
    import numpy as np
    import scipy
    import scipy.linalg
    X = np.array([[1, 2], [3, 4]])
    v = np.array([1, 2])
    print(np.dot(X, v)) # no broadcasting
    print(X*v) # broadcasting
    print(np.dot(v, X))
    print(X-X.mean(axis=0))
    print(scipy.linalg.svd(X, full_matrices=False))

def matplotlib_1_1_4():
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(0, 10, 50)
    sinus = np.sin(x)
    plt.plot(x, sinus)
    plt.show()
    
matplotlib_1_1_4()
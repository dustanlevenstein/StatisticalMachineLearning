#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:51:09 2020

@author: dustan
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dot_product_and_euclidean_norm():
    a = np.array([[2], [1]])
    b = np.array([[1], [1]])
    
    def euclidean(x, axis=0):
        return np.sqrt(np.sum(x*x, axis=axis))
    
    print(euclidean(a)) # should be sqrt(5)
    print(euclidean(a-b)) # should be 1
    
    
    print(((a.T @ b)/euclidean(a))[0,0])
    
    colors = sns.color_palette()
    
    ax = plt.gca()
    arrow1 = plt.arrow(0, 0, a[0,0], a[1, 0], head_width=.1, color=colors[0],
                       label="a")
    arrow2 = plt.arrow(0, 0, b[0,0], b[1, 0], head_width=.1, color=colors[1],
                       label="b")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    ax.set_aspect('equal', 'box')
    plt.legend([arrow1, arrow2], ['a', 'b'])
    plt.show()
    
    np.random.seed(42)
    
    numpoints = 100000
    X = np.random.random((numpoints, 2, 1))*4-2
    x_coords = X[:,0,0]
    y_coords = X[:,1,0]
    
    ax = plt.gca()
    arrow1 = plt.arrow(0, 0, a[0,0], a[1, 0], head_width=.1, color=colors[0],
                       label="a")
    arrow2 = plt.arrow(0, 0, b[0,0], b[1, 0], head_width=.1, color=colors[1],
                       label="b")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    ax.set_aspect('equal', 'box')
    plt.legend([arrow1, arrow2], ['a', 'b'])
    plt.scatter(x_coords, y_coords, color=colors[2], marker=".")
    plt.show()
    
    X_projections_on_a = ((a.T@X).reshape(numpoints)/euclidean(a))
    print("Projections onto a:", X_projections_on_a)
    
    sns.distplot(X_projections_on_a, bins=21)
    plt.show()
    
def covariance_matrix_and_m_norm():
    Mu = np.array([1, 1])
    Sigma = np.array([[1, .8], [.8, 1]])
    numpoints = 100
    np.random.seed(42)
    X = np.random.multivariate_normal(Mu, Sigma, size=numpoints)
    Xbar = X.mean(axis=0)
    print("Xbar:", Xbar, "compared to true mean of", Mu)
    Xcov = ((X-Xbar.T).T@(X-Xbar.T))/(numpoints-1)
    print("Xcov:\n", Xcov, "\ncompared to true cov\n", Sigma)
    
covariance_matrix_and_m_norm()
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:43:50 2020

@author: dustan
"""

import warnings
warnings.filterwarnings("ignore")
##
import numpy as np
import matplotlib.pyplot as plt
import plot_utils
import seaborn as sns  # nice color

np.random.seed(42)
colors = sns.color_palette()

n_samples, n_features = 100000, 2

mean, Cov, X = [None] * 4, [None] * 4, [None] * 4
mean[0] = np.array([-3.5, 3.5])
Cov[0] = np.array([[1, 0],
                   [0, 1]])

mean[1] = np.array([3.5, 3.5])
Cov[1] = np.array([[1, .5],
                   [.5, 1]])

mean[2] = np.array([-3.5, -3.5])
Cov[2] = np.array([[1, .9],
                   [.9, 1]])

mean[3] = np.array([3.5, -3.5])
Cov[3] = np.array([[1, -.9],
                   [-.9, 1]])

# Generate dataset
for i in range(len(mean)):
    X[i] = np.random.multivariate_normal(mean[i], Cov[i], n_samples)

# Plot
for i in range(len(mean)):
    # Points
    plt.scatter(X[i][:, 0], X[i][:, 1], color=colors[i], label="class %i" % i)
    # Means
    #plt.scatter(mean[i][0], mean[i][1], marker="o", s=200, facecolors='w',
    #            edgecolors=colors[i], linewidth=2)
    # Ellipses representing the covariance matrices
    plot_utils.plot_cov_ellipse(Cov[i], pos=mean[i], facecolor='none',
                                          linewidth=1, edgecolor='w')

plt.axis('equal')
_ = plt.legend(loc='upper left')
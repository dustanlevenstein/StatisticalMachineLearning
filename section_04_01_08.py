#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:30:53 2020

@author: dustan
"""

import numpy as np
np.random.seed(seed=42) # make example reproducible
# Dataset
n_samples, n_features = 100, 1000
n_info = n_features//10 # number of features with information
n1, n2 = n_samples//2, n_samples - n_samples//2
snr = .5
Y = np.random.randn(n_samples, n_features)
grp = np.array(["g1"] * n1 + ["g2"] * n2)
# Add some group effect for Pinfo features
Y[grp=="g1", :n_info] += snr
#
import scipy.stats as stats
import matplotlib.pyplot as plt
tvals, pvals = np.full(n_features, np.NAN), np.full(n_features, np.NAN)

# Okay, I see we're testing whether g1 and g2 have the same mean, depending on
# which column of the randn box we're looking at.
for j in range(n_features):
    tvals[j], pvals[j] = stats.ttest_ind(Y[grp=="g1", j], Y[grp=="g2", j],
                                         equal_var=True)
fig, axis = plt.subplots(3, 1)#, sharex='col')
axis[0].plot(range(n_features), tvals, 'o')
axis[0].set_ylabel("t-value")
axis[1].plot(range(n_features), pvals, 'o')
axis[1].axhline(y=0.05, color='red', linewidth=3, label="p-value=0.05")
#axis[1].axhline(y=0.05, label="toto", color='red')
axis[1].set_ylabel("p-value")
axis[1].legend()
axis[2].hist([pvals[n_info:], pvals[:n_info]],
             stacked=True, bins=100, label=["Negatives", "Positives"])
axis[2].set_xlabel("p-value histogram")
axis[2].set_ylabel("density")
axis[2].legend()
plt.tight_layout()
plt.show()


# My attempt to picture what's going on here.


ax = plt.gca()
ax.set_aspect("equal", "box")
ax.add_patch(plt.Rectangle((0, 0), 100, 100, facecolor = "red", edgecolor = "red"))
ax.add_patch(plt.Rectangle((100, 0), 900, 100, facecolor = "green", edgecolor = "green"))
plt.plot((100, 100), (0, 100), marker=None, color='black')
plt.plot((0, 1000), (50, 50), marker=None, color='black')
plt.text(10, 65, "+snr")
plt.show()
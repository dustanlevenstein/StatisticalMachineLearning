#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:15:55 2020

@author: dustan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://python-graph-gallery.com/wp-content/uploads/mtcars.csv'
df = pd.read_csv(url)

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Using the mask clears out the upper right triangle.

f, ax = plt.subplots(figsize=(5.5, 4.5))
cmap = sns.color_palette("RdBu_r", 11)
# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(corr, mask=None, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# Observation: the correlation between cyl and disp and the correlation
# between cyl and hp is 1, but the correlation between disp and hp is not one.

corr_truncated = corr.loc[['cyl', 'disp', 'hp'], ['cyl', 'disp', 'hp']]

f, ax = plt.subplots(figsize=(5.5, 4.5))
cmap = sns.color_palette("RdBu_r", 11)
# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(corr_truncated, mask=None, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
print(corr_truncated)


# convert correlation to distances
d = 2 * (1 - np.abs(corr))

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=3, linkage='single', affinity="precomputed").fit(d)
lab=0

clusters = [list(corr.columns[clustering.labels_==lab]) for lab in set(clustering.labels_)]
print(clusters)

reordered = np.concatenate(clusters)

R = corr.loc[reordered, reordered]

f, ax = plt.subplots(figsize=(5.5, 4.5))
# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(R, mask=None, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()




# Question: do the distances form a valid metric space?
# Testing triangle inequality
def dist(a, b):
    return d.loc[a,b]
for x in R.index:
    for y in R.index:
        for z in R.index:
            if dist(x, z) > dist(x, y) + dist(y, z):
                print(("Triangle inequality failure:"
                       " d({}, {}) > d({}, {}) + d({}, {})").format(
                           x, z, x, y, y, z))
                           
# Answer: there are a plethora of triangle inequality failures.
# For example,
print(d.loc[['mpg', 'cyl', 'qsec'], ['mpg', 'cyl', 'qsec']])
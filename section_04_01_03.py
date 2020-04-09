#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:23:51 2020

@author: dustan
"""

import matplotlib.pyplot as plt

plt.bar([0, 1, 2, 3], [1/8, 3/8, 3/8, 1/8], width=0.9)
_ = plt.xticks([0, 1, 2, 3], [0, 1, 2, 3])
plt.xlabel("Distribution of the number of head over 3 flip under the null hypothesis")
#Text(0.5, 0, 'Distribution of the number of head over 3 flip under the null hypothesis')
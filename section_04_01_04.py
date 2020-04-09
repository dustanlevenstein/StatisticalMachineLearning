#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:59:34 2020

@author: dustan
"""

import numpy as np
import scipy.stats as stats
n = 10
x = np.random.normal(size=n)
y = 2 * x + np.random.normal(size=n)
# Compute with scipy
cor, pval = stats.pearsonr(x, y)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:43:15 2020

@author: dustan
"""

# • Generate 2 random samples: x ∼ N (1.78, 0.1) and y ∼ N (1.66, 0.1), both
#   of size 10.

import numpy as np
x = np.random.normal(size=10) * .1 + 1.78
y = np.random.normal(size=10) * .1 + 1.66
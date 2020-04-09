#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:04:55 2020

@author: dustan
"""

import pandas as pd
# import matplotlib.pyplot as plt


url = 'https://raw.github.com/neurospin/pystatsml/master/datasets/salary_table.csv'
salary = pd.read_csv(url)


import seaborn as sns
sns.boxplot(x="education", y="salary", hue="management", data=salary)
sns.boxplot(x="management", y="salary", hue="education", data=salary)
sns.stripplot(x="management", y="salary", hue="education", data=salary,
              jitter=True, dodge=True, linewidth=1)

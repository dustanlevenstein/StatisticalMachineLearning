#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:52:29 2020

@author: dustan
"""

import pandas as pd
# df = pd.read_csv("https://github.com/neurospin/pystatsml/tree/master/datasets/iris.csv")
df = pd.read_csv("https://raw.githubusercontent.com/neurospin/pystatsml/master/datasets/iris.csv")
print(df.columns)
numerical = df.select_dtypes(include="number")
stats = df.groupby('species').mean()
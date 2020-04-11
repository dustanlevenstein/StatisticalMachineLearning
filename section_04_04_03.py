#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:33:54 2020

@author: dustan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("multiTimeline.csv", skiprows=2)

print(df.head())

# Rename columns
df.columns = ['month', 'diet', 'gym', 'finance']

# Describe
print(df.describe())

df.month = pd.to_datetime(df.month)
df.set_index('month', inplace=True)

print(df.head())

df.plot()
plt.xlabel('Year');
plt.show()

# change figure parameters
df.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.show()
# Plot single column
df[['diet']].plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

plt.show()


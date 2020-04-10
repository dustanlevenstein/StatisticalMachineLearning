#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:37:29 2020

@author: dustan
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def simple_linear_regression_and_correlation():
    df = pd.read_csv("birthwt.csv")
    print(df.head())
# =============================================================================
#     simple_linear_regression_and_correlation()
#        Unnamed: 0  low  age  lwt  race  smoke  ptl  ht  ui  ftv   bwt
#     0          85    0   19  182     2      0    0   0   1    0  2523
#     1          86    0   33  155     3      0    0   0   0    3  2551
#     2          87    0   20  105     1      1    0   0   0    1  2557
#     3          88    0   21  108     1      1    0   0   1    2  2594
#     4          89    0   18  107     1      1    0   0   1    0  2600
# =============================================================================
    
# =============================================================================
#     low
#     
#         indicator of birth weight less than 2.5 kg.
#     age
#     
#         mother's age in years.
#     lwt
#     
#         mother's weight in pounds at last menstrual period.
#     race
#     
#         mother's race (1 = white, 2 = black, 3 = other).
#     smoke
#     
#         smoking status during pregnancy.
#     ptl
#     
#         number of previous premature labours.
#     ht
#     
#         history of hypertension.
#     ui
#     
#         presence of uterine irritability.
#     ftv
#     
#         number of physician visits during the first trimester.
#     bwt
#     
#         birth weight in grams.
#     
# 
# =============================================================================
    # 1. Test the association of mother’s age and birth weight using the
    # correlation test and linear regeression.
    x = df['lwt'].to_numpy()
    y = df['bwt'].to_numpy()
    # We start with Spearman correlation.
    plt.plot(x, y, "bo")
    plt.title("Birth weight and mother's weight")
    plt.xlabel("mother's weight")
    plt.ylabel("baby's weight")
    cor, pval = stats.spearmanr(x, y)
    print("Spearman cor test, cor: %.4f, pval: %.4f" % (cor, pval))
    cor, pval = stats.pearsonr(x, y) # Pearson test yields the linear
                                     # regression correlation coefficient.
    print("Pearson cor test, cor: %.4f, pval: %.4f" % (cor, pval))
    beta, beta0, r_value, p_value, std_err = stats.linregress(x, y)
    print("linear regression p-value is %.4f" % p_value)    
    
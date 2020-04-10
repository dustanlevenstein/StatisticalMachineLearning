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
    print("Mother's weight and birth weight:")
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
    plt.show()
    

    print("Mother's age and birth weight:")
    x = df['age'].to_numpy()
    y = df['bwt'].to_numpy()
    # We start with Spearman correlation.
    plt.plot(x, y, "bo")
    plt.title("Birth weight and mother's age")
    plt.xlabel("mother's age")
    plt.ylabel("baby's weight")
    cor, pval = stats.spearmanr(x, y)
    print("Spearman cor test, cor: %.4f, pval: %.4f" % (cor, pval))
    cor, pval = stats.pearsonr(x, y) # Pearson test yields the linear
                                     # regression correlation coefficient.
    print("Pearson cor test, cor: %.4f, pval: %.4f" % (cor, pval))
    beta, beta0, r_value, p_value, std_err = stats.linregress(x, y)
    print("linear regression p-value is %.4f" % p_value)
    plt.show()
    # Conclusion: Although there is a small but highly statistically
    # significant correlation between the mother's weight and the baby's
    # weight, the association between them is not linear. Moreover, looking at
    # a scatterplot of data can be misleading, to say the least, since there is
    # no apparent correlation in the scatterplot comparing weights.
    
    # As far as mother's age goes, there seems to be no correlation.
    
def simple_linear_regression_maths():
    df = pd.read_csv("https://raw.github.com/neurospin/pystatsml/master/datasets/salary_table.csv")
    x = df['experience'].to_numpy()
    y = df['salary'].to_numpy()
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
    ybar = np.sum(y)/len(y)
    SS_tot = np.sum((y-ybar)*(y-ybar))
    yhat = intercept + slope*x
    SS_reg = np.sum((yhat-ybar)*(yhat-ybar))
    SS_res = np.sum((yhat-y)*(yhat-y))
    assert np.allclose(SS_reg+SS_res, SS_tot, atol=1e-05)
    Rsquared = SS_reg/SS_tot
    assert np.allclose(Rsquared, rvalue*rvalue, 1e-05)
    n = len(y)
    F = SS_reg/(SS_res/(n-2))
    print("F:", F)
    fvalues = np.linspace(10, 25, 100)
    plt.plot(fvalues, stats.f.pdf(fvalues, 1, n-2))
    upper_fvals = fvalues[fvalues >= F]
    plt.fill_between(upper_fvals, 0, stats.f.pdf(upper_fvals, 1, n-2))
    plt.xlabel("F values")
    plt.ylabel("Density")
    plt.title("F(1, {})".format(n-2))
    plt.show()
    p_val = 1-stats.f.cdf(F, 1, n-2)
    print("F-test p-value:", p_val)
    print("Linear regression p-value:", pvalue)
    print("Difference:", p_val-pvalue)
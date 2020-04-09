#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:41:01 2020

@author: dustan
"""

# Normal distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

mu = 0 # mean
variance = 2 #variance
sigma = np.sqrt(variance) #standard deviation\n",
x = np.linspace(mu-3*variance,mu+3*variance, 100)
plt.plot(x, norm.pdf(x, mu, sigma))

sample_normal = np.random.normal(0, sigma, 100)
sns.distplot(sample_normal)
plt.show()



# Chi square distribution

cvalues = np.linspace(.1, 5, 100)
from scipy.stats import chi2
plt.plot(cvalues, chi2.pdf(cvalues, 1), 'b-', label="Chi2(1)")
plt.plot(cvalues, chi2.pdf(cvalues, 5), 'r-', label="Chi2(5)")
plt.plot(cvalues, chi2.pdf(cvalues, 30), 'g-', label="Chi2(30)")
plt.legend()

#sample_chi2 = np.random.chisquare(1, 10)
#sns.distplot(sample_chi2)
plt.show()

from scipy.stats import f
fvalues = np.linspace(.1, 5, 100)
# pdf(x, df1, df2): Probability density function at x of F.
plt.plot(fvalues, f.pdf(fvalues, 1, 30), 'b-', label="F(1, 30)")
plt.plot(fvalues, f.pdf(fvalues, 5, 30), 'r-', label="F(5, 30)")
plt.legend()
# cdf(x, df1, df2): Cumulative distribution function of F.
# ie.
proba_at_f_inf_3 = f.cdf(3, 1, 30) # P(F(1,30) < 3)
# ppf(q, df1, df2): Percent point function (inverse of cdf) at q of F.
f_at_proba_inf_95 = f.ppf(.95, 1, 30) # q such P(F(1,30) < .95)
assert f.cdf(f_at_proba_inf_95, 1, 30) == .95
# sf(x, df1, df2): Survival function (1 - cdf) at x of F.
proba_at_f_sup_3 = f.sf(3, 1, 30) # P(F(1,30) > 3)
assert proba_at_f_inf_3 + proba_at_f_sup_3 == 1
# p-value: P(F(1, 30)) < 0.05
low_proba_fvalues = fvalues[fvalues > f_at_proba_inf_95]
plt.fill_between(low_proba_fvalues, 0, f.pdf(low_proba_fvalues, 1, 30),
                 alpha=.8, label="P < 0.05")
plt.show()
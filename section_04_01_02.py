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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:23:51 2020

@author: dustan
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns


plt.bar([0, 1, 2, 3], [1/8, 3/8, 3/8, 1/8], width=0.9)
_ = plt.xticks([0, 1, 2, 3], [0, 1, 2, 3])
plt.xlabel(
    "Distribution of the number of head over 3 flip under the null hypothesis")
plt.show()

# I notice that this snippet of code didn't come with any commentary on why
# the linspace has to be carefully calibrated to match the domain of the 
# probability mass function.
succes = np.linspace(30, 70, 41)
plt.plot(succes, scipy.stats.binom.pmf(succes, 100, 0.5), 'b-',
         label="Binomial(100, 0.5)")
upper_succes_tvalues = succes[succes >= 60]
plt.fill_between(upper_succes_tvalues, 0,
                 scipy.stats.binom.pmf(upper_succes_tvalues, 100, 0.5),
                 alpha=.8, label="p-value")
_ = plt.legend()
pval = 1 - scipy.stats.binom.cdf(59.5, 100, 0.5)
print(pval)
plt.show()

x = np.linspace(0, 100, 101)
plt.plot(x, 12*scipy.stats.binom.pmf(x, 100, 0.5), 'b-',
         label="Binomial(100, 0.5) approx density")
plt.plot(x, scipy.stats.binom.cdf(x, 100, 0.5), 'r-',
         label="Binomial(100, 0.5) CDF")
plt.legend()
plt.show()


sccess_h0 = scipy.stats.binom.rvs(100, 0.5, size=10000, random_state=4)
#sccess_h0 = np.array([) for i in range(5000)])
_ = sns.distplot(sccess_h0, hist=False)
pval_rnd = np.sum(sccess_h0 >= 60) / (len(sccess_h0) + 1)
print("P-value using monte-carlo sampling of the Binomial distribution under"
      " H0=",
      pval_rnd)
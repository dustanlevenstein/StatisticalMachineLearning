#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:32:39 2020

@author: dustan
"""

import pandas as pd
import matplotlib.pyplot as plt
url = 'https://raw.github.com/neurospin/pystatsml/master/datasets/salary_table.csv'
salary = pd.read_csv(url)

from scipy import stats
import numpy as np
y, x = salary.salary, salary.experience
beta, beta0, r_value, p_value, std_err = stats.linregress(x,y)
print("y = %f x + %f, r: %f, r-squared: %f,\np-value: %f, std_err: %f"
      % (beta, beta0, r_value, r_value**2, p_value, std_err))
print("Regression line with the scatterplot")
yhat = beta * x + beta0 # regression line

plt.plot(x, yhat, 'r-', x, y,'o')
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.show()
print("Using seaborn")
import seaborn as sns
sns.regplot(x="experience", y="salary", data=salary)
plt.show()

def SS(x, y):
    ybar = np.mean(y)
    SS_tot = np.sum((y-ybar)*(y-ybar))
    beta, beta0, r_value, p_value, std_err = stats.linregress(x,y)
    yhat = beta * x + beta0
    SS_reg = np.sum((yhat-ybar)*(yhat-ybar))
    SS_res = np.sum((y-yhat)*(y-yhat))
    return SS_tot, SS_reg, SS_res

SS_tot, SS_reg, SS_res = SS(x, y)
epsilon = 10**(-6)
assert (SS_tot -( SS_reg + SS_res)) < epsilon

x = np.random.random(100)
y = x+np.random.randn()
SS_tot, SS_reg, SS_res = SS(x, y)
epsilon = 10**(-6)
assert (SS_tot -( SS_reg + SS_res)) < epsilon



from scipy import linalg
np.random.seed(seed=42) # make the example reproducible

# Dataset
N, P = 50, 4
X = np.random.normal(size= N * P).reshape((N, P))
## Our model needs an intercept so we add a column of 1s:
X[:, 0] = 1 # THIS DELETES A COLUMN!!!
print(X[:5, :])
betastar = np.array([10, 1., .5, 0.1])
e = np.random.normal(size=N)
y = np.dot(X, betastar) + e
# Estimate the parameters
Xpinv = linalg.pinv2(X)
betahat = np.dot(Xpinv, y)
print("Estimated beta:\n", betahat)
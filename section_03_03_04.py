#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:04:55 2020

@author: dustan
"""

import pandas as pd
import matplotlib.pyplot as plt


url = 'https://raw.github.com/neurospin/pystatsml/master/datasets/salary_table.csv'
salary = pd.read_csv(url)


import seaborn as sns
sns.boxplot(x="education", y="salary", hue="management", data=salary)
sns.boxplot(x="management", y="salary", hue="education", data=salary)
sns.stripplot(x="management", y="salary", hue="education", data=salary,
              jitter=True, dodge=True, linewidth=1)

### Density plot with one figure containing multiple axis
# One figure can contain several axis, whose contain the graphic elements
# Set up the matplotlib figure: 3 x 1 axis
f, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
i = 0
for edu, d in salary.groupby(['education']):
    sns.distplot(d.salary[d.management == "Y"], color="b", bins=10,
                 label="Manager", ax=axes[i])
    sns.distplot(d.salary[d.management == "N"], color="r", bins=10,
                 label="Employee", ax=axes[i])
    axes[i].set_title(edu)
    axes[i].set_ylabel('Density')
    i += 1
ax = plt.legend()
plt.show()


# Violin plot without breakdown by management.
salary["Human"] = "Y"
ax = sns.violinplot(x = 'Human', y="salary", hue="education", data=salary)

plt.show()

# Tips dataset.
tips = sns.load_dataset("tips")
print(tips.head())
ax = sns.violinplot(x=tips["total_bill"])
plt.show()
# group by day
ax = sns.violinplot(x="day", y="total_bill", data=tips, palette="muted")
plt.show()
# group by day and color by time
ax = sns.violinplot(x="day", y="total_bill", hue="time", data=tips,
                    palette="muted", split=True)

# Pairwise scatterplots.
g = sns.PairGrid(salary, hue="management")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
ax = g.add_legend()
plt.show()
# I'm not clear on what that last one is showing me.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:49:15 2020

@author: dustan
"""
import pandas as pd
import matplotlib.pyplot as plt


url = 'https://raw.github.com/neurospin/pystatsml/master/datasets/salary_table.csv'
salary = pd.read_csv(url)
df = salary # ok, book. Whatever.

colors = colors_edu = {'Bachelor':'r', 'Master':'g', 'Ph.D':'blue'}
plt.scatter(df['experience'], df['salary'], c=df['education'].apply(
    lambda x: colors[x]), s=100)
## Figure size
plt.figure(figsize=(6,5))
## Define colors / sumbols manually
symbols_manag = dict(Y='*', N='.')
colors_edu = {'Bachelor':'r', 'Master':'g', 'Ph.D':'blue'}
## group by education x management => 6 groups
for values, d in salary.groupby(['education','management']):
    edu, manager = values
    plt.scatter(d['experience'], d['salary'], marker=symbols_manag[manager],
                color=colors_edu[edu], s=150, label=manager+"/"+edu)
## Set labels
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend(loc=4) # lower right
plt.show()

# =============================================================================
#     
#     df
#     Out[6]: 
#         salary  experience education management
#     0    13876           1  Bachelor          Y
#     1    11608           1      Ph.D          N
#     2    18701           1      Ph.D          Y
#     3    11283           1    Master          N
#     4    11767           1      Ph.D          N
#     5    20872           2    Master          Y
#     6    11772           2    Master          N
#     7    10535           2  Bachelor          N
#     8    12195           2      Ph.D          N
#     9    12313           3    Master          N
#     10   14975           3  Bachelor          Y
#     11   21371           3    Master          Y
#     12   19800           3      Ph.D          Y
#     13   11417           4  Bachelor          N
#     14   20263           4      Ph.D          Y
#     15   13231           4      Ph.D          N
#     16   12884           4    Master          N
#     17   13245           5    Master          N
#     18   13677           5      Ph.D          N
#     19   15965           5  Bachelor          Y
#     20   12336           6  Bachelor          N
#     21   21352           6      Ph.D          Y
#     22   13839           6    Master          N
#     23   22884           6    Master          Y
#     24   16978           7  Bachelor          Y
#     25   14803           8    Master          N
#     26   17404           8  Bachelor          Y
#     27   22184           8      Ph.D          Y
#     28   13548           8  Bachelor          N
#     29   14467          10  Bachelor          N
#     30   15942          10    Master          N
#     31   23174          10      Ph.D          Y
#     32   23780          10    Master          Y
#     33   25410          11    Master          Y
#     34   14861          11  Bachelor          N
#     35   16882          12    Master          N
#     36   24170          12      Ph.D          Y
#     37   15990          13  Bachelor          N
#     38   26330          13    Master          Y
#     39   17949          14    Master          N
#     40   25685          15      Ph.D          Y
#     41   27837          16    Master          Y
#     42   18838          16    Master          N
#     43   17483          16  Bachelor          N
#     44   19207          17    Master          N
#     45   19346          20  Bachelor          N
# =============================================================================

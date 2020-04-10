#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:37:29 2020

@author: dustan
"""

import pandas as pd


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
    